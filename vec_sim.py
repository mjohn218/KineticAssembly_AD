from KineticAssembly_AD.vectorized_rxn_net import VectorizedRxnNet
from KineticAssembly_AD import ReactionNetwork
import numpy as np

from torch import DoubleTensor as Tensor
from torch.nn import functional as F
import torch
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import random
from scipy import signal
import sys
import math
import psutil
from torch import nn

def _make_finite(t):
    temp = t.clone()
    temp[t == -np.inf] = -2. ** 32.
    temp[t == np.inf] = 2. ** 32.
    return temp


class VecSim:
    """
    Run a vectorized deterministic simulation. All data and parameters are represented as
    Torch Tensors, allowing for gradients to be tracked. This simulator was designed to
    fill three primary requirements.
        - The simulation must be fully differentiable.
    """

    def __init__(self,
                 net: VectorizedRxnNet,
                 runtime: float,
                 device='cuda:0',
                 calc_flux: bool = False,
                 rate_step: bool = False):
        """
        param VectorizedRxnNet net: The reaction network to run the simulation on
        param float runtime: Length (in seconds) of the simulation.
        param device: The device to run the simulation on
        param bool calc_flux: # TODO: What does this do?
        param bool rate_step: # TODO: What does this do?
        """

        # Choose device on which to simulate
        if torch.cuda.is_available() and "cpu" not in device:
            self.dev = torch.device(device)
            print("Using " + device)
        else:
            self.dev = torch.device("cpu")
            print("Using CPU")

        # Ensure that rn is a VectorizedRxnNet, not just a ReactionNetwork object
        if type(net) is ReactionNetwork:
            self.rn = VectorizedRxnNet(net, dev=self.dev)
        else:
            self.rn = net

        self.use_energies = self.rn.is_energy_set
        self.runtime = runtime
        self.observables = self.rn.observables
        self._constant = 1.
        self.avo = Tensor([6.022e23])
        self.steps = []
        self.rate_step=rate_step
        self.rate_step_array = []
        self.mod_start=-1
        self.cur_time=0
        self.gradients = []

        # Consider coupled rate constants, if specified
        self.coupled_kon = None
        if self.rn.rxn_coupling or self.rn.coupling:
            self.coupled_kon = torch.zeros(len(self.rn.kon),
                                           requires_grad=True).double()


    def simulate(self,
                 optim='yield',
                 node_str=None,
                 verbose=False,
                 corr_rxns=[[0],[1]],
                 conc_scale=1.0,
                 mod_factor=1.0,
                 conc_thresh=1e-5,
                 mod_bool=True,
                 yield_species=-1,
                 store_interval=-1,
                 change_cscale_tit=False):
        """
        Updates the reaction network by simulating reactions over time
        :return:
        """
        cur_time = 0
        prev_time=0
        self.cur_time=Tensor([0.])
        cutoff = 10000000
        mod_flag = True
        n_steps=0

        values = psutil.virtual_memory()
        if verbose:
            print("Start of simulation: memory Used: ",values.percent)

        # Update observables
        max_poss_yield = torch.min(self.rn.copies_vec[:self.rn.num_monomers].clone()).to(self.dev)

        if self.rn.max_subunits !=-1:
            max_poss_yield = max_poss_yield/self.rn.max_subunits
            if verbose:
                print("Max Poss Yield:", max_poss_yield)
        t95_flag = t85_flag = t50_flag = t99_flag = True
        t85 = t95 = t50 = t99 = -1
        if self.rn.coupling:
            #new_kon = torch.zeros(len(self.rn.kon), requires_grad=True).double()
            # print("Coupling")
            for i in range(len(self.rn.kon)):
                # print(i)
                if i in self.rn.rx_cid.keys():
                    #new_kon[i] = 1.0
                    # self.coupled_kon[i] = max(self.rn.kon[rate] for rate in self.rn.rx_cid[i])
                    self.coupled_kon[i] = max(self.rn.params_kon[self.rn.coup_map[rate]] for rate in self.rn.rx_cid[i])
                    # print("Max rate for reaction %s chosen as %.3f" %(i,self.coupled_kon[i]))
                else:
                    # self.coupled_kon[i] = self.rn.kon[i]
                    self.coupled_kon[i] = self.rn.params_kon[self.rn.coup_map[i]]
            l_k = self.rn.compute_log_constants(self.coupled_kon,self.rn.rxn_score_vec, self._constant)

        elif self.rn.homo_rates:
            counter=0
            for k,rids in self.rn.rxn_class.items():
                for r in rids:
                    self.rn.kon[r] = self.rn.params_kon[counter]
                counter+=1
            l_k = self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec, self._constant)
        else:
            l_k = self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec, self._constant)
        if verbose:
            print("Simulation rates: ",torch.exp(l_k))

        while cur_time < self.runtime:
            conc_counter=1
            if n_steps > 100:
                with torch.no_grad():
                    l_conc_prod_vec = self.rn.get_log_copy_prod_vector()
            else:
                l_conc_prod_vec = self.rn.get_log_copy_prod_vector()


            l_rxn_rates = l_conc_prod_vec + l_k
            l_total_rate = torch.logsumexp(l_rxn_rates, dim=0)
            l_step = 0 - l_total_rate
            rate_step = torch.exp(l_rxn_rates + l_step)
            delta_copies = torch.matmul(self.rn.M, rate_step)*conc_scale

            # Prevent negative copy cumbers explicitly (possible due to local linear approximation)

            if (torch.min(self.rn.copies_vec + delta_copies) < 0):

                if conc_scale>conc_thresh:
                    conc_scale = conc_scale/mod_factor
                    delta_copies = torch.matmul(self.rn.M, rate_step)*conc_scale
                elif mod_bool:
                    temp_copies = self.rn.copies_vec + delta_copies
                    mask_neg = temp_copies<0

                    zeros = torch.zeros([len(delta_copies)],dtype=torch.double,device=self.dev)
                    neg_species = torch.where(mask_neg,delta_copies,zeros)   #Get delta copies of all species that have neg copies
                    min_value = self.rn.copies_vec
                    modulator = torch.abs(neg_species)/min_value
                    min_modulator = torch.max(modulator[torch.nonzero(modulator)])   #Taking the smallest modulator
                    l_total_rate = l_total_rate - torch.log(0.99/min_modulator)
                    l_step = 0 - l_total_rate
                    rate_step = torch.exp(l_rxn_rates + l_step)
                    delta_copies = torch.matmul(self.rn.M, rate_step)*conc_scale
                    if mod_flag:
                        self.mod_start=cur_time
                        mod_flag=False

            # initial_monomers = self.rn.initial_copies
            # min_copies = torch.ones(self.rn.copies_vec.shape, device=self.dev) * np.inf
            # min_copies[0:initial_monomers.shape[0]] = initial_monomers
            self.rn.copies_vec = torch.max(self.rn.copies_vec + delta_copies, torch.zeros(self.rn.copies_vec.shape,
                                                                                          dtype=torch.double,
                                                                                          device=self.dev))


            step = torch.exp(l_step)
            if self.rate_step:
                self.rate_step_array.append(rate_step)


            if cur_time + step*conc_scale > self.runtime:
                # print("Current time: ",cur_time)
                if optim=='time':
                    # print("Exceeding time",t95_flag)
                    if t95_flag:
                        #Yield has not yeached 95%
                        print("Yield has not reached 95 %. Increasing simulation time")
                        self.runtime=(cur_time + step*conc_scale)*2
                        continue
                if self.rn.copies_vec[yield_species]/max_poss_yield > 0.5 and t50_flag:
                    t50=cur_time
                    t50_flag=False
                if self.rn.copies_vec[yield_species]/max_poss_yield > 0.85 and t85_flag:
                    t85=cur_time
                    t85_flag=False
                if self.rn.copies_vec[yield_species]/max_poss_yield > 0.95 and t95_flag:
                    t95=cur_time
                    t95_flag=False
                if self.rn.copies_vec[yield_species]/max_poss_yield > 0.99 and t99_flag:
                    t99=cur_time
                    t99_flag=False
                # print("Next time: ",cur_time + step*conc_scale)
                # print("Curr_time:",cur_time)
                if verbose:
                    # print("Mass Conservation T: ",self.rn.copies_vec[4]+self.rn.copies_vec[16])
                    print("Final Conc Scale: ",conc_scale)
                    print("Number of steps: ", n_steps)
                    print("Next time larger than simulation runtime. Ending simulation.")
                    values = psutil.virtual_memory()
                    print("Memory Used: ",values.percent)
                    print("RAM Usage (GB): ",values.used/(1024*1024*1024))

            #Update Time
            cur_time = cur_time + step * conc_scale
            self.cur_time = cur_time
            n_steps += 1

            if self.rn.copies_vec[yield_species]/max_poss_yield > 0.5 and t50_flag:
                t50=cur_time
                t50_flag=False
            if self.rn.copies_vec[yield_species]/max_poss_yield > 0.85 and t85_flag:
                t85=cur_time
                t85_flag=False
            if self.rn.copies_vec[yield_species]/max_poss_yield > 0.95 and t95_flag:
                # print("95% yield reached: ",self.rn.copies_vec[yield_species]/max_poss_yield)
                t95=cur_time
                t95_flag=False
            if self.rn.copies_vec[yield_species]/max_poss_yield > 0.99 and t99_flag:
                t99=cur_time
                t99_flag=False

            if store_interval==-1 or n_steps<=1:
                self.steps.append(cur_time.item())
                for obs in self.rn.observables.keys():
                    try:
                        self.rn.observables[obs][1].append(self.rn.copies_vec[int(obs)].item())
                        #self.flux_vs_time[obs][1].append(self.net_flux[self.flux_vs_time[obs][0]])
                    except IndexError:
                        print('bkpt')
                prev_time=cur_time
            else:
                if n_steps>1:
                    if (cur_time/prev_time)>=store_interval:
                        self.steps.append(cur_time.item())
                        for obs in self.rn.observables.keys():
                            try:
                                self.rn.observables[obs][1].append(self.rn.copies_vec[int(obs)].item())
                                #self.flux_vs_time[obs][1].append(self.net_flux[self.flux_vs_time[obs][0]])
                            except IndexError:
                                print('bkpt')

                        prev_time=cur_time

            if n_steps==1:
                prev_time = cur_time
            if len(self.steps) > cutoff:
                print("WARNING: sim was stopped early due to exceeding set max steps", sys.stderr)
                break
            if n_steps % 10000 == 0:
                if verbose:
                    values = psutil.virtual_memory()
                    print("Memory Used:", values.percent)
                    print("RAM Usage (GB):", values.used / (1024 ** 3))
                    print("Current Time:", cur_time)
        total_complete = self.rn.copies_vec[yield_species]/max_poss_yield

        final_yield = total_complete

        if verbose:
            print("Final Yield: ", final_yield)

        return(final_yield.to(self.dev),(t50,t85,t95,t99))


    def reset(self,runtime=None):
        self.steps=[]
        self.observables=self.rn.observables
        self.rate_step_array=[]
        self.cur_time=0
        self.gradients =[]
        if runtime is not None:
            self.runtime = runtime

    def plot_observable(self,nodes_list, ax=None,flux=False,legend=True,seed=None,color_input=None,lw=1.0):
        t = np.array(self.steps)
        colors_list = list(mcolors.CSS4_COLORS.keys())
        random.seed(a=seed)
        if not flux:
            counter=0
            for key in self.observables.keys():

                if self.observables[key][0] in nodes_list:
                    data = np.array(self.observables[key][1])
                    if color_input is not None:
                        clr=color_input[counter]
                    else:
                        clr=random.choice(colors_list)
                    if not ax:
                        plt.plot(t, data, label=self.observables[key][0],color=clr,linewidth=lw)
                    else:
                        ax.plot(t, data, label=self.observables[key][0],color=clr,linewidth=lw)
                counter+=1
        else:
            for key in self.flux_vs_time.keys():
                if self.flux_vs_time[key][0] in nodes_list:
                    data2 = np.array(self.flux_vs_time[key][1])
                    #print(data2)
                    if not ax:
                        plt.plot(t, data2, label=self.flux_vs_time[key][0],color=random.choice(colors_list))
                    else:
                        ax.plot(t, data2, label=self.flux_vs_time[key][0],color=random.choice(colors_list))
        if legend:
            lgnd = plt.legend(loc='best')
            for i in range(len(lgnd.legendHandles)):
                lgnd.legendHandles[i]._sizes = [30]

        plt.ticklabel_format(style='sci',scilimits=(-3,3))
        plt.tick_params(axis='both',labelsize=14.0)
        f_dict = {'fontsize':14}
        plt.ylabel(r'Conc in $\mu M$',fontdict=f_dict)
        plt.xlabel('Time (s)',fontdict=f_dict)

    def observables_to_csv(self, out_path):
        data = {}
        for key in self.rn.observables:
            entry = self.rn.observables[key]
            data[entry[0]] = entry[1]
        df = pd.DataFrame(data)
        df.to_csv(out_path)

