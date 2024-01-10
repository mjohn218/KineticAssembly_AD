from KineticAssembly_AD.vectorized_rxn_net_KinSim import VectorizedRxnNet_KinSim
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

class VecKinSim:
    """
    Run a vectorized deterministic simulation. All data and parameters are represented as
    Torch Tensors, allowing for gradients to be tracked. This simulator was designed to
    fill three primary requirements.
        - The simulation must be fully differentiable.
    """

    def __init__(self, net: VectorizedRxnNet_KinSim,
                 runtime: float,
                 device='cuda:0',rate_step=False):
        """

        Args:
            net: The reaction network to run the simulation on.
            runtime: Length (in seconds) of the simulation.

        """
        if torch.cuda.is_available() and "cpu" not in device:
            self.dev = torch.device(device)
            print("Using " + device)
        else:
            self.dev = torch.device("cpu")
            print("Using CPU")

        if type(net) is ReactionNetwork:
            self.rn = VectorizedRxnNet_KinSim(net, dev=self.dev)
        else:
            self.rn = net
        self.use_energies = self.rn.is_energy_set
        self.runtime = runtime
        self.observables = self.rn.observables
        self._constant = 1.
        self.avo = Tensor([6.022e23])
        self.steps = []
        self.flux_vs_time = self.rn.flux_vs_time
        self.net_flux = dict()
        self.switch=False
        self.uid_flux = torch.zeros(1,2*self.rn.reaction_network._rxn_count)
        self.mod_start=-1
        self.cur_time=0
        self.titration_end_conc=self.rn.titration_end_conc
        self.tit_stop_count=0
        self.titrationBool=False
        self.rate_step=rate_step
        self.rate_step_array = []



    def simulate(self, optim='yield',node_str=None,verbose=False,switch=False,switch_time=0,switch_rates=None,corr_rxns=[[0],[1]],conc_scale=1.0,mod_factor=1.0,conc_thresh=1e-5,mod_bool=True,yield_species=-1,store_interval=-1,change_cscale_tit=False,max_thresh=0.99):
        """
        modifies reaction network
        :return:
        """
        cur_time = 0
        prev_time=0
        self.cur_time=Tensor([0.])
        cutoff = 100000000
        mod_flag = True
        n_steps=0

        values = psutil.virtual_memory()
        # if torch.cuda.is_available() and "cpu" not in self.dev:
        #     print("Free: ",torch.cuda.mem_get_info()[0]/(1024*1024*1024))
        #     print("Used: ",torch.cuda.mem_get_info()[1]/(1024*1024*1024))
        print("Start of simulation: memory Used: ",values.percent)
        if optim=='time':
            print("Time based Optimization")

        # update observables
        max_poss_yield = torch.min(self.rn.copies_vec[:self.rn.num_monomers].clone()).to(self.dev)

        if self.rn.max_subunits !=-1:
            max_poss_yield = max_poss_yield/self.rn.max_subunits
            if verbose:
                print("Max Poss Yield: ",max_poss_yield)
        t95_flag=True
        t85_flag=True
        t50_flag=True
        t99_flag=True
        t85=-1
        t95=-1
        t50=-1
        t99=-1

        if self.rn.boolCreation_rxn:

            creation_amount={node:0 for node in self.rn.creation_rxn_data.keys()}
            if self.titration_end_conc!=-1:
                self.titrationBool=True
                max_poss_yield = self.titration_end_conc
            else:
                self.titrationBool=False
        if self.rn.chap_is_param:
            mask = torch.ones([len(self.rn.copies_vec[:self.rn.num_monomers])],dtype=bool)
            for species,uids in self.rn.chap_uid_map.items():
                mask[species]=False
            max_poss_yield = torch.min(self.rn.copies_vec[:self.rn.num_monomers][mask].clone()).to(self.dev)

        l_k = self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec, self._constant)
        if verbose:
            print("Simulation rates: ",torch.exp(l_k))

        while cur_time < self.runtime:
            conc_counter=1
            l_conc_prod_vec = self.rn.get_log_copy_prod_vector()
            if self.rn.boolCreation_rxn:

                array_dim = 2*len(self.rn.kon)-len(self.rn.creation_rxn_data)-len(self.rn.destruction_rxn_data)
                activator_arr = torch.ones((array_dim)).double()
                for node,values in self.rn.creation_rxn_data.items():
                    # self.rn.kon[self.rn.optim_rates[r]] = self.activate_titration(self.rn.params_kon[r])

                    end_time = self.rn.titration_time_map[values['uid']]
                    # if n_steps==1:
                    #     print("End TIME: ",end_time)
                    activator_arr[values['uid']] = self.activate_titration(values['uid'])
                l_rxn_rates = l_conc_prod_vec + l_k + torch.log(activator_arr)
                if not self.titrationBool and change_cscale_tit:
                    conc_scale = 1
                    change_cscale_tit=False
            else:
                l_rxn_rates = l_conc_prod_vec + l_k
            # l_rxn_rates = l_conc_prod_vec + l_k

            l_total_rate = torch.logsumexp(l_rxn_rates, dim=0)

            l_step = 0 - l_total_rate
            rate_step = torch.exp(l_rxn_rates + l_step)

            delta_copies = torch.matmul(self.rn.M, rate_step)*conc_scale


            if (torch.min(self.rn.copies_vec + delta_copies) < 0):
                if mod_bool:

                    temp_copies = self.rn.copies_vec + delta_copies
                    mask_neg = temp_copies<0

                    zeros = torch.zeros([len(delta_copies)],device=self.dev).double()
                    neg_species = torch.where(mask_neg,delta_copies,zeros)   #Get delta copies of all species that have neg copies
                    # print(neg_species)

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

            initial_monomers = self.rn.initial_copies
            min_copies = torch.ones(self.rn.copies_vec.shape, device=self.dev) * np.inf
            min_copies[0:initial_monomers.shape[0]] = initial_monomers
            self.rn.copies_vec = torch.max(self.rn.copies_vec + delta_copies, torch.zeros(self.rn.copies_vec.shape,
                                                                                          dtype=torch.double,
                                                                                          device=self.dev))




            # print("Final copies: ", self.rn.copies_vec)
            # values = psutil.virtual_memory()
            # print("Memory Used: ",values.percent)


            step = torch.exp(l_step)
            if self.rate_step:
                self.rate_step_array.append(rate_step)

            if self.rn.boolCreation_rxn:
                for node,data in self.rn.creation_rxn_data.items():
                    cr_rid = data['uid']
                    curr_path_contri = rate_step[cr_rid].detach().numpy()
                    creation_amount[node]+=  np.sum(curr_path_contri)*conc_scale



            # print("Full step: ",step)
            if cur_time + step*conc_scale > self.runtime:
                # print("Current time: ",cur_time)
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
                print("Next time: ",cur_time + step*conc_scale)
                # print("Curr_time:",cur_time)
                if verbose:
                    # print("Mass Conservation T: ",self.rn.copies_vec[4]+self.rn.copies_vec[16])
                    print("Final Conc Scale: ",conc_scale)
                    print("Number of steps: ", n_steps)
                    print("Next time larger than simulation runtime. Ending simulation.")
                    values = psutil.virtual_memory()
                    print("Memory Used: ",values.percent)
                    print("RAM Usage (GB): ",values.used/(1024*1024*1024))



                print("Max Possible Yield: ",max_poss_yield)


            cur_time = cur_time + step*conc_scale
            self.cur_time = cur_time
            n_steps+=1

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

            #Storing observables
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

            if self.rn.copies_vec[yield_species]/max_poss_yield > max_thresh:
                print("Reached max yield possible")
                if verbose:
                    # print("Mass Conservation T: ",self.rn.copies_vec[4]+self.rn.copies_vec[16])
                    print("Final Conc Scale: ",conc_scale)
                    print("Number of steps: ", n_steps)
                    print("Next time larger than simulation runtime. Ending simulation.")
                    values = psutil.virtual_memory()
                    print("Memory Used: ",values.percent)
                    print("RAM Usage (GB): ",values.used/(1024*1024*1024))
                break


            if len(self.steps) > cutoff:
                print("WARNING: sim was stopped early due to exceeding set max steps", sys.stderr)
                break
            if n_steps%1000==0:
                if verbose:
                    # values = psutil.virtual_memory()
                    # print("Memory Used: ",values.percent)
                    # print("RAM Usage (GB): ",values.used/(1024*1024*1024))
                    print("Current Time: ",cur_time)
                    # if torch.cuda.is_available() and "cpu" not in device:
                    #     print("Free: ",torch.cuda.mem_get_info()[0]/(1024*1024*1024))
                    #     print("Used: ",torch.cuda.mem_get_info()[1]/(1024*1024*1024))
            if self.rn.chaperone:
                total_complete = self.rn.copies_vec[yield_species]/max_poss_yield
                # dimer_yield = self.rn.copies_vec[yield_species]/max_poss_yield
                # dimer_yields_arr = torch.zeros([len(self.rn.optimize_species['substrate'])],requires_grad=True)
                # chap_species_arr = torch.zeros([len(self.rn.optimize_species['enz-subs'])],requires_grad=True)

                dimer_yield_sum=0
                chap_species_sum = 0

                dimer_max_yields_arr= []
                chap_max_yields_arr = []
                for s_iter in range(len(self.rn.optimize_species['substrate'])):
                    dimer_yield_sum+= self.rn.copies_vec[self.rn.optimize_species['substrate'][s_iter]]/max_poss_yield
                    dim_indx = np.argmax(self.rn.observables[self.rn.optimize_species['substrate'][s_iter]][1])
                    dimer_max_yields_arr.append(self.rn.observables[self.rn.optimize_species['substrate'][s_iter]][1][dim_indx]/max_poss_yield)

                for s_iter in range(len(self.rn.optimize_species['enz-subs'])):
                    chap_species_sum+= self.rn.copies_vec[self.rn.optimize_species['enz-subs'][s_iter]]/max_poss_yield
                    chap_indx = np.argmax(self.rn.observables[self.rn.optimize_species['enz-subs'][s_iter]][1])
                    chap_max_yields_arr.append(self.rn.observables[self.rn.optimize_species['enz-subs'][s_iter]][1][chap_indx]/max_poss_yield)

            values = psutil.virtual_memory()
            mem = values.available / (1024.0 ** 3)
            if mem < .5:
                # kill program if it uses to much ram
                total_complete = self.rn.copies_vec[yield_species]/max_poss_yield
                final_yield = total_complete
                print("Killing Simulation because too much RAM being used.")
                print(values.available,mem)
                return(final_yield.to(self.dev),(t50,t85,t95,t99))

        if self.rn.boolCreation_rxn:
            all_amounts = np.array(list(creation_amount.values()))
            print(all_amounts)
            total_complete = self.rn.copies_vec[yield_species]/np.min(all_amounts)
            unused_monomer = (np.min(all_amounts) - self.rn.copies_vec[yield_species])/np.min(all_amounts)
        else:
            total_complete = self.rn.copies_vec[yield_species]/max_poss_yield
        final_yield = total_complete

        if verbose:
            print("Final Yield: ", final_yield)



        return(final_yield.to(self.dev),(t50,t85,t95,t99))


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

    def activate_titration(self,rid=0):
        k_new=1e-6
        # el = torch.nn.functional.ELU(k_new)
        # print(el)
        end_time = self.rn.titration_time_map[rid]
        if self.titrationBool and (end_time < self.cur_time.item()):
            print("Ending Titration!")
            # print("Titration Map : ",self.rn.titration_end_time)
            # self.tit_stop_count+=1
            # print("Stop COunt= ",self.tit_stop_count)
            self.titrationBool=False

        delta_t = Tensor([end_time]) - self.cur_time
        # print("Delta t : ",delta_t)
        # return((1/delta_t)*(F.relu(delta_t)))
        # if not self.titrationBool:
        #     print("New rate: ",(1/delta_t)*(el(delta_t)))
        titration_mod = (1/delta_t)*(torch.nn.functional.elu(delta_t,alpha=k_new))
        # print(titration_mod)
        return(titration_mod)
