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

    def __init__(self, net: VectorizedRxnNet,
                 runtime: float,
                 device='cuda:0',calc_flux=False,rate_step=False):
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
            self.rn = VectorizedRxnNet(net, dev=self.dev)
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
        self.calc_flux=calc_flux
        self.rate_step=rate_step
        self.rate_step_array = []
        self.mod_start=-1
        self.cur_time=0
        self.titration_end_conc=self.rn.titration_end_conc
        self.tit_stop_count=0
        self.titrationBool=False
        self.gradients =[]


        if self.rn.rxn_coupling or self.rn.coupling:
            self.coupled_kon = torch.zeros(len(self.rn.kon), requires_grad=True).double()


    def simulate(self, optim='yield',node_str=None,verbose=False,switch=False,switch_time=0,switch_rates=None,corr_rxns=[[0],[1]],conc_scale=1.0,mod_factor=1.0,conc_thresh=1e-5,mod_bool=True,yield_species=-1,store_interval=-1,change_cscale_tit=False):
        """
        modifies reaction network
        :return:
        """
        cur_time = 0
        prev_time=0
        self.cur_time=Tensor([0.])
        cutoff = 10000000
        mod_flag = True
        n_steps=0

        values = psutil.virtual_memory()
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

        if self.rn.coupling:
            #new_kon = torch.zeros(len(self.rn.kon), requires_grad=True).double()
            # print("Coupling")
            if self.rn.partial_opt:
                for i in range(len(self.rn.kon)):
                    if i in self.rn.rx_cid.keys():
                        all_rates=[]
                        for rate in self.rn.rx_cid[i]:
                            if rate in self.rn.optim_rates:
                                all_rates.append(self.rn.params_kon[self.rn.coup_map[rate]])
                            else:
                                if self.rn.slow_rates is not None and rate in self.rn.slow_rates:
                                    all_rates.append(torch.mean(self.rn.params_kon)/self.rn.slow_ratio)
                                else:
                                    all_rates.append(self.rn.kon[rate])
                        self.coupled_kon[i] = max(all_rates)
                    else:
                        if i in self.rn.optim_rates:
                            self.coupled_kon[i] = self.rn.params_kon[self.rn.coup_map[i]]
                        else:
                            if (self.rn.slow_rates is not None) and (i in self.rn.slow_rates):
                                # print("Enter:")       #Can be replaced later so that the RN figures out by iteself which are fast  interfaces and which are slow.
                                self.coupled_kon[i] = torch.mean(self.rn.params_kon)/self.rn.slow_ratio
                            else:
                                self.coupled_kon[i] = self.rn.kon[i]
                print("SLow rates: ",self.coupled_kon[self.rn.slow_rates])
                l_k = self.rn.compute_log_constants(self.coupled_kon,self.rn.rxn_score_vec, self._constant)


            else:
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
        elif self.rn.partial_opt:
            # local_kon = torch.zeros(len(self.kon),requires_grad=True).double()
            for r in range(len(self.rn.params_kon)):
                # print("is_leaf: ",self.rn.kon[r].is_leaf, "is_grad: ",self.rn.kon[r].requires_grad)
                self.rn.kon[self.rn.optim_rates[r]] = self.rn.params_kon[r]

            l_k = self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec, self._constant)
        else:
            l_k = self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec, self._constant)
            if verbose:
                print("Simulation rates: ",torch.exp(l_k))

        while cur_time < self.runtime:
            conc_counter=1
            l_conc_prod_vec = self.rn.get_log_copy_prod_vector()
            # if self.rn.boolCreation_rxn:
                # l_conc_prod_vec[-1]=torch.log(torch.pow(Tensor([0]),Tensor([1])))
            # print("Prod Conc: ",l_conc_prod_vec)
            if self.rn.boolCreation_rxn:

                array_dim = 2*len(self.rn.kon)-len(self.rn.creation_rxn_data)-len(self.rn.destruction_rxn_data)
                activator_arr = torch.ones((array_dim),requires_grad=True).double()
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
            # print("Rates: ",l_rxn_rates)
            l_total_rate = torch.logsumexp(l_rxn_rates, dim=0)
            #l_total_rate = l_total_rate + torch.log(torch.min(self.rn.copies_vec))
            l_step = 0 - l_total_rate
            rate_step = torch.exp(l_rxn_rates + l_step)
            # conc_scale = 1  #Units uM
            # if torch.min(self.rn.copies_vec[torch.nonzero(self.rn.copies_vec)]) < conc_scale:
            #     conc_scale = torch.min(self.rn.copies_vec[torch.nonzero(self.rn.copies_vec)]).item()

            delta_copies = torch.matmul(self.rn.M, rate_step)*conc_scale




            #Calculate rxn_flux
            if self.calc_flux:
                rxn_flux = self.rn.get_reaction_flux()





            if (torch.min(self.rn.copies_vec + delta_copies) < 0):
                # temp_copies = self.rn.copies_vec + delta_copies
                # min_idx = torch.argmin(temp_copies)
                # min_value = self.rn.copies_vec[min_idx]
                #
                # delta_copy = torch.matmul(self.rn.M[min_idx,:],rate_step)
                # modulator = mod_factor*min_value/abs(delta_copy)
                #
                # print("Taking smaller timestep")
                # print("Previous slope: ",delta_copies/(torch.exp(l_step)*conc_scale))
                # # print(self.rn.copies_vec + delta_copies)
                # print("Previous rate step: ",rate_step)
                #
                # #Take a smaller time step
                # # l_total_rate = l_total_rate - torch.log(torch.min(self.rn.copies_vec[torch.nonzero(self.rn.copies_vec)]))
                # print("Modulator: ",modulator)
                # l_total_rate = l_total_rate - torch.log(modulator)
                # l_step = 0 - l_total_rate
                # rate_step = torch.exp(l_rxn_rates + l_step)
                # delta_copies = torch.matmul(self.rn.M, rate_step)
                #
                # print("New rate step: ",rate_step)

                if conc_scale>conc_thresh:
                    conc_scale = conc_scale/mod_factor
                    # conc_scale = torch.min(self.rn.copies_vec[torch.nonzero(self.rn.copies_vec)]).item()
                    # print("New Conc Scale: ",conc_scale)
                    delta_copies = torch.matmul(self.rn.M, rate_step)*conc_scale
                    # print("New Delta Copies: ",delta_copies)
                elif mod_bool:
                    # print("Previous rate step : ",rate_step,torch.sum(rate_step))
                    # print("Old copies : ",self.rn.copies_vec)
                    # print("Old delta copies: ",delta_copies)
                    # print("Changing conc. scale")
                    temp_copies = self.rn.copies_vec + delta_copies
                    mask_neg = temp_copies<0
                    # max_delta = torch.max(delta_copies[mask_neg])

                    zeros = torch.zeros([len(delta_copies)],dtype=torch.double,device=self.dev)
                    neg_species = torch.where(mask_neg,delta_copies,zeros)   #Get delta copies of all species that have neg copies
                    # print("Neg species: ",neg_species)

                    min_value = self.rn.copies_vec

                    modulator = torch.abs(neg_species)/min_value
                    min_modulator = torch.max(modulator[torch.nonzero(modulator)])   #Taking the smallest modulator
                    # min_idx = torch.argmin(temp_copies)
                    # min_value = self.rn.copies_vec[min_idx]
                    # delta_copy = torch.matmul(self.rn.M[sp_indx,:],rate_step)

                    # modulator = min_value/abs(delta_copy)
                    # print(min_value)
                    # print("Modulator: ",modulator)
                    # print("SPecies: ",sp_indx)
                    # print("Modulator: ",modulator)
                    l_total_rate = l_total_rate - torch.log(0.99/min_modulator)
                    l_step = 0 - l_total_rate
                    rate_step = torch.exp(l_rxn_rates + l_step)
                    delta_copies = torch.matmul(self.rn.M, rate_step)*conc_scale
                    # print("New rate step : ",rate_step,torch.sum(rate_step))
                    # print("New copies : ",self.rn.copies_vec + delta_copies)
                    # print("New delta copies: ",delta_copies)
                    # print("Current Time Step: ",torch.exp(l_step)*conc_scale)
                    # print("Copies : ",self.rn.copies_vec[-1])
                    # print("Delta_Copies: ",delta_copies[-1])

                    # min_val = torch.min(self.rn.copies_vec[torch.nonzero(self.rn.copies_vec)]).item()
                    # conc_scale = min_val/mod_factor
                    # delta_copies = torch.matmul(self.rn.M, rate_step)*conc_scale
                    #
                    # print("New Conc scale = ",conc_scale)
                    # print("Copies : ",self.rn.copies_vec)
                    # print("Current time: ",cur_time)
                    if mod_flag:
                        self.mod_start=cur_time
                        mod_flag=False
            # elif self.rn.boolCreation_rxn:
            #     if conc_scale > (torch.min(self.rn.copies_vec)):
            #         print("Warning!!! Conc_scale greater than min conc.")
            #         print("Current Time: ",cur_time)
            #         print(self.rn.copies_vec)
            #
            #         if len(torch.nonzero(self.rn.copies_vec))!=0:
            #
            #             conc_scale = torch.min(self.rn.copies_vec[torch.nonzero(self.rn.copies_vec)])
            #             print("New conc_scale: ",conc_scale)
            #     elif conc_scale < (torch.min(self.rn.copies_vec)):
            #         min_copies = torch.min(self.rn.copies_vec).detach().numpy()
            #         power = math.floor(np.log10(min_copies))
            #         conc_scale = 1*10**(power)

                    # print("Conc Scale INCREASED: ",conc_scale)



            # print("-----------------------------")
            # print("Total number of A: ", self.rn.copies_vec[0]+self.rn.copies_vec[4]+self.rn.copies_vec[5]+self.rn.copies_vec[6]) #For trimer model
            # print("Total number of A: ", self.rn.copies_vec[0]+self.rn.copies_vec[2]+2*self.rn.copies_vec[3]+2*self.rn.copies_vec[4]) #For repeat model
            # print("Total number of A: ", self.rn.copies_vec[0]+self.rn.copies_vec[1]*2+3*self.rn.copies_vec[2])   #For homotrimer model
            # print("Total COpies of A: ", self.rn.copies_vec[0]+self.rn.copies_vec[3]+self.rn.copies_vec[4]+self.rn.copies_vec[7]+2*(self.rn.copies_vec[5]+self.rn.copies_vec[8]+self.rn.copies_vec[9]+self.rn.copies_vec[10]))
            # print("Prod of conc.: ", torch.exp(l_conc_prod_vec))
            # print(l_k)
            # print("Rxn rates: ", torch.exp(l_rxn_rates))
            # print("Total rxn rate: ",l_total_rate)
            # print("Rate step: ",rate_step)
            # print("Copies: ",self.rn.copies_vec)
            #
            # print("Next step size: ",torch.exp(l_step))
            # print("Sum of steps: ", torch.sum(rate_step))
            # print("Matrix: ",self.rn.M)
            # print("delta_copies: ", delta_copies)
            # print("A delta: ",torch.mul(self.rn.M[0,:],rate_step)*conc_scale)
            # print("AB delta: ",torch.mul(self.rn.M[4,:],rate_step)*conc_scale)
            # print("ABT delta: ",torch.mul(self.rn.M[8,:],rate_step)*conc_scale)
            # print("ABC delta: ",torch.mul(self.rn.M[7,:],rate_step)*conc_scale)

            # print("Delta Conservation: ",delta_copies[0]+delta_copies[4]+delta_copies[5]+delta_copies[7]+delta_copies[8])
            # mass_cons = self.rn.copies_vec[0]+self.rn.copies_vec[4]+self.rn.copies_vec[5]+self.rn.copies_vec[7]+self.rn.copies_vec[8]
            # print("Mass Conservation A: ",mass_cons)
            # if mass_cons > 100.01:
            #     break
            # print("Mass Conservation B: ",self.rn.copies_vec[1]+self.rn.copies_vec[4]+self.rn.copies_vec[6]+self.rn.copies_vec[7]+self.rn.copies_vec[8])
            # print("Mass Conservation C: ",self.rn.copies_vec[2]+self.rn.copies_vec[6]+self.rn.copies_vec[5]+self.rn.copies_vec[7])
            # print("Mass Conservation T: ",self.rn.copies_vec[3]+self.rn.copies_vec[8])
            # print("SUM: ",torch.sum(delta_copies))
            #
            # print("Current time: ",cur_time)
            # Prevent negative copy cumbers explicitly (possible due to local linear approximation)
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


            #Calculating total amount of each species titrated. Required for calculating yield
            if self.rn.boolCreation_rxn:
                for node,data in self.rn.creation_rxn_data.items():
                    cr_rid = data['uid']
                    curr_path_contri = rate_step[cr_rid].detach().numpy()
                    creation_amount[node]+=  np.sum(curr_path_contri)*conc_scale

            # print("Full step: ",step)
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

                    # if self.rn.boolCreation_rxn:
                    #     print("MASS Conservation: ")
                        # print("Molecules added: ",creation_amount[0],creation_amount[1],creation_amount[2])
                    #     print("Total amount of A in system: ",self.rn.copies_vec[0]+self.rn.copies_vec[3]+self.rn.copies_vec[4]+self.rn.copies_vec[-1])

                # for obs in self.rn.observables.keys():
                #     try:
                #         self.rn.observables[obs][1].pop()
                #     except IndexError:
                #         print('bkpt')
                # break

            #Add a switching criteria. Jump rates to optimized value
            # if switch and (cur_time + step > switch_time):
            #     print("Rates switched")
            #     self.switch = True
            #     # print("New rates: ",self.rn.kon)
            #     l_k = self.rn.compute_log_constants(switch_rates, self.rn.rxn_score_vec, self._constant)
            #     print("Time: ", cur_time+step)
            #     print(l_k)
            #     switch=False

            cur_time = cur_time + step*conc_scale
            self.cur_time = cur_time
            n_steps+=1

            #Only for testing puprose in CHaperone
            # for c in range(len(self.rn.chap_params)):
            #     self.rn.chap_params[c].grad = None
            # # nn.Module.zero_grad()
            # obj_yield = self.rn.copies_vec[yield_species]/max_poss_yield
            # self.gradients.append(torch.autograd.grad(obj_yield,self.rn.chap_params,retain_graph=True))

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
            if self.calc_flux:
                self.uid_flux = torch.cat((self.uid_flux,rxn_flux),0)

            if n_steps==1:
                prev_time = cur_time
            # print("Current time: ",cur_time)
            #Calculate the flux
            # self.net_flux = self.rn.calculate_total_flux()
            # for obs in self.rn.flux_vs_time.keys():
            #     try:
            #         self.flux_vs_time[obs][1].append(self.net_flux[self.flux_vs_time[obs][0]])
            #     except IndexError:
            #         print('bkpt')

            if len(self.steps) > cutoff:
                print("WARNING: sim was stopped early due to exceeding set max steps", sys.stderr)
                break
            if n_steps%10000==0:
                if verbose:
                    values = psutil.virtual_memory()
                    print("Memory Used: ",values.percent)
                    print("RAM Usage (GB): ",values.used/(1024*1024*1024))
                    print("Current Time: ",cur_time)
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


            #Old code when there was only one chap reaction
            # dimer_yield = self.rn.copies_vec[self.rn.optimize_species['substrate']]/max_poss_yield
            # chap_species = self.rn.copies_vec[self.rn.optimize_species['enz-subs']]/max_poss_yield
            #
            # dim_indx = np.argmax(self.rn.observables[self.rn.optimize_species['substrate']][1])
            # chap_indx = np.argmax(self.rn.observables[self.rn.optimize_species['enz-subs']][1])
            #
            # dimer_max_yield = self.rn.observables[self.rn.optimize_species['substrate']][1][dim_indx]/max_poss_yield
            # # print("Time of max DImer yield: ",self.steps[dim_indx])
            # chap_max_yield = self.rn.observables[self.rn.optimize_species['enz-subs']][1][chap_indx]/max_poss_yield

            print("Max Possible Yield: ",max_poss_yield)
            # for n in self.rn.network.nodes():
            # print("Gradient: ",torch.autograd.grad(total_complete,self.rn.chap_params))
        elif self.rn.boolCreation_rxn:
            all_amounts = np.array(list(creation_amount.values()))
            print(all_amounts)
            total_complete = self.rn.copies_vec[yield_species]/np.min(all_amounts)
            unused_monomer = (np.min(all_amounts) - self.rn.copies_vec[yield_species])/np.min(all_amounts)
        else:
            total_complete = self.rn.copies_vec[yield_species]/max_poss_yield

        # total_complete = torch.max(torch.DoubleTensor([self.rn.copies_vec[3],self.rn.copies_vec[4],self.rn.copies_vec[5]]))
        # final_yield = torch.abs(0.66932 - (total_complete / max_poss_yield))
        # final_yield = total_complete/max_poss_yield
        final_yield = total_complete

        if verbose:
            print("Final Yield: ", final_yield)
        if optim=='flux_coeff':
            final_yield = self.calc_corr_coeff(corr_rxns)
            # print(final_yield)
            return(Tensor([final_yield]).to(self.dev),None)


        if optim == 'flux':
            if node_str != None:
                return(final_yield.to(self.dev),self.net_flux[node_str].to(self.dev))
            else:
                return(final_yield.to(self.dev),self.net_flux[list(self.net_flux.keys())[-1]].to(self.dev))
        else:
            # return (final_yield.to(self.dev),self.net_flux[list(self.net_flux.keys())[-1]].to(self.dev))
            if self.rn.boolCreation_rxn:
                if optim=='yield':
                    return(final_yield.to(self.dev),cur_time,unused_monomer.to(self.dev),(t50,t85,t95,t99))
                elif optim=='time':
                    return(final_yield.to(self.dev),t95,unused_monomer.to(self.dev),(t50,t85,t95,t99))
            elif self.rn.chaperone:
                return(final_yield.to(self.dev),dimer_yield_sum,chap_species_sum,dimer_max_yields_arr,chap_max_yields_arr,self.steps[-1],(t50,t85,t95,t99))
            else:
                return(final_yield.to(self.dev),(t50,t85,t95,t99))

    def simulate_wrt_expdata(self, optim='yield',node_str=None,verbose=False,conc_scale=1.0,mod_factor=1.0,conc_thresh=1e-5,mod_bool=True,yield_species=-1):
        """
        modifies reaction network
        :return:
        """
        cur_time = 0
        prev_time=0
        self.cur_time=Tensor([0.])
        cutoff = 10000000
        mod_flag = True
        n_steps=0
        conc_tensor=[]


        values = psutil.virtual_memory()
        print("Start of simulation: memory Used: ",values.percent)

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

        l_k = self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec, self._constant)
        if verbose:
            print("Simulation rates: ",torch.exp(l_k))

        while cur_time < self.runtime:
            conc_counter=1

            l_conc_prod_vec = self.rn.get_log_copy_prod_vector()

            l_rxn_rates = l_conc_prod_vec + l_k
            l_total_rate = torch.logsumexp(l_rxn_rates, dim=0)
            l_step = 0 - l_total_rate
            rate_step = torch.exp(l_rxn_rates + l_step)
            delta_copies = torch.matmul(self.rn.M, rate_step)*conc_scale

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
            initial_monomers = self.rn.initial_copies
            min_copies = torch.ones(self.rn.copies_vec.shape, device=self.dev) * np.inf
            min_copies[0:initial_monomers.shape[0]] = initial_monomers
            self.rn.copies_vec = torch.max(self.rn.copies_vec + delta_copies, torch.zeros(self.rn.copies_vec.shape,
                                                                                          dtype=torch.double,
                                                                                          device=self.dev))

            step = torch.exp(l_step)
            if self.rate_step:
                self.rate_step_array.append(rate_step)


            #Calculating total amount of each species titrated. Required for calculating yield
            if self.rn.boolCreation_rxn:
                for node,data in self.rn.creation_rxn_data.items():
                    cr_rid = data['uid']
                    curr_path_contri = rate_step[cr_rid].detach().numpy()
                    creation_amount[node]+=  np.sum(curr_path_contri)*conc_scale

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

            if n_steps>1:
                self.steps.append(cur_time.item())
                for obs in self.rn.observables.keys():
                    try:
                        self.rn.observables[obs][1].append(self.rn.copies_vec[int(obs)].item())
                        #self.flux_vs_time[obs][1].append(self.net_flux[self.flux_vs_time[obs][0]])
                    except IndexError:
                        print('bkpt')

                prev_time=cur_time

                # print(conc_tensor.shape)
                # print(conc_tensor)
                # print(self.rn.copies_vec[yield_species])
                # conc_tensor = torch.cat((conc_tensor,self.rn.copies_vec[yield_species]),dim=0)

            if n_steps==1:
                prev_time = cur_time
                # conc_tensor = torch.Tensor([self.rn.copies_vec[yield_species]])

            if len(self.steps) > cutoff:
                print("WARNING: sim was stopped early due to exceeding set max steps", sys.stderr)
                break
            if n_steps%10000==0:
                if verbose:
                    values = psutil.virtual_memory()
                    print("Memory Used: ",values.percent)
                    print("RAM Usage (GB): ",values.used/(1024*1024*1024))
                    print("Current Time: ",cur_time)

            total_complete = self.rn.copies_vec[yield_species]/max_poss_yield
            conc_tensor.append(self.rn.copies_vec[yield_species].clone())

        final_yield = total_complete


        if verbose:
            print("Final Yield: ", final_yield)

        return(final_yield.to(self.dev),conc_tensor,(t50,t85,t95,t99))

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

    def calc_corr_coeff(self,rid):
        flux_data=self.uid_flux.detach().numpy()[:-1,:]
        total_coeff = 0
        total_lag = 0
        print(rid)
        for i in range(len(rid[0])):
            x=flux_data[:,rid[0][i]]
            y=flux_data[:,rid[1][i]]
            # print(x)
            corr_array = signal.correlate(x,y,mode='full')
            lag1 = np.argmax(corr_array)-np.floor(corr_array.shape[0]/2)
            coeff = np.corrcoef(x,y,rowvar=False)

            # print(coeff)
            total_coeff += coeff[0,1]
            total_lag+=lag1

        return(total_coeff/len(rid[0]))
        # return(total_lag/len(rid[0]))

    def activate_titration(self,rid=0):
        k_new=1e-6
        el = torch.nn.ELU(k_new)
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
        return((1/delta_t)*(el(delta_t)))
