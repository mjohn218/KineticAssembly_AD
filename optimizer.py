import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import psutil
from KineticAssembly_AD import VecSim
from KineticAssembly_AD import VectorizedRxnNet
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiplicativeLR
import random
import pandas as pd

class Optimizer:
    def __init__(self, reaction_network,
                 sim_runtime: float,
                 optim_iterations: int,
                 learning_rate: float,
                 device='cpu',
                 method='Adam',
                 lr_change_step=None,
                 gamma=None,
                 mom=0,
                 random_lr=False):

        # Load device for PyTorch (e.g. GPU or CPU)
        if torch.cuda.is_available() and "cpu" not in device:
            self.dev = torch.device(device)
            print("Using " + device)
        else:
            self.dev = torch.device("cpu")
            device = 'cpu'
            print("Using CPU")
        self._dev_name = device

        self.sim_class = VecSim
        if type(reaction_network) is not VectorizedRxnNet:
            try:
                self.rn = VectorizedRxnNet(reaction_network, dev=self.dev)
            except Exception:
                raise TypeError("Must be type ReactionNetwork or VectorizedRxnNetwork.")
        else:
            self.rn = reaction_network
        self.sim_runtime = sim_runtime
        param_itr = self.rn.get_params()

        if method == 'Adam':
            if self.rn.partial_opt:
                params_list=[]
                self.lr_group=[]
                print("Params: ",param_itr)
                for i in range(len(param_itr)):
                    # print("Learn Rate: ",learning_rate)
                    learn_rate = random.uniform(learning_rate,learning_rate*10)
                    params_list.append({'params':param_itr[i], 'lr':torch.mean(param_itr[i]).item()*learn_rate})
                    self.lr_group.append(learn_rate)
                self.optimizer = torch.optim.Adam(params_list)
            elif self.rn.chap_is_param:
                param_list = []
                for i in range(len(param_itr)):
                    lr_val = torch.mean(param_itr[i]).item()*learning_rate[i]
                    if lr_val>=torch.min(param_itr[i]).item()*0.1:
                        lr_val = torch.min(param_itr[i]).item()*1
                    param_list.append({'params':param_itr[i], 'lr':lr_val})
                self.optimizer = torch.optim.Adam(param_list)
            else:
                self.optimizer = torch.optim.Adam(param_itr, learning_rate)
        elif method =='RMSprop':
            if self.rn.chap_is_param:
                param_list = []
                # param_list2 = []

                for i in range(len(param_itr)):
                    lr_val = torch.mean(param_itr[i]).item()*learning_rate[i]
                    if lr_val>=torch.min(param_itr[i]).item()*0.1:
                        lr_val = torch.min(param_itr[i]).item()*1
                    param_list.append({'params':param_itr[i], 'lr':lr_val})
                self.optimizer = torch.optim.RMSprop(param_list,momentum=mom)
            else:
                if self.rn.partial_opt and not self.rn.coupling:
                    params_list=[]
                    self.lr_group=[]
                    print("Params: ",param_itr)
                    for i in range(len(param_itr)):
                        # print("Learn Rate: ",learning_rate)
                        if random_lr:
                            learn_rate = random.uniform(learning_rate,learning_rate*10)
                        else:
                            learn_rate = learning_rate[i]
                        params_list.append({'params':param_itr[i], 'lr':learn_rate})
                        self.lr_group.append(learn_rate)
                    self.optimizer = torch.optim.RMSprop(params_list,momentum=mom)
                else:
                    self.optimizer = torch.optim.RMSprop(param_itr, learning_rate)

        self.lr = learning_rate
        self.optim_iterations = optim_iterations
        self.sim_observables = []
        self.parameter_history = []
        self.yield_per_iter = []
        self.flux_per_iter = []
        self.is_optimized = False
        self.dt = None
        self.final_solns = []
        self.final_yields = []
        self.curr_time= []
        self.final_t50 = []
        self.final_t85 = []
        self.final_t95 = []
        self.final_t99 = []
        self.final_unused_mon = []
        self.dimer_max=[]
        self.chap_max=[]
        self.endtimes=[]
        if lr_change_step is not None:
            if gamma == None:
                gamma = 0.5
            # self.scheduler = StepLR(self.optimizer,step_size=lr_change_step,gamma=gamma)
            # self.scheduler = ReduceLROnPlateau(self.optimizer,'max',patience=30)
            if self.rn.assoc_is_param:
                if self.rn.partial_opt:
                    self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=[self.creat_lambda for i in range(len(self.rn.params_kon))])
                    self.lambda_ct = -1
                    self.gamma = gamma
                else:
                    self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=self.assoc_lambda)
            if self.rn.chap_is_param:
                self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=[self.lambda_c,self.lambda_k])
            self.lr_change_step = lr_change_step
        else:
            self.lr_change_step = None

    def assoc_lambda(self, opt_itr):
        new_lr = torch.min(self.rn.kon).item() * self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr / curr_lr)
    def creat_lambda(self, opt_itr):
        return(self.gamma)
    def lambda1(self, opt_itr):
        new_lr = torch.min(self.rn.params_k[0]).item() * self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr/curr_lr)
    def lambda2(self, opt_itr):
        new_lr = torch.min(self.rn.params_k).item() * self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr / curr_lr)
    def lambda_c(self, opt_itr):
        new_lr = torch.min(self.rn.chap_params[0]).item() * 100 * self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr / curr_lr)
    def lambda_k(self, opt_itr):
        new_lr = torch.min(self.rn.chap_params[1]).item() * self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][1]['lr']
        return(new_lr / curr_lr)
    def lambda5(self, opt_itr):
        new_lr = torch.min(self.rn.params_k[2]).item() * self.lr_group[2]
        curr_lr = self.optimizer.state_dict()['param_groups'][2]['lr']
        return(new_lr / curr_lr)

    def update_counter(self): # Currently does nothing
        lr_ct = 1
        # if self.counter == len(self.rn.params_k):
        #     self.counter=0

    def lambda_master(self, opt_itr):
        # update_counter()
        # print("***** LAMBDA MASTER:  {:d}*****".format(self.lambda_counter))
        # self.counter+=
        self.lambda_ct += 1
        return(torch.min(self.rn.params_k[self.lambda_ct % len(self.rn.params_k)]).\
               item() * self.lr_group[self.lambda_ct%len(self.rn.params_k)] / \
                self.optimizer.state_dict()['param_groups'][self.lambda_ct % len(self.rn.params_k)]['lr'])

    def plot_observable(self, iteration, nodes_list, ax=None):
        t = self.sim_observables[iteration]['steps']

        for key in self.sim_observables[iteration].keys():
            if key == 'steps':
                continue

            elif self.sim_observables[iteration][key][0] in nodes_list:
                data = np.array(self.sim_observables[iteration][key][1])
                if not ax:
                    plt.plot(t, data, label=self.sim_observables[iteration][key][0])
                else:
                    ax.plot(t, data, label=self.sim_observables[iteration][key][0])
        lgnd = plt.legend(loc='best')
        for i in range(len(lgnd.legendHandles)):
            lgnd.legendHandles[i]._sizes = [30]
        plt.title = 'Sim iteration ' + str(iteration)
        plt.show()

    def plot_yield(self,flux_bool=False):
        steps = np.arange(len(self.yield_per_iter))
        data = np.array(self.yield_per_iter, dtype=np.float)
        flux = np.array(self.flux_per_iter,dtype=np.float)
        plt.plot(steps, data,label='Yield')
        if flux_bool:
            plt.plot(steps,flux,label='Flux')
        #plt.ylim((0, 1))
        plt.title = 'Yield at each iteration'
        plt.show()

    def optimize(self,optim='yield',
                 node_str=None,
                 max_yield=0.5,
                 corr_rxns=[[1],[5]],
                 max_thresh=10,
                 lowvar=False,
                 conc_scale=1.0,
                 mod_factor=1.0,
                 conc_thresh=1e-5,
                 mod_bool=True,
                 verbose=False,
                 change_runtime=False,
                 yield_species=-1,
                 creat_yield=-1,
                 varBool=True,
                 chap_mode=1,
                 change_lr_yield=0.98,
                 var_thresh=10):
        print("Reaction Parameters before optimization: ")
        print(self.rn.get_params())

        print("Optimizer State:", self.optimizer.state_dict)
        calc_flux_optim = False
        if optim == 'flux_coeff':
            calc_flux_optim = True
        for i in range(self.optim_iterations):
            # Reset for new simulator
            self.rn.reset()

            if self.rn.boolCreation_rxn and change_runtime:
                #Change the runtime so that the simulation is stopped after a certain number of molecules have been dumped.
                final_conc = 100
                #Get current rates of dumping
                # min_rate = torch.min(self.rn.get_params()[0])
                rates = np.array(self.rn.get_params())
                titration_end = final_conc/rates

                titration_time_map = {v['uid'] : final_conc / v['k_on']
                                      for v in self.rn.creation_rxn_data.values()}
                for r in range(len(rates)):
                    titration_time_map[self.rn.optim_rates[r]]  = titration_end[r]
                self.rn.titration_time_map=titration_time_map
                # print("Titration Map : ",self.rn.titration_end_time)
                new_runtime = np.max(list(titration_time_map.values())) + self.sim_runtime
                print("New Runtime:", new_runtime)
                sim = self.sim_class(self.rn,
                                     new_runtime,
                                     device=self._dev_name,
                                     calc_flux=calc_flux_optim)
                # print(sim.calc_flux)
            else:
                sim = self.sim_class(self.rn,
                                     self.sim_runtime,
                                     device=self._dev_name,
                                     calc_flux=calc_flux_optim)

            # Perform simulation
            self.optimizer.zero_grad()
            if self.rn.boolCreation_rxn:
                total_yield, cur_time,unused_monomer, total_flux = \
                    sim.simulate(optim,
                                 node_str,
                                 corr_rxns=corr_rxns,
                                 conc_scale=conc_scale,
                                 mod_factor=mod_factor,
                                 conc_thresh=conc_thresh,
                                 mod_bool=mod_bool,
                                 verbose=verbose)
            elif self.rn.chaperone:
                total_yield, dimer_yield, chap_sp_yield, dimer_max, chap_max, endtime, total_flux = \
                    sim.simulate(optim,
                                 node_str,
                                 corr_rxns=corr_rxns,
                                 conc_scale=conc_scale,
                                 mod_factor=mod_factor,
                                 conc_thresh=conc_thresh,
                                 mod_bool=mod_bool,
                                 verbose=verbose,
                                 yield_species=yield_species)
            else:
                total_yield, total_flux = \
                    sim.simulate(optim,
                                 node_str,
                                 corr_rxns=corr_rxns,
                                 conc_scale=conc_scale,
                                 mod_factor=mod_factor,
                                 conc_thresh=conc_thresh,
                                 mod_bool=mod_bool,
                                 verbose=verbose,
                                 yield_species=yield_species)
            #print("Type/class of yield: ", type(total_yield))

            #Check change in yield from last gradient step. Break if less than a tolerance
            # if i > 1 and (total_yield - self.yield_per_iter[-1] >0 and total_yield - self.yield_per_iter[-1] < 1e-8):
            #     counter+=1
            #     print(total_yield,self.yield_per_iter[-1])
            #     if counter >10 :
            #         print("Max tolerance reached. Stopping optimization")
            #         print('yield on sim iteration ' + str(i) + ' was ' + str(total_yield.item() * 100)[:4] + '%')
            #         return self.rn
            # else:
            #     counter = 0
            self.yield_per_iter.append(total_yield.item())
            # self.flux_per_iter.append(total_flux.item())
            # update tracked data
            self.sim_observables.append(self.rn.observables.copy())
            self.sim_observables[-1]['steps'] = np.array(sim.steps)
            self.parameter_history.append(self.rn.kon.clone().detach().to(torch.device('cpu')).numpy())



            if optim in ['yield', 'time']:
                if optim == 'yield':
                    print(f'Yield on sim. iteration {i} was f{str(total_yield.item() * 100)[:4]}%.')
                elif optim == 'time':
                    print(f'Yield on sim iteration {i} was {str(total_yield.item() * 100)[:4]}%' + \
                          '\tTime :', str(cur_time))
                # print(self.rn.copies_vec)
                # preform gradient step
                if i != self.optim_iterations - 1:
                    # if self.rn.coupling:
                    #     k = torch.exp(self.rn.compute_log_constants(self.rn.params_kon, self.rn.params_rxn_score_vec,
                    #                                             scalar_modifier=1.))
                    #     physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)  # stops zeroing or negating params
                    #     cost = -total_yield + physics_penalty
                    #
                    #     cost.backward(retain_graph=True)
                    # elif self.rn.partial_opt:
                    #     k = torch.exp(self.rn.compute_log_constants(self.rn.params_kon, self.rn.params_rxn_score_vec,
                    #                                             scalar_modifier=1.))
                    #     physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)  # stops zeroing or negating params
                    #     cost = -total_yield + physics_penalty
                    #
                    #     cost.backward(retain_graph=True)
                    # else:
                    #     k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                    #                                             scalar_modifier=1.))
                    #     physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)  # stops zeroing or negating params
                    #     cost = -total_yield + physics_penalty

                        # cost.backward()
                    if self.rn.coupling:
                        new_params = self.rn.params_kon.clone().detach()
                        # for rc in range(len(self.rn.kon)):
                        #     if rc in self.rn.cid.keys():
                        #         self.rn.kon[rc] = self.rn.params_kon[self.rn.coup_map[self.rn.cid[rc]]]
                        #     else:
                        #         self.rn.kon[rc] = self.rn.params_kon[self.rn.coup_map[rc]]
                    elif self.rn.partial_opt and self.rn.assoc_is_param:
                        # new_params = self.rn.params_kon.clone().detach()
                        new_params = [p.clone().detach() for p in self.rn.params_kon]
                        # for r in range(len(self.rn.params_kon)):
                        #     print("Is leaf : ",self.rn.params_kon[r].is_leaf, "Grad: ",self.rn.params_kon[r].requires_grad)
                        #     self.rn.kon[self.rn.optim_rates[r]] = self.rn.params_kon[r]
                    elif self.rn.homo_rates and self.rn.assoc_is_param:
                        new_params = self.rn.params_kon.clone().detach()
                    elif self.rn.copies_is_param:
                        new_params = self.rn.c_params.clone().detach()
                    elif self.rn.chap_is_param:
                        new_params = [l.clone().detach() for l in self.rn.chap_params]
                    elif self.rn.dissoc_is_param:
                        if self.rn.partial_opt:
                            new_params = self.rn.params_koff.clone().detach()
                            self.rn.params_kon = self.rn.params_koff / \
                                (self.rn._C0 * torch.exp(self.rn.params_rxn_score_vec))
                            for r in range(len(new_params)):
                                self.rn.kon[self.rn.optim_rates[r]] = self.rn.params_kon[r]
                            print("Current On rates:", self.rn.kon)
                        else:
                            print("Current On rates:", torch.exp(k)[:len(self.rn.kon)])
                            new_params = [l.clone().detach() for l in self.rn.params_koff]
                    elif self.rn.dG_is_param:
                        # print("Current On rates: ", torch.exp(k)[:len(self.rn.kon)])

                        if self.rn.dG_mode==1:
                            new_params = [l.clone().detach() for l in self.rn.params_k]
                        else:
                            new_params = [l.clone().detach() for l in self.rn.params_k]
                            # new_params = self.rn.params_k.clone().detach()
                    else:
                        new_params = self.rn.kon.clone().detach()
                    #print('New reaction rates: ' + str(self.rn.kon.clone().detach()))
                    # new_params = self.rn.kon.clone().detach()
                    print('current params:', str(new_params))
                    #Store yield and params data
                    if total_yield-max_yield > 0:
                        if self.rn.chap_is_param:
                            self.final_yields.append([total_yield,dimer_yield,chap_sp_yield])
                            self.dimer_max.append(dimer_max)
                            self.chap_max.append(chap_max)
                            self.endtimes.append(endtime)
                        else:
                            self.final_yields.append(total_yield)

                        self.final_solns.append(new_params)
                        self.final_t50.append(total_flux[0])
                        self.final_t85.append(total_flux[1])
                        self.final_t95.append(total_flux[2])
                        self.final_t99.append(total_flux[3])
                        if self.rn.boolCreation_rxn:
                            self.final_unused_mon.append(unused_monomer)
                            self.curr_time.append(cur_time)

                    if self.rn.assoc_is_param:
                        if self.rn.coupling:
                            k = torch.exp(self.rn.compute_log_constants(self.rn.params_kon, self.rn.params_rxn_score_vec,scalar_modifier=1.))
                            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                            physics_penalty = torch.sum(100 * F.relu(-1 * (k - curr_lr * 10))).to(self.dev) #+ torch.sum(10 * F.relu(1 * (k - max_thresh))).to(self.dev) # stops zeroing or negating params
                            cost = -total_yield + physics_penalty
                            cost.backward(retain_graph=True)   #retain_graph = True only required for partial_opt + coupled model
                        elif self.rn.partial_opt:
                            if self.rn.boolCreation_rxn:
                                local_kon = torch.zeros([len(self.rn.params_kon)], requires_grad=True).double()
                                for r in range(len(local_kon)):
                                    local_kon[r]=self.rn.params_kon[r]
                                k = torch.exp(self.rn.compute_log_constants(local_kon, self.rn.params_rxn_score_vec,scalar_modifier=1.))
                                curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                                physics_penalty = torch.sum(100 * F.relu(-1 * (k - curr_lr * 10))).to(self.dev) # stops zeroing or negating params
                                if optim=='yield':
                                    if creat_yield==-1:
                                        unused_penalty = max_thresh*unused_monomer
                                        # cost = -total_yield -(total_yield/cur_time) + physics_penalty #+ unused_penalty
                                        cost = -100*total_yield/cur_time + physics_penalty
                                        cost.backward(retain_graph=True)
                                        print("Grad: ",end="")
                                        for r in range(len(self.rn.params_kon)):
                                            print(self.rn.params_kon[r],"-",self.rn.params_kon[r].grad,end=" ")
                                        print("")
                                    else:
                                        var_penalty=0
                                        if varBool:
                                            var_tensor = torch.zeros((len(self.rn.params_kon)))
                                            for r in range(len(self.rn.params_kon)):
                                                var_tensor[r] = self.rn.params_kon[r]

                                            var_penalty = F.relu(-1 * (torch.var(var_tensor)/torch.mean(var_tensor) - var_thresh/len(self.rn.params_kon)))    #var_thresh is how much should the minimum variance be
                                            print("Var: ",torch.var(var_tensor),"Penalty: ",var_penalty)
                                        cost =  -total_yield +var_penalty + physics_penalty #- total_yield/cur_time
                                        cost.backward(retain_graph=True)
                                        print("Grad: ",end="")
                                        for r in range(len(self.rn.params_kon)):
                                            print(self.rn.params_kon[r],"-",self.rn.params_kon[r].grad,end=" ")
                                        print("")
                                elif optim=='time':
                                    cost = cur_time
                                    cost.backward(retain_graph=True)
                                    print("Grad: ",end="")
                                    for r in range(len(self.rn.params_kon)):
                                        print(self.rn.params_kon[r],"-",self.rn.params_kon[r].grad,end=" ")
                                    print("")

                            else:
                                unused_penalty=0
                                k = torch.exp(self.rn.compute_log_constants(self.rn.params_kon, self.rn.params_rxn_score_vec,scalar_modifier=1.))
                                curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                                physics_penalty = torch.sum(10 * F.relu(-1 * (k - curr_lr * 10))).to(self.dev) + torch.sum(10 * F.relu(1 * (k - max_thresh))).to(self.dev) # stops zeroing or negating params ; Second term prevents exceeding a max_thresh
                                cost = -total_yield + physics_penalty + unused_penalty
                                cost.backward(retain_graph=True)
                        elif self.rn.homo_rates:
                            k = torch.exp(self.rn.compute_log_constants(self.rn.params_kon, self.rn.params_rxn_score_vec,scalar_modifier=1.))
                            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                            physics_penalty = torch.sum(10 * F.relu(-1 * (k - curr_lr * 10))).to(self.dev) + torch.sum(10 * F.relu(1 * (k - max_thresh))).to(self.dev) # stops zeroing or negating params
                            cost = -total_yield + physics_penalty
                            cost.backward(retain_graph=True)
                        else:
                            k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                                                            scalar_modifier=1.))
                            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                            physics_penalty = torch.sum(10 * F.relu(-1 * (k - curr_lr * 10))).to(self.dev) + torch.sum(10 * F.relu(1 * (k - max_thresh))).to(self.dev)
                            if lowvar:
                                mon_rxn = self.rn.rxn_class[1]
                                var_penalty = 100*F.relu(1 * (torch.var(k[mon_rxn])))
                                print("Var penalty: ",var_penalty,torch.var(k[:3]))
                            else:
                                var_penalty=0
                            # ratio_penalty = 1000*F.relu(1*((torch.max(k[3:])/torch.min(k[:3])) - 500 ))
                            # print("Var penalty: ",var_penalty,torch.var(k[:3]))
                            # print("Ratio penalty: ",ratio_penalty,torch.max(k[3:])/torch.min(k[:3]))

                            # dimer_penalty = 10*F.relu(1*(k[16] - self.lr*20))+10*F.relu(1*(k[17] - self.lr*20))+10*F.relu(1*(k[18] - self.lr*20))
                            cost = -total_yield + physics_penalty + var_penalty #+ dimer_penalty#+ var_penalty #+ ratio_penalty
                            cost.backward()
                            print("Grad: ",self.rn.kon.grad)
                    elif self.rn.copies_is_param:
                        c = self.rn.c_params.clone().detach()
                        physics_penalty = torch.sum(10 * F.relu(-1 * (c))).to(self.dev)# stops zeroing or negating params
                        cost = -total_yield + physics_penalty
                        cost.backward()
                    elif self.rn.chap_is_param:
                        n_copy_params = len(self.rn.paramid_copy_map.keys())
                        n_rxn_params = len(self.rn.paramid_uid_map.keys())

                        pen_copies = torch.zeros((n_copy_params),requires_grad=True).double()
                        pen_rates = torch.zeros((n_rxn_params),requires_grad=True).double()
                        for r in range(n_copy_params):
                            pen_copies[r] = self.rn.chap_params[r].clone()
                        for r in range(n_rxn_params):
                            pen_rates[r]= self.rn.chap_params[r+n_copy_params]
                        # c = self.rn.chap_params[0].clone().detach()
                        # k = self.rn.chap_params[1].clone().detach()
                        physics_penalty = torch.sum(max_thresh * F.relu(-10 * (pen_copies-1))).to(self.dev) + torch.sum(max_thresh * F.relu(-1 * (pen_rates - 1e-2))).to(self.dev) #+ torch.sum(00 * F.relu(c-1e2)).to(self.dev)
                        print("Penalty: ",physics_penalty, "Dimer yield: ",dimer_yield,"ABT yield: ",chap_sp_yield)

                        # cost = -total_yield + physics_penalty + 1*dimer_yield + 1*chap_sp_yield
                        # cost = 1*chap_sp_yield #-total_yield #+1*dimer_yield
                        if chap_mode == 1:
                            cost = -total_yield-dimer_yield
                        elif chap_mode ==2:
                            cost = chap_sp_yield+dimer_yield
                        elif chap_mode==3:
                            cost = -total_yield
                        cost.backward(retain_graph=True)
                        for i in range(len(self.rn.chap_params)):
                            print("Grad: ",self.rn.chap_params[i].grad,end="")
                        print("")
                    elif self.rn.dissoc_is_param:
                        if self.rn.partial_opt:
                            k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,scalar_modifier=1.))
                            new_l_k = torch.cat([k,torch.log(self.rn.params_koff)],dim=0)
                            physics_penalty = torch.sum(10 * F.relu(-1 * (new_l_k))).to(self.dev)  # stops zeroing or negating params
                            cost = -total_yield + physics_penalty
                            cost.backward(retain_graph=True)
                        else:

                            k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                                                                scalar_modifier=1.))
                            physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)
                            cost = -total_yield + physics_penalty
                            # print(self.optimizer.state_dict)
                            cost.backward()
                            metric = torch.mean(self.rn.params_koff[0].clone().detach()).item()
                    elif self.rn.dG_is_param:
                        k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                                                            scalar_modifier=1.))
                        g = self.rn.compute_total_dG(k)
                        print("Total Complex dG = ",g)

                        dG_penalty = F.relu((g-(self.rn.complx_dG+2))) + F.relu(-1*(g-(self.rn.complx_dG-2)))
                        print("Current On rates: ", k[:len(self.rn.kon)])
                        physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev) + torch.sum(100 * F.relu((k - 1e2))).to(self.dev)
                        cost = -total_yield + physics_penalty + 10*dG_penalty
                        # print(self.optimizer.state_dict)
                        cost.backward(retain_graph=True)
                        metric = torch.mean(self.rn.params_k[1].clone().detach()).item()


                    # self.scheduler.step(metric)
                    if (self.lr_change_step is not None) and (total_yield>=change_lr_yield):
                        change_lr = True
                        print("Curr learning rate : ")
                        for param_groups in self.optimizer.param_groups:
                            print(param_groups['lr'])
                            if param_groups['lr'] < 1e-2:
                                change_lr=False
                        if change_lr:
                            self.scheduler.step()


                    #Changing learning rate
                    if (self.lr_change_step is not None) and (i%self.lr_change_step ==0) and (i>0):
                        print("New learning rate : ")
                        for param_groups in self.optimizer.param_groups:
                            print(param_groups['lr'])

                    self.optimizer.step()

                    #print("Previous reaction rates: ",str(self.rn.kon.clone().detach()))




            # elif optim =='flux':
            #     print('Flux on sim iteration ' + str(i) + ' was ' + str(total_flux.item()))
            #     # preform gradient step
            #     if i != self.optim_iterations - 1:
            #         k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
            #                                                     scalar_modifier=1.))
            #         physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)  # stops zeroing or negating params
            #         cost = -total_flux + physics_penalty
            #         cost.backward()
            #         self.optimizer.step()
            #         new_params = self.rn.kon.clone().detach()
            #         print('current params: ' + str(new_params))

            elif optim == 'flux_coeff':
                print("Optimizing Flux Correlations")
                print(f'Yield on sim iteration {i} was {total_yield.item()}.')
                if i != self.optim_iterations - 1:
                        k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                                                                    scalar_modifier=1.))
                        physics_penalty = torch.sum(10 * F.relu(-1 * (k - self.lr * 10))).to(self.dev)  # stops zeroing or negating params
                        cost = -total_yield + physics_penalty
                        cost.backward()
                        self.optimizer.step()
                        new_params = self.rn.kon.clone().detach()
                        print('current params: ' + str(new_params))

                        # if total_yield-max_yield > 0:
                        #     self.final_yields.append(total_yield)
                        #     self.final_solns.append(new_params)




            values = psutil.virtual_memory()
            mem = values.available / (1024.0 ** 3)
            if mem < .5:
                # kill program if it uses to much ram
                print("Killing optimization because too much RAM being used.")
                print(values.available,mem)
                return self.rn
            if i == self.optim_iterations - 1:
                print("optimization complete")
                print("Final params: " + str(new_params))
                return self.rn

            del sim


    def optimize_wrt_expdata(self,optim='yield',node_str=None,max_yield=0.5,max_thresh=10,lowvar=False,conc_scale=1.0,mod_factor=1.0,conc_thresh=1e-5,mod_bool=True,verbose=False,yield_species=-1,trueval=None,time_threshmax=1):
        print("Reaction Parameters before optimization: ")
        print(self.rn.get_params())

        rate_data = pd.read_csv(trueval,delimiter='\t',comment='#',names=['Timestep','Conc'])

        self.mse_error = []



        print("Optimizer State:",self.optimizer.state_dict)
        counter = 0
        calc_flux_optim=False
        if optim=='flux_coeff':
            calc_flux_optim=True
        for i in range(self.optim_iterations):
            # reset for new simulator
            self.rn.reset()

            sim = self.sim_class(self.rn,
                                     self.sim_runtime,
                                     device=self._dev_name,calc_flux=calc_flux_optim)


            # preform simulation
            self.optimizer.zero_grad()

            total_yield,conc_tensor,total_flux = sim.simulate_wrt_expdata(optim,node_str,conc_scale=conc_scale,mod_factor=mod_factor,conc_thresh=conc_thresh,mod_bool=mod_bool,verbose=verbose,yield_species=yield_species)

            self.yield_per_iter.append(total_yield.item())
            # self.flux_per_iter.append(total_flux.item())
            # update tracked data
            self.sim_observables.append(self.rn.observables.copy())
            self.sim_observables[-1]['steps'] = np.array(sim.steps)
            self.parameter_history.append(self.rn.kon.clone().detach().to(torch.device('cpu')).numpy())


            if optim =='yield' or optim=='time':
                if optim=='yield':
                    print('yield on sim iteration ' + str(i) + ' was ' + str(total_yield.item() * 100)[:4] + '%')
                # elif optim=='time':
                #     print('yield on sim iteration ' + str(i) + ' was ' + str(total_yield.item() * 100)[:4] + '%' + '\tTime : ',str(cur_time))
                # print(self.rn.copies_vec)
                # preform gradient step
                if i != self.optim_iterations - 1:

                    new_params = self.rn.kon.clone().detach()
                    print('current params: ' + str(new_params))

                    #Store yield and params data
                    if total_yield-max_yield > 0:

                        self.final_yields.append(total_yield)
                        self.final_solns.append(new_params)
                        self.final_t50.append(total_flux[0])
                        self.final_t85.append(total_flux[1])
                        self.final_t95.append(total_flux[2])
                        self.final_t99.append(total_flux[3])

                    if self.rn.assoc_is_param:
                        k = torch.exp(self.rn.compute_log_constants(self.rn.kon, self.rn.rxn_score_vec,
                                                            scalar_modifier=1.))
                        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                        physics_penalty = torch.sum(100 * F.relu(-1 * (k - curr_lr * 1000))).to(self.dev) #+ torch.sum(10 * F.relu(1 * (k - max_thresh))).to(self.dev)


                        sel_parm_indx = []
                        time_thresh=1e-4
                        time_array = np.array(sim.steps)
                        conc_array = conc_tensor


                        print(type(conc_array))

                        #Experimental data
                        mask1 = (rate_data['Timestep']>=time_thresh) & (rate_data['Timestep']<time_threshmax)
                        exp_time = np.array(rate_data['Timestep'][mask1])

                        exp_conc = np.array(rate_data['Conc'][mask1])

                        # mse_tensor = torch.zeros(len(exp_time))
                        # mse_tensor.requires_grad=True
                        mse=torch.Tensor([0.])
                        mse.requires_grad=True
                        total_time_diff = 0
                        for e_indx in range(len(exp_time)):
                            curr_time = exp_time[e_indx]
                            time_diff = (np.abs(time_array-curr_time))
                            get_indx = time_diff.argmin()
                            total_time_diff+=time_diff[get_indx]
                            mse = mse+ (exp_conc[e_indx] - conc_array[get_indx])**2
                            # mse_tensor[e_indx] = (exp_conc[e_indx] - conc_array[get_indx])**2


                        # print(type(mse_tensor))
                        # mse = mse/len(exp_time)
                        mse_mean = torch.mean(mse)
                        self.mse_error.append(mse_mean.item())
                        print("Total time diff: ",total_time_diff)
                        # print(mse)
                        cost = mse_mean + physics_penalty
                        cost.backward()
                        print('MSE on sim iteration ' + str(i) + ' was ' + str(mse_mean))
                        print("Grad: ",self.rn.kon.grad)

                    if (self.lr_change_step is not None) and (total_yield>=change_lr_yield):
                        change_lr = True
                        print("Curr learning rate : ")
                        for param_groups in self.optimizer.param_groups:
                            print(param_groups['lr'])
                            if param_groups['lr'] < 1e-2:
                                change_lr=False
                        if change_lr:
                            self.scheduler.step()


                    #Changing learning rate
                    if (self.lr_change_step is not None) and (i%self.lr_change_step ==0) and (i>0):
                        print("New learning rate : ")
                        for param_groups in self.optimizer.param_groups:
                            print(param_groups['lr'])

                    self.optimizer.step()


            values = psutil.virtual_memory()
            mem = values.available / (1024.0 ** 3)
            if mem < .5:
                # kill program if it uses to much ram
                print("Killing optimization because too much RAM being used.")
                print(values.available,mem)
                return self.rn
            if i == self.optim_iterations - 1:
                print("optimization complete")
                print("Final params: " + str(new_params))
                return self.rn

            del sim



if __name__ == '__main__':
    from steric_free_simulator import ReactionNetwork
    base_input = './input_files/dimer.bngl'
    rn = ReactionNetwork(base_input, one_step=True)
    rn.reset()
    rn.intialize_activations()
    optim = Optimizer(reaction_network=rn,
                      sim_runtime=.001,
                      optim_iterations=10,
                      learning_rate=10,)
    vec_rn = optim.optimize()
