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
            # print("Using CPU")
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
            self.optimizer = torch.optim.Adam(param_itr, learning_rate)
        elif method =='RMSprop':
            self.optimizer = torch.optim.RMSprop(param_itr, learning_rate)

        self.lr = learning_rate
        self.optim_iterations = optim_iterations
        self.sim_observables = []
        self.parameter_history = []
        self.yield_per_iter = []
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
        self.endtimes=[]
        if lr_change_step is not None:
            if gamma == None:
                gamma = 0.5
            # self.scheduler = StepLR(self.optimizer,step_size=lr_change_step,gamma=gamma)
            # self.scheduler = ReduceLROnPlateau(self.optimizer,'max',patience=30)
            if self.rn.assoc_is_param:
                self.scheduler = MultiplicativeLR(self.optimizer,lr_lambda=self.assoc_lambda)
            self.lr_change_step = lr_change_step
        else:
            self.lr_change_step = None

    def assoc_lambda(self, opt_itr):
        new_lr = torch.min(self.rn.kon).item() * self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr / curr_lr)
    
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

    def plot_yield(self):
        steps = np.arange(len(self.yield_per_iter))
        data = np.array(self.yield_per_iter, dtype=np.float)
        
        plt.plot(steps, data,label='Yield')
        plt.title = 'Yield at each iteration'
        plt.xlabel("Iterations")
        plt.ylabel("Yield(%)")
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

        for i in range(self.optim_iterations):
            # Reset for new simulator
            self.rn.reset()

            sim = self.sim_class(self.rn,
                                    self.sim_runtime,
                                    device=self._dev_name)

            # Perform simulation
            self.optimizer.zero_grad()
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

            self.yield_per_iter.append(total_yield.item())
            # update tracked data
            self.sim_observables.append(self.rn.observables.copy())
            self.sim_observables[-1]['steps'] = np.array(sim.steps)
            self.parameter_history.append(self.rn.kon.clone().detach().to(torch.device('cpu')).numpy())


            print(f'Yield on sim. iteration {i} was {str(total_yield.item() * 100)[:4]}%.')

            # preform gradient step
            if i != self.optim_iterations - 1:
                    
                    if self.rn.coupling:
                        new_params = self.rn.params_kon.clone().detach()
                    elif self.rn.homo_rates and self.rn.assoc_is_param:
                        new_params = self.rn.params_kon.clone().detach()
                    else:
                        new_params = self.rn.kon.clone().detach()
                    print('current params:', str(new_params))
                    #Store yield and params data
                    if total_yield-max_yield > 0:
                        self.final_yields.append(total_yield)
                        self.final_solns.append(new_params)
                        self.final_t50.append(total_flux[0])
                        self.final_t85.append(total_flux[1])
                        self.final_t95.append(total_flux[2])
                        self.final_t99.append(total_flux[3])
                        
                    if self.rn.assoc_is_param:
                        if self.rn.coupling:
                            k = torch.exp(self.rn.compute_log_constants(self.rn.params_kon, self.rn.params_rxn_score_vec,scalar_modifier=1.))
                            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                            physics_penalty = torch.sum(100 * F.relu(-1 * (k - curr_lr * 10))).to(self.dev) #+ torch.sum(10 * F.relu(1 * (k - max_thresh))).to(self.dev) # stops zeroing or negating params
                            cost = -total_yield + physics_penalty
                            cost.backward(retain_graph=True)   #retain_graph = True only required for partial_opt + coupled model
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
                            # print("Grad: ",self.rn.kon.grad)
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
    from KineticAssembly_AD import ReactionNetwork
    base_input = './input_files/dimer.bngl'
    rn = ReactionNetwork(base_input, one_step=True)
    rn.reset()
    rn.intialize_activations()
    optim = Optimizer(reaction_network=rn,
                      sim_runtime=.001,
                      optim_iterations=10,
                      learning_rate=10,)
    vec_rn = optim.optimize()
