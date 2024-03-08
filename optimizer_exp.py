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
from torch import DoubleTensor as Tensor
import random
import pandas as pd
from scipy.interpolate import CubicSpline



class OptimizerExp:

    def __init__(self, reaction_network,
                 sim_runtime: float,
                 optim_iterations: int,
                 learning_rate,
                 device='cpu',method='Adam',lr_change_step=None,gamma=None,mom=0,random_lr=False,reg_penalty=10,dG_penalty=10):
        if torch.cuda.is_available() and "cpu" not in device:
            self.dev = torch.device(device)
            print("Using " + device)
        else:
            self.dev = torch.device("cpu")
            device = 'cpu'
            print("Using CPU")
        self._dev_name = device
        self.sim_class = VecSim
        if not isinstance(reaction_network,VectorizedRxnNet):
            try:
                self.rn = VectorizedRxnNet(reaction_network, dev=self.dev)
            except Exception:
                raise TypeError(" Must be type ReactionNetwork or VectorizedRxnNetwork.")
        else:
            self.rn = reaction_network
        self.sim_runtime = sim_runtime
        param_itr = self.rn.get_params()

        if method =='Adam':
            if self.rn.homo_rates and self.rn.dG_is_param:
                param_list=[]
                for i in range(len(param_itr)):
                    # print("#####")
                    # print(param_itr[i])
                    param_list.append({'params':param_itr[i],'lr':learning_rate[i],'momentum':mom})

                self.optimizer = torch.optim.RMSprop(param_list)
            elif self.rn.coupling and self.rn.dG_is_param:
                param_list=[]
                for i in range(len(param_itr)):
                    # print("#####")
                    # print(param_itr[i])
                    param_list.append({'params':param_itr[i],'lr':learning_rate[i],'momentum':mom})
                self.optimizer = torch.optim.RMSprop(param_list)
            else:
                self.optimizer = torch.optim.RMSprop(param_itr, learning_rate,momentum=mom)
        elif method =='RMSprop':
            if self.rn.homo_rates and self.rn.dG_is_param:
                param_list=[]
                for i in range(len(param_itr)):
                    # print("#####")
                    # print(param_itr[i])
                    param_list.append({'params':param_itr[i],'lr':learning_rate[i],'momentum':mom})

                self.optimizer = torch.optim.RMSprop(param_list)
            elif self.rn.coupling and self.rn.dG_is_param:
                param_list=[]
                for i in range(len(param_itr)):
                    # print("#####")
                    # print(param_itr[i])
                    param_list.append({'params':param_itr[i],'lr':learning_rate[i],'momentum':mom})
                self.optimizer = torch.optim.RMSprop(param_list)
            else:
                self.optimizer = torch.optim.RMSprop(param_itr, learning_rate,momentum=mom)

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
        self.reg_penalty=reg_penalty
        self.dG_penalty=dG_penalty
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

    def assoc_lambda(self,opt_itr):
        new_lr = torch.min(self.rn.kon).item()*self.lr
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return(new_lr/curr_lr)

    def plot_observable(self, iteration, nodes_list,ax=None):
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

    

    def optimize(self,optim='yield',batch_mode="conc",node_str=None,max_yield=0.5,
    max_thresh=10,conc_scale=1.0,mod_factor=1.0,conc_thresh=1e-5,mod_bool=True,verbose=False,yield_species=-1,
    conc_files_pref=None,conc_files_range=[],yield_threshmin=0.05,yield_threshmax=1,sse_mode='square'):

        """
        Optimize the reaction network with respect to concentration using multiple concentration curves.

        Parameters:
        - optim (str): Optimization method, default is 'yield'.
        - batch_mode (str): Batch mode, default is 'conc'.
        - node_str (str): Node string, default is None.
        - max_yield (float): Maximum yield, default is 0.5.
        - max_thresh (float): Maximum threshold, default is 10.
        - conc_scale (float): Concentration scale, default is 1.0.
        - mod_factor (float): Modification factor, default is 1.0.
        - conc_thresh (float): Concentration threshold, default is 1e-5.
        - mod_bool (bool): Modification boolean, default is True.
        - verbose (bool): Verbose mode, default is False.
        - yield_species (int): Yield species, default is -1.
        - conc_files_pref (str): Concentration files prefix, default is None.
        - conc_files_range (list): Concentration files range, default is an empty list.
        - yield_threshmin (float): Yield threshold minimum, default is 0.05.
        - yield_threshmax (float): Yield threshold maximum, default is 1.
        - sse_mode (str): Sum of squared errors mode, default is 'square'.
        """
        
        n_batches = len(conc_files_range)
        print("Total number of batches: ",n_batches)
        print("Optimizer State:",self.optimizer.state_dict)
        self.mse_error = []
        time_threshmin=1e-4
        # torch.autograd.set_detect_anomaly(True)

        for i in range(self.optim_iterations):
            # reset for new simulator
            self.rn.reset()
            sim = self.sim_class(self.rn,
                                     self.sim_runtime,
                                     device=self._dev_name)



            mse=torch.Tensor([0.])
            mse.requires_grad=True

            #This boolean is required for homo_rates and coupling protocol since in each call of simulate() method,
            #the kon is updated from the corresponsing parameters. Now that works for optimization in other functions, but in this case
            #since we run multiple simulations in 1 optim iteration, each simulation call will update the kon value. This leads to error since between
            #each gradient update steps, the parameters cannot be changed. So this boolean ensures that the kon is updated from params_kon only for the 1st simulate call in each iteration.
            update_kon_bool=True
            for b in range(n_batches):
                init_conc = float(conc_files_range[b])

                new_file = conc_files_pref+str(init_conc)+"uM"
                sp_names = ['A'+str(j) for j in range(self.rn.num_monomers)]
                var_names=['Timestep','Conc','c_scale','runtime']
                rate_data = pd.read_csv(new_file,delimiter='\t',comment='#',names=var_names+sp_names)
                conc_scale = rate_data['c_scale'][0]
                conc_thresh=conc_scale

                #TODO: Need a better way to read the conc from a file

                self.rn.initial_copies[0:self.rn.num_monomers] = Tensor([rate_data.iloc[0,4:]])

                time_mask_max = rate_data['Conc']/torch.min(self.rn.initial_copies[0:self.rn.num_monomers])>yield_threshmax
                time_mask_min = rate_data['Conc']/torch.min(self.rn.initial_copies[0:self.rn.num_monomers])>yield_threshmin
                time_indx_max = time_mask_max.loc[time_mask_max==True].index[0]
                time_indx_min = time_mask_min.loc[time_mask_min==True].index[0]
                time_threshmax=rate_data['Timestep'][time_indx_max]
                time_threshmin=rate_data['Timestep'][time_indx_min]



                # self.rn.initial_copies = update_copies_vec
                self.rn.reset()
                sim.reset(runtime=time_threshmax)    #Resets the variables of the sim class that are tracked during a simulation.
                # sim = self.sim_class(self.rn,
                                         # self.sim_runtime,
                                         # device=self._dev_name)
                print("----------------- Starting new batch of Simulation ------------------------------")
                print("------------------ Concentration : %f,%f,%f,%f -------------------------------" %(self.rn.initial_copies[0],self.rn.initial_copies[1],self.rn.initial_copies[2],self.rn.initial_copies[3]))
                # preform simulation
                self.optimizer.zero_grad()
                total_yield,conc_tensor,total_flux = sim.simulate_wrt_expdata(optim,node_str,conc_scale=conc_scale,mod_factor=mod_factor,conc_thresh=conc_thresh,mod_bool=mod_bool,verbose=verbose,yield_species=yield_species,update_kon_bool=update_kon_bool)
                update_kon_bool=False

                print('yield for simulation with  ' + str(init_conc)+"uM" + ' was ' + str(total_yield.item() * 100)[:4] + '%')



                #Extracting simulation data
                time_array = Tensor(np.array(sim.steps))
                conc_array = conc_tensor

                #Experimental data
                mask1 = (rate_data['Timestep']>=time_threshmin) & (rate_data['Timestep']<time_threshmax)
                exp_time = Tensor(np.array(rate_data['Timestep'][mask1]))
                exp_conc = Tensor(np.array(rate_data['Conc'][mask1]))

                #Interpolating data between time points
                cs_inter = CubicSpline(exp_time,exp_conc)
                time_points = np.geomspace(time_threshmin,time_threshmax,num=100)
                conc_points = cs_inter(time_points)

                total_time_diff = 0
                init_conc = torch.min(self.rn.initial_copies[0:self.rn.num_monomers])

                for e_indx in range(len(time_points)):
                    curr_time = time_points[e_indx]
                    time_diff = (np.abs(time_array-curr_time))

                    get_indx = time_diff.argmin()

                    total_time_diff+=time_diff[get_indx]
                    if sse_mode=='square':
                        curr_mse = ((conc_points[e_indx] - conc_array[get_indx])/init_conc)**2
                    elif sse_mode=='abs':
                        curr_mse = torch.abs((conc_points[e_indx] - conc_array[get_indx])/init_conc)
                    mse = mse+ curr_mse
                print("SSE at %f :  %f" %(init_conc,mse.item()))

                print("Exp Yield: ",conc_points[e_indx]/init_conc,"Sim Yield: ",conc_array[get_indx]/init_conc, "  at time threshold: ",time_threshmax)
            #End of running all batches of simulations
            #Calculate the avg of mse over all conc ranges
            # mse_mean = mse/n_batches
            mse_mean = mse


            if self.rn.coupling or self.rn.homo_rates:

                if self.rn.dG_is_param:
                    new_params_kon = self.rn.params_k[0].clone()
                    new_params_koff = self.rn.params_k[1].clone()
                    new_params = new_params_kon.detach().tolist() + new_params_koff.detach().tolist()
                    # new_params=self.rn.kon

                    print('current kon: ' + str(new_params_kon.detach()))
                    print('current koff: ' + str(new_params_koff.detach()))
                    curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    koff_lr = self.optimizer.state_dict()['param_groups'][1]['lr']
                    physics_penalty = torch.sum(self.reg_penalty * F.relu(-1 * (new_params_kon - curr_lr * 10))).to(self.dev) #+ torch.sum(10 * F.relu(1 * (k - max_thresh))).to(self.dev)

                    #Calculating current dG
                    if self.rn.homo_rates:
                        dG = -1*torch.log(self.rn.params_k[0][0]*self.rn._C0/self.rn.params_k[1][0])
                        print("Current dG: ",dG)
                        # min_dG = self.rn.base_dG-self.rn.ddG_fluc   #More stable
                        # max_dG = self.rn.base_dG+self.rn.ddG_fluc   #Less stable
                        dG_penalty = self.dG_penalty*F.relu(-1*(dG-self.rn.ddG_fluc_min)) + self.dG_penalty*F.relu(dG-self.rn.ddG_fluc_max)

                    elif self.rn.coupling:
                        dG = -1*torch.log(torch.div(self.rn.params_k[0]*self.rn._C0,self.rn.params_k[1]))
                        print("Current dG: ",dG)
                        # min_dG = self.rn.base_dG-self.rn.ddG_fluc   #More stable
                        # max_dG = self.rn.base_dG+self.rn.ddG_fluc   #Less stable
                        # dG_penalty = torch.sum(self.dG_penalty*F.relu(-1*(dG-min_dG))) + torch.sum(self.dG_penalty*F.relu(dG-max_dG))
                        dG_penalty = torch.sum(self.dG_penalty*F.relu(-1*(dG-self.rn.ddG_fluc_min)) + self.dG_penalty*F.relu(dG-self.rn.ddG_fluc_max))
                        physics_penalty = physics_penalty + torch.sum(self.reg_penalty * F.relu(-1 * (new_params_koff - koff_lr * 10))).to(self.dev)



                    cost = mse_mean + physics_penalty + dG_penalty
                    cost.backward(retain_graph=True)
                    print('MSE on sim iteration ' + str(i) + ' was ' + str(mse_mean))
                    print("Reg Penalty: ",physics_penalty)
                    print("dG_penalty: ",dG_penalty)
                    print("Grad: ",self.rn.params_k[0].grad,self.rn.params_k[1].grad)
                    self.mse_error.append(mse_mean.item())

                    self.yield_per_iter.append(total_yield.item())
                    self.sim_observables.append(self.rn.observables.copy())
                    self.sim_observables[-1]['steps'] = np.array(sim.steps)

                    self.parameter_history.append([new_params_kon.detach().tolist(),new_params_koff.detach().tolist()])

                else:
            # if False:
                    new_params = self.rn.params_kon.clone().detach()
                    # new_params=self.rn.kon
                    k = self.rn.params_kon

                    print('current params: ' + str(new_params))
                    curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    physics_penalty = torch.sum(self.reg_penalty * F.relu(-1 * (k - curr_lr * 50))).to(self.dev) #+ torch.sum(10 * F.relu(1 * (k - max_thresh))).to(self.dev)

                    cost = mse_mean + physics_penalty
                    cost.backward(retain_graph=True)
                    print('MSE on sim iteration ' + str(i) + ' was ' + str(mse_mean))
                    print("Reg Penalty: ",physics_penalty)
                    print("Grad: ",self.rn.params_kon.grad)

                    self.mse_error.append(mse_mean.item())

                    self.yield_per_iter.append(total_yield.item())
                    self.sim_observables.append(self.rn.observables.copy())
                    self.sim_observables[-1]['steps'] = np.array(sim.steps)
                    self.parameter_history.append(self.rn.params_kon.clone().detach().numpy())

            else:
                # new_params = self.rn.params_kon.clone().detach()
                new_params = self.rn.kon.clone().detach()
                k = self.rn.kon

                print('current params: ' + str(new_params))
                curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                physics_penalty = torch.sum(self.reg_penalty * F.relu(-1 * (k - curr_lr * 50))).to(self.dev) #+ torch.sum(10 * F.relu(1 * (k - max_thresh))).to(self.dev)

                cost = mse_mean #+ physics_penalty
                cost.backward(retain_graph=True)
                print('MSE on sim iteration ' + str(i) + ' was ' + str(mse_mean))
                print("Reg Penalty: ",physics_penalty)
                print("Grad: ",self.rn.kon.grad)



                self.mse_error.append(mse_mean.item())



                self.yield_per_iter.append(total_yield.item())
                self.sim_observables.append(self.rn.observables.copy())
                self.sim_observables[-1]['steps'] = np.array(sim.steps)
                self.parameter_history.append(self.rn.kon.clone().detach().to(torch.device('cpu')).numpy())

            self.optimizer.step()

            values = psutil.virtual_memory()
            mem = values.available / (1024.0 ** 3)
            if mem < .5:
                # kill program if it uses to much ram
                print("Killing optimization because too much RAM being used.")
                print(values.available,mem)
                return self.rn
            if i == self.optim_iterations - 1:
                print("Batch optimization complete")
                print("Updated params: " + str(new_params))

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
