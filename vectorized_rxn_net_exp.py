from typing import Tuple

from KineticAssembly_AD import ReactionNetwork
from KineticAssembly_AD import reaction_network as RN
from KineticAssembly_AD import VectorizedRxnNet

import networkx as nx

import torch
from torch import DoubleTensor as Tensor
from torch import rand
from torch import nn



class VectorizedRxnNetExp (VectorizedRxnNet):
    """
    Provides a lightweight class that inherits core functionality from VectorizedRxnNet and adds the ability to
    predict binding energies and association rates for a given reaction network. This class is designed to be used
    in conjunction with the OptimizerExp class to optimize the association rates and binding energies of a reaction
    network to fit experimental data.

    Units:
    units of Kon assumed to be [copies]-1 S-1, units of Koff S-1
    units of reaction scores are treated as J * c / mol where c is a user defined scalar
    """

    def __init__(self, rn: ReactionNetwork, initialize_params=True,assoc_is_param=True,dev='cpu',std_c=1e6,optim_rates=None,slow_rates=None,slow_ratio=1):
        """

        :param rn: The reaction network template
        :param assoc_is_param: whether the association constants should be treated as parameters for optimization
        :param dev: the device to use for torch tensors
        :param coupling : If two reactions have same kon. i.e. Addition of new subunit is same as previous subunit
        :param cid : Reaction ids in a dictionary format. {child_reaction:parent_reaction}. Set the rate of child_reaction to parent_reaction
        """

        super(VectorizedRxnNetExp,self).__init__(rn,initialize_params=False)



        self.dG_is_param = rn.dG_is_param
        if self.dG_is_param:
            self.ddG_fluc_min = rn.ddG_fluc_min
            self.ddG_fluc_max = rn.ddG_fluc_max

        self.base_dG = self.rxn_score_vec[0].clone().detach()
        self.initial_copies = self.copies_vec.clone().detach()
        self.assoc_is_param = assoc_is_param
        #Initialize Parameters for Experimental Optimization
        self.initialize_parameters()



    def initialize_parameters(self):
        if self.coupling == True:
            # c_rxn_count = len(rn.rxn_cid.keys())
            if self.partial_opt:
                c_rxn_count=len(self.optim_rates)
                self.params_kon = torch.zeros([c_rxn_count],requires_grad=True).double()
                self.params_rxn_score_vec = torch.zeros([c_rxn_count]).double()
                rid=0
                for i in range(c_rxn_count):
                    self.params_kon[rid] = self.kon.clone().detach()[self.optim_rates[i]]
                    self.params_rxn_score_vec[rid] = self.rxn_score_vec[self.optim_rates[i]]
                    self.coup_map[self.optim_rates[i]]=rid           #Map reaction index for independent reactions in self.kon to self.params_kon. Used to set the self.kon from self.params_kon
                    rid+=1
                self.params_kon.requires_grad_(True)
                self.initial_params = Tensor(self.params_kon).clone().detach()
                self.params_kon = nn.Parameter(self.params_kon, requires_grad=True)
            elif self.dG_is_param:
                ind_rxn_count = len(self.rxn_class[(1,1)])
                params_kon = torch.zeros([ind_rxn_count],requires_grad=True).double()
                params_koff = torch.zeros([ind_rxn_count],requires_grad=True).double()
                self.initial_params=[]
                rid=0
                for i in range(ind_rxn_count):
                    params_kon[rid] = self.kon.clone().detach()[self.rxn_class[(1,1)][i]]  ##Get the first uid of each class.Set that as the param for that class of rxns
                    params_koff[rid] = params_kon[rid]*self._C0*torch.exp(self.rxn_score_vec[self.rxn_class[(1,1)][i]])
                    self.coup_map[self.rxn_class[(1,1)][i]]=rid           #Map reaction index for independent reactions in self.kon to self.params_kon. Used to set the self.kon from self.params_kon
                    rid+=1

                self.initial_params.append(Tensor(params_kon).clone().detach())
                self.initial_params.append(Tensor(params_koff).clone().detach())
                self.params_k = []
                self.params_k.append(nn.Parameter(params_kon,requires_grad=True))
                self.params_k.append(nn.Parameter(params_koff,requires_grad=True))

            else:
                ind_rxn_count = len(self.rxn_class[(1,1)])
                self.params_kon = torch.zeros([ind_rxn_count], requires_grad=True).double()   #Create param Tensor for only the independant reactions
                self.params_rxn_score_vec = torch.zeros([ind_rxn_count]).double()
                #self.kon.requires_grad_(False)
                rid=0
                for i in range(ind_rxn_count):
                    # if i not in cid.keys():
                        ##Independent reactions
                    self.params_kon[rid] = self.kon.clone().detach()[self.rxn_class[(1,1)][i]]
                    self.params_rxn_score_vec[rid] = self.rxn_score_vec[self.rxn_class[(1,1)][i]]
                    self.coup_map[self.rxn_class[(1,1)][i]]=rid           #Map reaction index for independent reactions in self.kon to self.params_kon. Used to set the self.kon from self.params_kon
                    rid+=1
                self.params_kon.requires_grad_(True)
                self.initial_params = Tensor(self.params_kon).clone().detach()
                self.params_kon = nn.Parameter(self.params_kon, requires_grad=True)    #Setting the required kon values to be parameters for ooptimization
        elif self.partial_opt == True:
            c_rxn_count = len(self.optim_rates)
            # self.params_kon = torch.zeros([c_rxn_count], requires_grad=True).double()   #Create param Tensor for only the independant reactions
            # self.params_rxn_score_vec = torch.zeros([c_rxn_count]).double()
            # for i in range(c_rxn_count):
            #     self.params_kon[i] = self.kon.clone().detach()[self.optim_rates[i]]
            #     self.params_rxn_score_vec[i] = self.rxn_score_vec[self.optim_rates[i]]
            # self.params_kon.requires_grad_(True)
            # self.initial_params = Tensor(self.params_kon).clone().detach()


            params_kon = torch.zeros([c_rxn_count], requires_grad=True).double()
            self.initial_params = []
            self.params_rxn_score_vec = torch.zeros([c_rxn_count]).double()
            for i in range(c_rxn_count):
                params_kon[i] = self.kon.clone().detach()[self.optim_rates[i]]
                self.params_rxn_score_vec[i] = self.rxn_score_vec[self.optim_rates[i]]
                self.initial_params.append(self.kon.clone().detach()[self.optim_rates[i]])
            self.params_kon=[]
            for i in range(len(params_kon)):
                print(params_kon.clone().detach()[i])
                self.params_kon.append(params_kon.clone().detach()[i])
            for i in range(len(params_kon)):
                self.params_kon[i].requires_grad_(True)
                self.params_kon[i] = nn.Parameter(self.params_kon[i],requires_grad=True)
            for i in range(len(params_kon)):
                print("is Leaf: ",self.params_kon[i].is_leaf)

            self.kon.requires_grad_(True)     #Not sure if this is necessary

        elif self.homo_rates == True:
            if self.partial_opt:
                c_rxn_count=len(self.optim_rates)
                self.params_kon = torch.zeros([c_rxn_count],requires_grad=True).double()
                self.params_rxn_score_vec = torch.zeros([c_rxn_count]).double()
                rid=0
                for i in range(c_rxn_count):
                    self.params_kon[rid] = self.kon.clone().detach()[self.optim_rates[i]]
                    self.params_rxn_score_vec[rid] = self.rxn_score_vec[self.optim_rates[i]]
                    rid+=1
                self.params_kon = nn.Parameter(self.params_kon, requires_grad=True)
            elif self.dG_is_param:
                params_kon = torch.zeros([len(self.rxn_class.keys())],requires_grad=True).double()
                params_koff = torch.zeros([1],requires_grad=True).double()


                self.params_rxn_score_vec = torch.zeros([len(self.rxn_class.keys())]).double()
                self.initial_params=[]
                counter=0
                for k,rid in self.rxn_class.items():

                    params_kon[counter] = self.kon.clone().detach()[rid[0]]  ##Get the first uid of each class.Set that as the param for that class of rxns
                    self.params_rxn_score_vec[counter] = self.rxn_score_vec[rid[0]]
                    counter+=1

                self.initial_params.append(Tensor(params_kon).clone().detach())
                # self.initial_params.append(Tensor(params_kon[1:]).clone().detach())
                for p in range(len(params_koff)):
                    params_koff[p] = params_kon[p]*self._C0*torch.exp(self.params_rxn_score_vec[p])

                self.initial_params.append(Tensor(params_koff).clone().detach())
                self.params_k = []
                self.params_k.append(nn.Parameter(params_kon,requires_grad=True))
                # self.params_k.append(nn.Parameter(params_kon[1:],requires_grad=True))
                self.params_k.append(nn.Parameter(params_koff,requires_grad=True))
                print("After parametrixation: ",self.kon)

            else:
                self.params_kon = torch.zeros([len(self.rxn_class.keys())],requires_grad=True).double()
                self.params_rxn_score_vec = torch.zeros([len(self.rxn_class.keys())]).double()
                counter=0
                for k,rid in self.rxn_class.items():

                    self.params_kon[counter] = self.kon.clone().detach()[rid[0]]  ##Get the first uid of each class.Set that as the param for that class of rxns
                    self.params_rxn_score_vec[counter] = self.rxn_score_vec[rid[0]]
                    counter+=1
                self.params_kon.requires_grad_(True)
                self.initial_params = Tensor(self.params_kon).clone().detach()
                self.params_kon = nn.Parameter(self.params_kon, requires_grad=True)

        else:
            self.kon = nn.Parameter(self.kon, requires_grad=True)
            self.initial_params = Tensor(self.kon).clone().detach()

        print("Shifting to device: ", self.dev)
        self.to(self.dev)


    def reset(self, reset_params=False):
        self.copies_vec = self.initial_copies.clone()

        if reset_params:
            if self.coupling:
                if self.dG_is_param:
                    self.params_k[0] = nn.Parameter(self.initial_params[0].clone(),requires_grad=True)
                    self.params_k[1] = nn.Parameter(self.initial_params[1].clone(),requires_grad=True)
                else:
                    self.params_kon = nn.Parameter(self.initial_params.clone(), requires_grad=True)
            elif self.partial_opt:
                # self.params_kon = nn.Parameter(self.initial_params.clone(), requires_grad=True)
                for i in range(len(self.initial_params)):
                    self.params_kon[i] = nn.Parameter(self.initial_params[i].clone(),requires_grad=True)
            elif self.homo_rates:
                if self.dG_is_param:
                    for i in range(len(self.initial_params)):
                        self.params_k[i] = nn.Parameter(self.initial_params[i].clone(),requires_grad=True)
                else:
                    self.params_kon = nn.Parameter(self.initial_params.clone(), requires_grad=True)
            else:
                self.kon = nn.Parameter(self.initial_params.clone(), requires_grad=True)
        for key in self.observables:
            self.observables[key] = (self.observables[key][0], [])

    def get_params(self):

        if self.coupling:
            if self.dG_is_param:
                return self.params_k
            else:
                return [self.params_kon]
        elif self.partial_opt:
            return self.params_kon
        elif self.homo_rates:
            if self.dG_is_param:
                return self.params_k
            else:
                return [self.params_kon]
        else:
            return [self.kon]

    def to(self, dev):
        self.M = self.M.to(dev)
#        self._avo = self._avo.to(dev)
#        self._R = self._R.to(dev)
#        self._T = self._T.to(dev)
#        self._C0 = self._C0.to(dev)
        if self.coupling:
            if self.dG_is_param:
                for i in range(len(self.params_k)):
                    self.params_k[i] = nn.Parameter(self.params_k[i].data.clone().detach().to(dev),requires_grad=True)
            else:
                self.params_kon = nn.Parameter(self.params_kon.data.clone().detach().to(dev), requires_grad=True)
        elif self.partial_opt and self.assoc_is_param:
            # self.params_kon = nn.Parameter(self.params_kon.data.clone().detach().to(dev), requires_grad=True)
            for i in range(len(self.params_kon)):
                self.params_kon[i]=nn.Parameter(self.params_kon[i].data.clone().detach().to(dev),requires_grad=True)
        elif self.homo_rates and self.assoc_is_param and not self.dG_is_param:
            self.params_kon = nn.Parameter(self.params_kon.data.clone().detach().to(dev), requires_grad=True)
        elif self.homo_rates and self.dG_is_param:
            for i in range(len(self.params_k)):
                self.params_k[i] = nn.Parameter(self.params_k[i].data.clone().detach().to(dev),requires_grad=True)
        else:
            self.kon = nn.Parameter(self.kon.data.clone().detach().to(dev), requires_grad=True)
        self.copies_vec = self.copies_vec.to(dev)
        self.initial_copies = self.initial_copies.to(dev)
        self.rxn_score_vec = self.rxn_score_vec.to(dev)
        self.dev = dev
        return self



    def compute_log_constants(self, kon: Tensor, dGrxn: Tensor, scalar_modifier) -> Tensor:
        """
        Returns log(k) for each reaction concatenated with log(koff) for each reaction
        """
        # above conversions cancel
        # std_c = Tensor([1e6])  # units umols / L
        l_kon = torch.log(kon)  # umol-1 s-1
        # l_koff = (dGrxn * scalar_modifier / (self._R * self._T)) + l_kon + torch.log(std_c)       #Units of dG in J/mol
        l_koff = (dGrxn * scalar_modifier) + l_kon + torch.log(self._C0)

        l_k = torch.cat([l_kon, l_koff], dim=0)
        # print("Rates: ",torch.exp(l_k))
        return l_k.clone().to(self.dev)
