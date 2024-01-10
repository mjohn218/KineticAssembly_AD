
from KineticAssembly_AD.vec_sim import VecSim
from KineticAssembly_AD import ReactionNetwork
import numpy as np

from torch import DoubleTensor as Tensor
import torch
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import random
from scipy import signal
import sys
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline


class TrapMetric:

    def __init__(self,sim: VecSim ):

        self.sim_class = sim

    def calc_timeEQ(self,eq_conc,time,conc,ini_conc,thresh=0.1):
        eq_yield = eq_conc*100/ini_conc
        for i in range(len(time)):
            curr_yield = conc[i]*100/ini_conc
            if abs(curr_yield-eq_yield) <= thresh:
                return(time[i])
        return(time[-1])


    def calc_slope(self,time,conc,mode='delta'):
        #There are 3 modes to calc slopes
        #Mode 1 - "delta" : mode which is just ratio of finite differences
        #Mode 2 - "log" : Gradient calc using numpy gradient function, but time is in logspace
        #Mode 3: "regular" : Gradient calc with time in normal space

        if mode=="delta":
            slopes=[]
            for i in range(len(time)-1):
                delta_c = conc[i+1]=conc[i]
                delta_t = np.log(time[i+1]-time[i])

                s = delta_c/delta_t
                slope.append(s)

            return(slopes)
        elif mode=='log':
            l_grad = np.gradient(conc,np.log(time))
            return(l_grad)
        elif mode == "regular":
            grad = np.gradient(conc,time)
            return(grad)

    def do_interpool(self,time,conc_complx,conc_mon,inter_gap=3):


        #If the conc profile requires interpolation of some points, most likely the trapped region,
        #First finding start and end time points of the trapped region

        slopes_complx = self.calc_slope(time,conc_complx)
        slopes_mon = self.calc_slope(time,conc_mon)

        #Find time point when slope of monmomer becomes zero.
        #In trapped system this is also when conc of monomer is close to zero
        #To get the zero point, first normalizing the slopes and taking only absolute values
        #Here it can get tricky to define what is zero. Have to choose a proper threshold.
        norm_slopes_mon = slopes_mon/abs(np.min(slopes_mon))
        abs_slopes_mon = np.absolute(norm_slopes_mon)
        mask = abs_slopes_mon < 1e-6
        inter_time_start = time[:-1][mask][0]
        indx_start = np.argwhere(mask)[0][0]

        #To find the end point, we have to check when the slope of dimer or trimer increases.
        #But it is not robust. Something to do for later.
        #Currently just using user defined parameter than can be adjusted by looking at the conc. profile.
        indx_end = indx_start + inter_gap


        #Defining the interpolation time range
        time_inter = np.logspace(np.log10(time[indx_start]),np.log10(time[indx_end]),num=2,endpoint=True)

        complx_inter = UnivariateSpline(time[indx_start:indx_end],conc_complx[indx_start:indx_end],k=1,s=0.1)

        new_time = np.concatenate((time[:indx_start+1],time_inter[10:-10],time[indx_end:]))
        new_conc_complx = np.concatenate((conc[:indx_start+1],complx_inter(time_new[10:-10]),conc_complx[indx_end:]))

        return(new_time,new_conc_complx)

    def get_time_bounds(self,time,eq_time,grad,l_grad,l_grad2,tan_thresh=89,n_traps=1):
        peak_indx = np.argmax(l_grad)
        mask_eq = time < eq_time
        eq_indx = np.argwhere(mask_eq)[-1][0]
        flag_min = False
        flag_max = True
        count=0
        for i in range(len(l_grad2)):
            if flag_max and l_grad2[i]<0:
                flag_min=True
                flag_max=False
                first_peak=time[i]
            if flag_min and l_grad2[i]>0:
                split_time = time[i]
                split_indx=i
                flag_min=False
                flag_max=True
                count+=1

                if count==n_traps:
                    break
        second_region_grad = l_grad[split_indx:eq_indx]

        # tan_inv = np.degrees(np.arctan(second_region_grad))
        # tan_thresh = np.degrees(np.arctan(np.max(l_grad[:split_indx])))
        # mask_inf = np.where(tan_inv>=tan_thresh)
        # second_region_grad = second_region_grad[:mask_inf[0][0]]

        second_peak_indx = np.argmax(second_region_grad)
        second_peak = time[split_indx+second_peak_indx]

        return(first_peak,split_time,second_peak)

    def clean_data(self,time,l_grad,thresh_freq=1,bin_num=50,mode='hist'):

        if mode == 'conc_step':
            #In this mode we ignore data points which occur immediately after
            #a step change in conc_scale parameter in the simulations. These points
            #discontinouous and are outliers in gradient calculation.

            #To find timepoints for step change, we measure the time point where the
            #step change in negative
            step_size=[]
            for i in range(len(time)-1):
                delta = time[i+1]-time[i]
                step_size.append(delta)
            remove_indx = []
            for i in range(len(step_size)-1):
                delta = step_size[i+1]-step_size[i]
                if delta < 0:
                    remove_indx.append(i)
            mask_bool = np.ones((len(time)),dtype='bool')
            for i in range(len(remove_indx)):
                mask_bool[remove_indx[i]:remove_indx[i]+5]=False
            new_time_arr = time[mask_bool]
            l_grad_new = l_grad[mask_bool]

            return(new_time_arr,l_grad_new)
        elif mode =='hist':
            data=np.histogram(l_grad,bins=bin_num)
            # print(data)
            flag=False
            count=0
            bin_val_min=0
            bin_val_max=0
            for i in range(len(data[0])):
                if data[0][i] >=10 and not flag:
                    flag=True
                    count+=1
                    bin_val_min = data[1][i]
                elif data[0][i] <=10 and flag:
                    count+=1
                    bin_val_max=data[1][i]
                    break

            mask_out = (l_grad <= bin_val_max) & (l_grad >= bin_val_min)
            new_time = np.array(time)[mask_out]
            l_grad_new = l_grad[mask_out]

            return(new_time,l_grad_new)


    def calc_lag(self,time,conc_complx,eq_conc):

        ini_conc = self.sim_class.rn.initial_copies[0].item()
        grad = self.calc_slope(time,conc_complx,mode='regular')
        l_grad = self.calc_slope(time,conc_complx,mode='log')

        clean_time,clean_grad = self.clean_data(time,l_grad,mode='conc_step')
        l_grad2 = self.calc_slope(clean_time,clean_grad,mode='log')
        clean_time,clean_grad2 = self.clean_data(clean_time,l_grad2,mode='hist')
        eq_time = self.calc_timeEQ(eq_conc,time,conc_complx,ini_conc)

        time_bounds = self.get_time_bounds(clean_time,eq_time,grad,clean_grad,clean_grad2)

        if abs(np.log(time_bounds[2]/time_bounds[1])) < 1e-2 :
            lag_time = 0
        else:
            lag_time = np.log(time_bounds[2]/time_bounds[0])

        return(lag_time,time_bounds)
