#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:31:41 2023

@author: eric
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.transforms import Bbox
from scipy.optimize import lsq_linear, minimize
from scipy.integrate import quad

from DRT_Functions import DRT_Fit, integrate_test_functions, Gaussian_Func

def RLC_Admittance(freq,res_freq,band_width):
    #Admittance of a parallel RLC circuit with R = 1
    Y_L = res_freq[None,:]**2/(1j*freq[:,None]*band_width[None,:])
    Y_C = 1j*freq[:,None]/band_width[None,:]
    Y_RLC = 1 + Y_L + Y_C
    return Y_RLC

def RL_Admittance(freq,rel_freq):
    #Admittance of a parallel RL circuit with R = 1
    Y_L = rel_freq[None,:]/(1j*freq[:,None])
    Y_RL = 1+Y_L
    return Y_RL

def RC_Admittance(freq,rel_freq):
    #Admittance of a parallel RC circuit with R = 1
    # Y_C = 1j*freq[:,None]/rel_freq[None,:]
    # Y_RC = 1+Y_C
    # return Y_RC
    Y_C = 1j*freq/rel_freq
    Y_RC = 1+Y_C
    return Y_RC

# def Gaussian_Func(x,a):
#     y = np.exp(-(a/1.662*x**2))
#     return y

def cost_func(x,A_re,A_im,b,norm_param):
    
    x_re = x[0:A_re.shape[1]]
    x_im = x[A_re.shape[1]:A_re.shape[1]+A_im.shape[1]]
    R = x[A_re.shape[1]+A_im.shape[1]]
        
    Z_sim_re = np.matmul(A_re,x_re)+R
    Z_sim_im = np.matmul(A_im,x_im)
    
    Z_exp_re = np.real(b)
    Z_exp_im = np.imag(b)


    
    Re_cost = 1.0*np.sum(np.abs(Z_exp_re-Z_sim_re))
    Im_cost = 1.0*np.sum(np.abs(Z_exp_im-Z_sim_im))
    # Norm_cost = norm_param/x.shape[0]*(np.sum(np.convolve(x,np.array((-1/560,8/315,-1/5,8/5,-205/72,8/5,-1/5,8/315,-1/560)))**2)+1e-1*np.sum(np.exp(np.abs(x))))
    Norm_cost = norm_param*np.sum(np.abs(x[0:-1]))
    
    cost = Re_cost + Im_cost + Norm_cost
    
    
    return cost

def Bode_Plot(Z_exp=None,Z_sim=None,title=None):

    
    fig, ax = plt.subplots(2,1,figsize = (4,5),dpi=200)

    ax[0].set_title(title)
    
    if Z_sim is not None:
        freq = Z_sim[0]
        ReZ = np.real(Z_sim[1])
        ImZ = np.imag(Z_sim[1])
        ax[0].plot(freq,ReZ,lw=0.5)
        ax[1].plot(freq,ImZ,lw=0.5)

        
    if Z_exp is not None:
        freq = Z_exp[0]
        ReZ = np.real(Z_exp[1])
        ImZ = np.imag(Z_exp[1])
        ax[0].scatter(freq,ReZ,s=5,marker='x',lw=0.5)
        ax[1].scatter(freq,ImZ,s=5,marker='x',lw=0.5)

    ax[0].set_xscale('log')
    # ax[0].set_xlabel('Frequency [rad/s]')
    ax[0].set_ylabel('Real Impedence [Ohm]')
        
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Frequency [rad/s]')
    ax[1].set_ylabel('Imaginary Impedence [Ohm]')
    
    plt.tight_layout()



#%%

#Close any open graphs
plt.close('all')

"""Data pre-processing"""

#PARAMS
path = '/home/eric/Documents/Fermilab_Superconductors/Elechtrochemistry_Project/Data/EP-Data/Half-Cell_EP/12-20-2022-SPEIS-0,5to4V-10C_C01.txt'
# path = '/home/eric/Documents/Fermilab_Superconductors/Elechtrochemistry_Project/Data/EP-Data/Half-Cell_EP/12-20-2022-SPEIS-5to2V-10C_C01.txt'

# path = '/home/eric/Documents/Fermilab_Superconductors/Elechtrochemistry_Project/Data/EP-Data/DRT_Analysis_Data/Data_Export/S67_SPEIS_0-4V_13C_Trial3_C01.txt'
# path = '/home/eric/Documents/Fermilab_Superconductors/Elechtrochemistry_Project/Data/EP-Data/DRT_Analysis_Data/Data_Export/S68_SPEIS_0-4V_13C_C01.txt'
# path = '/home/eric/Documents/Fermilab_Superconductors/Elechtrochemistry_Project/Data/EP-Data/DRT_Analysis_Data/Data_Export/S75_SPEIS_0-4V_21C_trial1_C01.txt'
# path = '/home/eric/Documents/Fermilab_Superconductors/Elechtrochemistry_Project/Data/EP-Data/DRT_Analysis_Data/Data_Export/S35_SPEIS_0-4V_21C_trial1_C01.txt'

# path = './S75_SPEIS_0,5-1V_21C_trial2_C01.txt'

#Load txt file into numpy array
Data = np.loadtxt(path, skiprows=1)

#Remove junk data
Data = Data[Data[:,0]!=0,:]
Data = Data.T

#Determine the number of potential steps
potential_steps = int(np.max(Data[3,:]))
# potential_steps = 2

#Determine the number of frequency steps
frequency_steps = int(Data.shape[1]/potential_steps)

#Create empty arrays for data
Freq = np.zeros((frequency_steps,potential_steps))
ReZ = np.zeros((frequency_steps,potential_steps))
ImZ = np.zeros((frequency_steps,potential_steps))

Electrode_Potential = np.zeros((potential_steps))
Current = np.zeros((potential_steps))
Current_min = np.zeros((potential_steps))
Current_max = np.zeros((potential_steps))

#Divide the data based on potential step
for i in range(1,potential_steps+1):
    Freq[:,i-1] = 2*np.pi*Data[0,Data[3,:]==i]
    ReZ[:,i-1] = Data[1,Data[3,:]==i]
    ImZ[:,i-1] = -Data[2,Data[3,:]==i]
    
    Electrode_Potential[i-1] = np.mean(Data[4,Data[3,:]==i])
    Current[i-1] = Data[5,Data[3,:]==i][-1]
    Current_min[i-1] = np.min(Data[5,Data[3,:]==i])
    Current_max[i-1] = np.max(Data[5,Data[3,:]==i])
    




Z = ReZ + 1j*ImZ





#%%

"""Model Fitting"""

#PARAMS
f_min = np.min(Freq)
f_max = np.max(Freq)
fitting_params_per_decade = 5


#Convert to log frequency
log_f_min = np.log(f_min)
log_f_max = np.log(f_max)

#Generate the fitting test function offsets
number_of_fitting_params = int(fitting_params_per_decade*(log_f_max-log_f_min))
log_f_n = np.linspace(log_f_min,log_f_max,number_of_fitting_params)
gaussian_width = (np.max(log_f_n)-np.min(log_f_n))/log_f_n.shape[0]

#Convert experimental frequencies to log space
log_Freq = np.log(Freq)

# #Generate parameter array
# #real, imag, resistance
# x_params = np.zeros((2*log_f_n.shape[0]+1,potential_steps))+1




x_re = np.zeros((number_of_fitting_params,potential_steps))
x_im = np.zeros((number_of_fitting_params,potential_steps))
R = np.zeros((potential_steps))
L = np.zeros((potential_steps))


import time

#Generate regularization hyper-parameter array
reg_params = np.array((1e-4,1e-2,5e-2))

for i in range(potential_steps):
    # start the timer
    start_time = time.time()
    
    x_re[:,i], x_im[:,i], R[i], L[i] = DRT_Fit(log_Freq[:,i], Z[:,i], log_f_n, reg_params)
    

    # end the timer
    end_time = time.time()
    
    # calculate the elapsed time
    elapsed_time = end_time - start_time
    
    # print the elapsed time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    
    # #Integrate the test functions
    # fn_mesh, fm_mesh = np.meshgrid(log_Freq[:,i], log_f_n)
    
    # #Real integral
    # integral = np.array([quad(lambda y: Gaussian_Func(y,gaussian_width)*1/(1+np.exp(-2*(y-fn-fm))), -100*gaussian_width, 100*gaussian_width)[0] for fn, fm in zip(fn_mesh.flatten(), fm_mesh.flatten())])
                                       
    # A_re = integral.reshape(fn_mesh.shape).T
    
    # #Imag integral
    # integral = np.array([quad(lambda y: Gaussian_Func(y,gaussian_width)*np.exp(-(y-fn-fm))/(1+np.exp(-2*(y-fn-fm))), -100*gaussian_width, 100*gaussian_width)[0] for fn, fm in zip(fn_mesh.flatten(), fm_mesh.flatten())])    
    
    # A_im = integral.reshape(fn_mesh.shape).T
    
    # #Fit to experimental data
    # b = Z[:,i]

    # #Solve for parameters
    # res = minimize(cost_func,x_params[:,i],args=(A_re,A_im,b,1e-3))
    # x_params[:,i] = res['x']




#%%
x_pos = (x_re - x_im)/2
x_neg = (x_re + x_im)/2
"""Plotting"""

# Plot gamma functions
#Params
plot_freq_min = 1e0
plot_freq_max = 1e6
plot_points_per_decade = 10


#Convert to log
log_plot_freq_min = np.log(plot_freq_min)
log_plot_freq_max = np.log(plot_freq_max)
plot_points = int(plot_points_per_decade*(log_plot_freq_max-log_plot_freq_min))

#Generate plotting frequencies
log_plot_freq = np.linspace(log_plot_freq_min,log_plot_freq_max,plot_points)

pos_plot_freq = np.exp(log_plot_freq)
plot_freq = np.concatenate((np.flip(-pos_plot_freq),pos_plot_freq))


gamma_pos = np.sum(x_pos[None,:,:]*Gaussian_Func(log_plot_freq[:,None,None]-log_f_n[None,:,None],gaussian_width),axis=1)
gamma_neg = np.sum(x_neg[None,:,:]*Gaussian_Func(log_plot_freq[:,None,None]-log_f_n[None,:,None],gaussian_width),axis=1)

gamma = np.concatenate((np.flip(gamma_neg),gamma_pos))

fig, (ax1, ax2) = plt.subplots(ncols=2,dpi=300,sharey=True,gridspec_kw={'width_ratios': [1, 5]})

X,Y = np.meshgrid(plot_freq,Electrode_Potential,indexing='ij')

mesh = ax2.pcolormesh(X,Y,gamma,shading='flat', cmap='viridis')
mesh.set_norm(SymLogNorm(linthresh=1e0,vmin=-2e1,vmax=2e1))
cbar = fig.colorbar(mesh)

ax2.set_xscale('symlog')
ax2.set_xlabel('Frequency [rad/s]')
ax2.set_xticks([-1e5,-1e3,-1e1,0,1e1,1e3,1e5])
ax2.set_ylim(0,5)

ax1.plot(Current/1000,Electrode_Potential)
ax1.fill_betweenx(Electrode_Potential, Current_min/1000, Current_max/1000, alpha = 0.2)
ax1.set_ylabel('Polishing Voltage [V]')
ax1.set_xlabel('Current [A]')
ax1.invert_xaxis()
# ax1.set_xticks([0.1,0.2])
ax1.set_yticks(np.arange(0,6,1))
# ax1.set_xlim(0.2,0)

fig.subplots_adjust(wspace = 0)

#%%

# #Plot Bode

# for i in range(potential_steps):
#     #Integrate the test functions
#     A_re, A_im = integrate_test_functions(log_plot_freq,log_f_n,gaussian_width/2.355)
    
#     Z_sim = R[i] + 1j*pos_plot_freq*L[i] + np.matmul(A_re,x_re[:,i]) + 1j*np.matmul(A_im,x_im[:,i])
#     Z_exp = Z[:,i]
    
#     # Bode_Plot(Z_exp=np.array((Freq[:,i],Z_exp)),Z_sim=np.array((pos_plot_freq,Z_sim)),title=Electrode_Potential[i])
    
    
#     Bode_Plot(Z_exp=np.array((Freq[:,i],Z_exp)),Z_sim=np.array((pos_plot_freq,Z_sim)),title=Electrode_Potential[i])
    
    
    
    
    

