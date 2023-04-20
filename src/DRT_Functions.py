#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:25:36 2023

@author: eric
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear, minimize
# from scipy.integrate import quad
from scipy.special import hermite

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
    return 

def quad(f, a, b, n=5):
    # Generate the quadrature rule coefficients
    x, w = np.polynomial.legendre.leggauss(n)
    
    # Scale the quadrature points and weights to the interval [a, b]
    x_scaled = 0.5 * (b - a) * x + 0.5 * (b + a)
    w_scaled = 0.5 * (b - a) * w
    
    # Evaluate the function at the quadrature points and sum up the weighted contributions
    integral = np.sum(w_scaled[:,None] * f(x_scaled),axis=0)
    
    return integral

def re_integrand(x, fn, fm, gaussian_sigma):
    x_diff = x[:,None] - fn[None,:] + fm[None,:]
    return Gaussian_Func(x, gaussian_sigma)[:,None] / (1 + np.exp(-2 * x_diff))

def im_integrand(x, fn, fm, gaussian_sigma):
    x_diff = x[:,None] - fn[None,:] + fm[None,:]
    return Gaussian_Func(x, gaussian_sigma)[:,None] * np.exp(-x_diff) / (1 + np.exp(-2 * x_diff))

def Gaussian_Func(x,sigma):
    y = np.exp(-x**2/(2*sigma**2)) / (2.5066*sigma)
    return y

def Gaussian_Derivative(x,n,sigma):
    hermite_poly = hermite(n)
    y = (-1)**n * Gaussian_Func(x,sigma) * hermite_poly(x/sigma) * (1/sigma)**n
    return y

def integrate_test_functions(log_fm,log_fn,gaussian_sigma):

    #Integrate the test functions
    fn_mesh, fm_mesh = np.meshgrid(log_fm, log_fn, indexing='ij')
    
    # reshape fn_mesh and fm_mesh into column vectors
    fn = fn_mesh.flatten()
    fm = fm_mesh.flatten()
    
    #Real integral
    # integral = np.array([quad(lambda y: Gaussian_Func(y,gaussian_sigma)*1/(1+np.exp(-2*(y-fn+fm))), -3*gaussian_sigma, 3*gaussian_sigma) for fn, fm in zip(fn_mesh.flatten(), fm_mesh.flatten())])
    integral = quad(lambda y: re_integrand(y,fn,fm,gaussian_sigma), -3*gaussian_sigma, 3*gaussian_sigma)
                                   
    A_re = integral.reshape(fn_mesh.shape)
    
    #Imag integral
    # integral = np.array([quad(lambda y: Gaussian_Func(y,gaussian_sigma)*np.exp(-(y-fn+fm))/(1+np.exp(-2*(y-fn+fm))), -3*gaussian_sigma, 3*gaussian_sigma) for fn, fm in zip(fn_mesh.flatten(), fm_mesh.flatten())])    
    integral = quad(lambda y: im_integrand(y,fn,fm,gaussian_sigma), -3*gaussian_sigma, 3*gaussian_sigma)
    
    A_im = integral.reshape(fn_mesh.shape)
    
    return A_re, A_im

def cost_func(x,f_m,A_re,A_im,b,norm_matrix):
    
    x_re = x[0:A_re.shape[1]]
    x_im = x[A_re.shape[1]:A_re.shape[1]+A_im.shape[1]]
    R = x[A_re.shape[1]+A_im.shape[1]]
    L = x[A_re.shape[1]+A_im.shape[1]+1]
        
    Z_sim_re = np.matmul(A_re,x_re)+R
    Z_sim_im = np.matmul(A_im,x_im)+np.exp(f_m)*L
    
    Z_exp_re = np.real(b)
    Z_exp_im = np.imag(b)


    
    Re_cost = 1.0*np.sum(np.abs(Z_exp_re-Z_sim_re))
    Im_cost = 1.0*np.sum(np.abs(Z_exp_im-Z_sim_im))
    Re_norm_cost = np.abs(np.matmul(x_re,np.matmul(norm_matrix,x_re.T)))/(x_re.shape[0]**2)
    Im_norm_cost = np.abs(np.matmul(x_im,np.matmul(norm_matrix,x_im.T)))/(x_im.shape[0]**2)
    
    cost = Re_cost + Im_cost + Re_norm_cost + Im_norm_cost

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
    
def norm_integrand(x, fn, fm, k, gaussian_sigma):
    x_diff = x[:,None] - fn[None,:] + fm[None,:]
    return Gaussian_Derivative(x_diff,k,gaussian_sigma)*Gaussian_Derivative(x,k,gaussian_sigma)[:,None]
    
def DRT_Fit(measured_frequencies, measured_Z, fitting_frequencies, reg_params):
    
    f_m = measured_frequencies
    f_n = fitting_frequencies
    Z_m = measured_Z
       
    reg_order = reg_params.shape[0]
    gaussian_width = (np.max(fitting_frequencies)-np.min(fitting_frequencies))/fitting_frequencies.shape[0]
        
    #Generate parameter array
    #real, imag, resistance
    x_params = np.zeros((2*fitting_frequencies.shape[0]+2))
    
    #Integrate the test functions wrt the measured frequencies
    A_re, A_im = integrate_test_functions(measured_frequencies, fitting_frequencies, gaussian_width/2.355)
    
    #Calculate the regularization matrix
    fn_mesh, fm_mesh = np.meshgrid(f_n, f_n, indexing='ij')
    
    fn = fn_mesh.flatten()
    fm = fm_mesh.flatten()
    
    M = np.zeros((fitting_frequencies.shape[0],fitting_frequencies.shape[0]))
    for k in range(reg_order):
        integral = quad(lambda y: norm_integrand(y,fn,fm,k,gaussian_width/2.355), -3*gaussian_width/2.355, 3*gaussian_width/2.355)
        M += reg_params[k]*integral.reshape(fn_mesh.shape).T
    
    
    #Fit to experimental data
    b = Z_m
    
    #Solve for parameters
    res = minimize(cost_func,x_params,args=(f_m,A_re,A_im,b,M))
    x_params = res['x']
    
    #Calculate the gamma function coefficients
    x_re = x_params[:fitting_frequencies.shape[0]]
    x_im = x_params[fitting_frequencies.shape[0]:-2]
    R = x_params[-2]
    L = x_params[-1]
    
    return x_re, x_im, R, L
    
# #%%

# #Close any open graphs
# plt.close('all')

# """Data pre-processing"""

# #PARAMS
# path = '/home/eric/Documents/Fermilab_Superconductors/Elechtrochemistry_Project/Data/EP-Data/Half-Cell_EP/12-20-2022-SPEIS-0,5to4V-10C_C01.txt'
# # path = 'S75_SPEIS_0,5-1V_21C_trial2_C01.txt'


# #Load txt file into numpy array
# Data = np.loadtxt(path, skiprows=1)

# #Remove junk data
# Data = Data[Data[:,0]!=0,:]
# Data = Data.T

# #Determine the number of potential steps
# potential_steps = int(np.max(Data[3,:]))

# #Determine the number of frequency steps
# frequency_steps = int(Data.shape[1]/potential_steps)


# #Divide the data based on potential step

# Freq = 2*np.pi*Data[0,Data[3,:]==1]
# ReZ = Data[1,Data[3,:]==1]
# ImZ = -Data[2,Data[3,:]==1]

# Electrode_Potential = np.mean(Data[4,Data[3,:]==1])
# Current = Data[5,Data[3,:]==1][-1]
    


# Z = ReZ + 1j*ImZ


# #%%

# """Model Fitting"""



# #PARAMS
# f_min = np.min(Freq)
# f_max = np.max(Freq)
# fitting_params_per_decade = 2



# #Convert to log frequency
# log_f_min = np.log(f_min)
# log_f_max = np.log(f_max)

# #Generate the fitting test function offsets
# number_of_fitting_params = int(fitting_params_per_decade*(log_f_max-log_f_min))
# log_f_n = np.linspace(log_f_min,log_f_max,number_of_fitting_params)

# #Convert experimental frequencies to log space
# log_Freq = np.log(Freq)

# #Generate regularization hyper-parameter array
# reg_params = np.array((1e-7,1e-7,1e-7))

   
# x_re, x_im, R, L = DRT_Fit(log_Freq, Z, log_f_n, reg_params)
    
# x_pos = (x_re + x_im)/2
# x_neg = (x_re - x_im)/2
    
    




# #%%

# # y = np.linspace(-100*gaussian_width, 100*gaussian_width,num = 10000)
# # fn = log_f_n[20]
# # fm = log_Freq[10]
# # integrand = Gaussian_Func(y,gaussian_width)*np.exp(-(y-fn-fm))/(1+np.exp(-2*(y-fn-fm)))

# def Bode_Plot(Z_exp=None,Z_sim=None,title=None):

    
#     fig, ax = plt.subplots(2,1,figsize = (4,5),dpi=200)

#     ax[0].set_title(title)
    
#     if Z_sim is not None:
#         freq = Z_sim[0]
#         ReZ = np.real(Z_sim[1])
#         ImZ = np.imag(Z_sim[1])
#         ax[0].plot(freq,ReZ,lw=0.5)
#         ax[1].plot(freq,ImZ,lw=0.5)

        
#     if Z_exp is not None:
#         freq = Z_exp[0]
#         ReZ = np.real(Z_exp[1])
#         ImZ = np.imag(Z_exp[1])
#         ax[0].scatter(freq,ReZ,s=5,marker='x',lw=0.5)
#         ax[1].scatter(freq,ImZ,s=5,marker='x',lw=0.5)

#     ax[0].set_xscale('log')
#     # ax[0].set_xlabel('Frequency [rad/s]')
#     ax[0].set_ylabel('Real Impedence [Ohm]')
        
#     ax[1].set_xscale('log')
#     ax[1].set_xlabel('Frequency [rad/s]')
#     ax[1].set_ylabel('Imaginary Impedence [Ohm]')
    
#     plt.tight_layout()
    
    
    
# #Plot Bode
# #Params
# plot_freq_min = f_min
# plot_freq_max = f_max
# plot_points_per_decade = 5


# #Convert to log
# log_plot_freq_min = np.log(plot_freq_min)
# log_plot_freq_max = np.log(plot_freq_max)
# plot_points = int(plot_points_per_decade*(log_plot_freq_max-log_plot_freq_min))

# g_width = (log_f_max-log_f_min)/number_of_fitting_params

# #Generate plotting frequencies
# log_plot_freq = np.linspace(log_plot_freq_min,log_plot_freq_max,plot_points)
# pos_plot_freq = np.exp(log_plot_freq)
# plot_freq = np.concatenate((np.flip(-pos_plot_freq),pos_plot_freq))

# A_re, A_im = integrate_test_functions(log_plot_freq, log_f_n, g_width/2.355)

# Z_sim = R + 1j*pos_plot_freq*L + np.matmul(A_re,x_re) + 1j*np.matmul(A_im,x_im)
# Z_exp = Z

# # Bode_Plot(Z_exp=np.array((Freq[:,i],Z_exp)),Z_sim=np.array((pos_plot_freq,Z_sim)),title=Electrode_Potential[i])


# Bode_Plot(Z_exp=np.array((Freq,Z_exp)),Z_sim=np.array((pos_plot_freq,Z_sim)),title='Half-Cell Cavity EP')





# gamma_pos = np.sum(x_pos[None,:]*Gaussian_Func(log_plot_freq[:,None]-log_f_n[None,:],g_width),axis=1)
# gamma_neg = np.sum(x_neg[None,:]*Gaussian_Func(log_plot_freq[:,None]-log_f_n[None,:],g_width),axis=1)

# gamma = np.concatenate((np.flip(gamma_neg),gamma_pos))

# fig, ax = plt.subplots(dpi=120)

# ax.plot(plot_freq,gamma)

# ax.set_title('Polishing Voltage = '+str(round(Electrode_Potential,2))+' [V]')
# ax.set_xscale('symlog')
# ax.set_xlabel('Frequency [rad/s]')
# ax.set_ylabel('DRT [Ohm*s]')
