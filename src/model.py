import numpy as np
from scipy.optimize import minimize
from typing import Callable

class Function:
    """Base class for all functions used in the EIS model fitting algorithm. Must contain two python functions that evaluate to the function itself and to its jacobian."""


    def __init__(self, function: Callable, jacobian: Callable, arguments: dict):
        self.function = function
        self.jacobian = jacobian
        self.arguments = arguments

    def evaluate(self,x):

        return self.function(x,**self.arguments)
    
    def evaluate_jacobian(self,x):
        
        return self.jacobian(x,**self.arguments)


class Model:
    """An electrochemical model that consists of a cost function to be minimized and a fit function that estimates the experimental values based on the fitting parameters."""

    def __init__(self, cost_functions: list[Function], fit_functions: list[Function], initial_guess: list):
        self.cost_functions = cost_functions
        self.fit_functions = fit_functions
        self.fit_parameters = initial_guess

    def cost(self,x):
        cost = 0

        for function in self.cost_functions:
            cost += function.evaluate(x)

        return cost
    
    def cost_jacobian(self,x):
        cost_jacobian = 0

        for function in self.cost_functions:
            cost_jacobian += function.jacobian(x)

        return cost_jacobian

    def fit(self):

        if len(self.fit_parameters) > 1:
            params = np.concatenate(*self.fit_parameters)
        else:
            params = self.fit_parameters[0]

        #Solve for parameters
        res = minimize(self.cost,
                       params,
                       method='BFGS',
                       jac=self.cost_jacobian)
        self.fit_parameters = res['x']

        return self.fit_parameters


class Least_Squares(Function):

    def __init__(self, Z_exp_re, Z_exp_im, fit_function: Function):

        self.Z_exp_re = Z_exp_re
        self.Z_exp_im = Z_exp_im
        self.fit_function = fit_function

    
    def evaluate(self,x):
    
        Z_sim_re, Z_sim_im = self.fit_function.evaluate(x)

        Re_cost = np.linalg.norm((self.Z_exp_re-Z_sim_re))
        Im_cost = np.linalg.norm((self.Z_exp_im-Z_sim_im))
        
        cost = Re_cost + Im_cost

        return cost
    
    def jacobian(self,x):

        Z_sim_re, Z_sim_im = self.fit_function.evaluate(x)

        dZ_sim_re, dZ_sim_im = self.fit_function.jacobian(x)

        dRe_cost = -2*np.sum((self.Z_exp_re-Z_sim_re)[:,None]*dZ_sim_re,axis=0)
        dIm_cost = -2*np.sum((self.Z_exp_im-Z_sim_im)[:,None]*dZ_sim_im,axis=0)

        dcost = dRe_cost + dIm_cost

        return dcost
    
class Normalization(Function):

    def __init__(self, norm_matrix):

        self.norm_matrix = norm_matrix

    
    def evaluate(self,x):
        Re_norm_cost = (np.matmul(x,np.matmul(self.norm_matrix,x.T)))
        Im_norm_cost = (np.matmul(x,np.matmul(self.norm_matrix,x.T)))

        return Re_norm_cost + Im_norm_cost
    
    def jacobian(self,x):
    
        dRe_norm_cost = np.matmul(x,self.norm_matrix)+np.matmul(self.norm_matrix,x.T)
        dIm_norm_cost = np.matmul(x,self.norm_matrix)+np.matmul(self.norm_matrix,x.T)

        return dRe_norm_cost + dIm_norm_cost

class DRT_Fit(Function):

    def __init__(self,A_re,A_im):

        self.A_re = A_re
        self.A_im = A_im



    def evaluate(self,x):

        Z_sim_re = np.matmul(self.A_re,x)
        Z_sim_im = np.matmul(self.A_im,x)

        return Z_sim_re, Z_sim_im    
        
    def jacobian(self,x):

        return self.A_re, self.A_im
























from scipy.special import hermite
import matplotlib.pyplot as plt

def Gaussian_Func(x,sigma):
    y = np.exp(-x**2/(2*sigma**2)) / (2.5066282*sigma)
    return y

def Gaussian_Derivative(x,n,sigma):
    hermite_poly = hermite(n)
    y = (-1)**n * Gaussian_Func(x,sigma) * hermite_poly(x/(1.41421356*sigma))
    return y

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

def norm_integrand(x, fn, fm, k, gaussian_sigma):
    x_diff = x[:,None] - fn[None,:] + fm[None,:]
    return Gaussian_Derivative(x_diff,k,gaussian_sigma)*Gaussian_Derivative(x,k,gaussian_sigma)[:,None]

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





import pandas as pd
import numpy as np

import os
import re

# get the list of files and directories in the raw data  directory
directory = 'Data/Nb_Strip/'
text_files = os.listdir(directory)
file = text_files[0]



data = pd.read_csv(directory+file,delimiter='\t').drop('Unnamed: 6', axis=1)
data = data[data['freq/Hz']!=0.0]
data = data.astype({'cycle number': int})




cycle_data = data[data['cycle number'] == 1]

#Load data
Freq = np.array(cycle_data['freq/Hz'])
Z_exp_re = np.array(cycle_data['Re(Z)/Ohm'])
Z_exp_im = np.array(cycle_data['-Im(Z)/Ohm'])




#PARAMS
f_min = np.min(Freq)/1000
f_max = np.max(Freq)
fitting_params_per_decade = 12


#Convert to log frequency
log_f_min = np.log(f_min)
log_f_max = np.log(f_max)

#Generate the fitting test function offsets
number_of_fitting_params = int(fitting_params_per_decade*(log_f_max-log_f_min))
log_f_n = np.linspace(log_f_min,log_f_max,number_of_fitting_params)
gaussian_width = (np.max(log_f_n)-np.min(log_f_n))/log_f_n.shape[0]

#Convert experimental frequencies to log space
log_Freq = np.log(Freq)


f_m = log_Freq
f_n = log_f_n    

A_re, A_im = integrate_test_functions(f_m, f_n, gaussian_width/2.355)


fit_function = DRT_Fit(A_re,A_im)

cost_function = Least_Squares(Z_exp_re,Z_exp_im,fit_function)



#Calculate the regularization matrix
fn_mesh, fm_mesh = np.meshgrid(f_n, f_n, indexing='ij')

fn = fn_mesh.flatten()
fm = fm_mesh.flatten()

reg_params = np.array((5e-2,5e-2,5e-3))

M = np.zeros((number_of_fitting_params,number_of_fitting_params))
for k in range(reg_params.shape[0]):
    integral = quad(lambda y: norm_integrand(y,fn,fm,k,gaussian_width/2.355), -3*gaussian_width/2.355, 3*gaussian_width/2.355)
    M[:f_n.shape[0],:f_n.shape[0]] += reg_params[k]*integral.reshape(fn_mesh.shape).T



norm_function = Normalization(M)

parameters = np.zeros((number_of_fitting_params))


new_model = Model((cost_function,norm_function),(fit_function,),(parameters,))

result = new_model.fit()

plt.figure()
plt.plot(result)
plt.savefig('test.png')