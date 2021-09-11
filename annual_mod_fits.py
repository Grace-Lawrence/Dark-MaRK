import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import unicodedata

__all__ = ["fit_curve"]

def fit_func_S(t, S_0, S_m,t0):
    """Functional form for annual modulation from Bernabei et al. (2008) with 
    seperate S0, Sm terms"""
    return(S_0 + S_m*np.cos(((2*np.pi)/365)*(t-t0)))

def fit_func_A(t,A,t0):
    """Functional form for annual modulation"""
    return(A*np.cos(((2*np.pi)/365)*(t-t0)))


def fit_curve(y_data, fit, verbose = False):
    """Using scipy.optimize.curve_fit to find fit parameters for annual 
    modulation curves"""
    x_data = np.linspace(0.001,365,365,dtype=int)
    if fit == 'A': #Subtracting the median from data to examine the residuals
        y_data = np.subtract(y_data,np.mean(y_data))
        params, params_covariance = optimize.curve_fit(fit_func_A, x_data, y_data, p0 = [0.023,152.5]) #p0 values can be altered
        if verbose:
            print(f'A: Modulation Amplitude ={params[0]},\nt0: The phase = {params[1]}')
            plt.clf()
            plt.figure(figsize=(6, 4))
            plt.scatter(x_data, y_data, label='Data')
            plt.plot(x_data, fit_func_A(x_data, params[0], params[1]),label='Fitted function', color='red')
            plt.xlabel('Time (d)')
            plt.ylabel(r'$ \frac{dR}{dE_r} (dru)$')
            plt.title('Residual Count Rate')
            plt.legend()
            plt.show()
    elif fit == 'S': #Fitting both the constant rate S0 and the modulation S_0
        max_day = np.where(y_data == y_data.max())[0]
        params, params_covariance = optimize.curve_fit(fit_func_S, x_data, y_data, p0 = [1,1,max_day], bounds=([-5,-5,0], [5., 5., 365])) #bounds can be altered
        perr = np.sqrt(np.diag(params_covariance))
        if verbose:
            print(f"S_0: Constant part of the signal = {params[0]} " + unicodedata.lookup("Plus-Minus Sign") +f" {perr[0]}") 
            print(f"S_m: Modulation Amplitde = {params[1]} " + unicodedata.lookup("Plus-Minus Sign") +f" {perr[1]}") 
            print(f"t_0: The phase = {params[2]} " + unicodedata.lookup("Plus-Minus Sign") +f" {perr[2]}") 
    return params, params_covariance

