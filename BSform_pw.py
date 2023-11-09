# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Article: "Explicit approximations of option prices via Malliavin calculus in a
general stochastic volatility framework"

Description: Computes the usual Black Scholes Put/Call formula for piecewise-constant
parameters.

"""


import copy as cp
from math import sqrt, exp, log
from collections import deque
from scipy.stats import norm 

def BSform_pw(S0, sig, Strk, _rd_deque, _rf_deque, _dt, option):

    """
    Computes the usual Black Scholes Put/Call formula for piecewise-constant
    parameters.

    S0 (float): initial spot.
    sig (float): initial volatility.
    Strk (float): strike value of the contract.
    rd_deque (deque): domestic interest rate, given backward, e.g., rd_deque = deque([rd2, rd1]).
    rf_deque (deque): foreign interest rate, given backward, e.g., rf_deque = deque([rf2, rf1]).
    dt (deque): deque of time increments over which each parameter is 'alive',
        given backward, e.g., dt = deque([dt2, dt1]). Note sum(dt) gives option maturity T.
    option (str): 'Put' or 'Call'.
    """

    
    # Copy deques
    rd_deque = cp.copy(_rd_deque)
    rf_deque = cp.copy(_rf_deque)
    dt = cp.copy(_dt)
    
    
    # We now compute discretised versions of int (rd - rf)dt, e^(-int rd dt)
    # and e^(-int rf dt), as well as T
    rsumdt = 0
    expmrd = 1
    expmrf = 1
    T = 0

    lastlayer = deque([])
    while dt != lastlayer:    
        DT = dt.popleft()
        RD = rd_deque.popleft()
        RF = rf_deque.popleft()
        R = RD - RF
        rsumdt += R*DT
        expmrd *= exp(-DT*RD)
        expmrf *= exp(-DT*RF)
        T += DT
        
        
        
    sqrtT = sqrt(T)
    sigsqrtT = sig*sqrtT
    
    lograt = log(S0/Strk)
    dpl = (lograt + rsumdt)/sigsqrtT+ 0.5*sigsqrtT
    dm = dpl - sigsqrtT
    
    if option == 'Put':
        H  = Strk*expmrd*norm.cdf(-1.0*dm) - S0*expmrf*norm.cdf(-1.0*dpl)
        
    elif option == 'Call':
        H  = S0*expmrf*norm.cdf(dpl) - Strk*expmrd*norm.cdf(dm)

        
    
    return H









# Example usage:

if __name__ == '__main__':


    S0 = 100
    sig = 0.20
    Strk = S0*1.01
    rd3 = 0.02
    rd2 = 0.01
    rd1 = 0.01
    rf3 = 0.00
    rf2 = 0.00
    rf1 = 0.00
    dt3 = 1/12
    dt2 = 1/12
    dt1 = 1/12
    
    rd_deque = deque([rd3, rd2, rd1])
    rf_deque = deque([rf3, rf2, rf1])
    dt = deque([dt3, dt2, dt1])
    
    option = 'Put'
    
        
    print(BSform_pw(S0, sig, Strk, rd_deque, rf_deque, dt, option))