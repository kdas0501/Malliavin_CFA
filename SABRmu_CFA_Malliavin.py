# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Article: "Explicit approximations of option prices via Malliavin calculus in a
general stochastic volatility framework"

Description: Computes the price of a European put option in the SABR-mu model
with piecewise-constant parameters:

dS = (rd - rf) S dt + V S dW,
dV = lam V^mu dB,
dW dB = rho dt

via the Malliavin closed-form approximation formula given by Corollary F.1 in the article.

"""


from collections import deque
from OmgG import OmgG1, OmgG2, OmgG3
from parPBS_pw import parPBS_pw
import numpy as np
import copy as cp
from timeit import default_timer as timer 



def SABRmu_CFA_Malliavin(model_params, mu, global_params):

    """
    Computes the price of a European put option in the SABR-mu model
    with piecewise-constant parameters via the Malliavin closed-form approximation formula given by Corollary F.1 in the article.

    model_params (list): [lam, rho] is a list of np.arrays, 
        where e.g. lam_ar = np.array([lam2, lam1]), piecewise parameters are 
        specified backward in time.
    mu (float): Power of V in the SDE for dV. Should be chosen in the interval [1/2, 1]. mu = 1 gives the classical SABR model.
    globals_params (list): [S0, V0, rd_deque, rf_deque, Strk, dt].
        S0 (float): initial spot.
        V0 (float): initial volatility.
        rd_deque (deque): domestic interest rate, given backwards , e.g., rd_deque = deque([rd2, rd1]).
        rf_deque (deque): foreign interest rate, given backwards, e.g., rf_deque = deque([rf2, rf1]).
        Strk (float): Strike of the contract.
        dt (deque): deque of time increments over which each parameter is 'alive',
            given backwards, e.g., dt = deque([dt2, dt1]). Note sum(dt) gives option maturity T.

    """
    
    
    # lam and rho are arrays, but we have not put the subscript _ar
    # for simplicity
    lam = model_params[0]
    rho = model_params[1]

    S0 = global_params[0]
    V0 = global_params[1]
    _rd_deque = global_params[2]
    _rf_deque = global_params[3]
    Strk = global_params[4]
    _dt = global_params[5]
    
    # Start timer
    start = timer()
    
    x0 = np.log(S0)    
    
    # Copy deques
    rd_deque = cp.copy(_rd_deque)
    rf_deque = cp.copy(_rf_deque)
    dt = cp.copy(_dt)
    
    T = sum(dt)

    
    

    
    
    
    
    # Precompute the parameters to input into OmgG functions
    lamsq = lam**2
    rholam = rho*lam
    
    # Initilise deques for SDE parameters
    d_lamsq = deque([])
    d_rholam = deque([])
    d_ones = deque([])
    d_zeros = deque([])

    # Deque for v
    v = deque([V0]*len(dt))
    
    # Convert all SDE parameter arrays to deques
    for (i, DT) in enumerate(dt):

        lamsq_temp = lamsq[i]
        rholam_temp = rholam[i]
            
        d_lamsq.append(lamsq_temp)
        d_rholam.append(rholam_temp)
        d_ones.append(1.0)
        d_zeros.append(0.0)
    

     # Compute partial derivatives of PBS
    (PBS, yPBS, yyPBS, xyPBS, xxyPBS, xxyyPBS) = parPBS_pw(x0, V0**2*T, Strk, rd_deque, rf_deque, dt)  
        
    # Compute coeffcients of the PBS partial derivatives
    yterm = OmgG2(d_zeros, d_zeros, d_lamsq, d_zeros, d_zeros, d_ones, v,  deque([2*mu, 0]), dt)
    
    xyterm = 2*OmgG2(d_zeros, d_zeros, d_rholam, d_zeros, d_zeros, d_ones, v, deque([mu+1, 1]), dt)
        
    xxyterm = 2*OmgG3(d_zeros, d_zeros, d_rholam, d_zeros, d_zeros, d_rholam, d_zeros, d_zeros, d_ones, v, deque([mu+1, mu+1, 0]), dt)+\
    2*mu*OmgG3(d_zeros, d_zeros, d_rholam, d_zeros, d_zeros, d_rholam, d_zeros, d_zeros, d_ones, v, deque([mu+1, 2*mu-1, 1]), dt)+\
    2*OmgG3(d_zeros, d_zeros, d_rholam, d_zeros, d_zeros, d_rholam, d_zeros, d_zeros, d_ones, v, deque([mu+1, mu, 1]), dt)
    
    yyterm = 4*OmgG3(d_zeros, d_zeros, d_lamsq, d_zeros, d_zeros, d_ones, d_zeros, d_zeros, d_ones, v, deque([2*mu, 1, 1]), dt)
    
    xxyyterm = 2*(OmgG2(d_zeros, d_zeros, d_rholam, d_zeros, d_zeros, d_ones, v, deque([mu+1, 1]), dt))**2
   


    res = PBS + xyterm*xyPBS + xxyterm*xxyPBS + yyterm*yyPBS \
        + xxyyterm*xxyyPBS + yterm*yPBS

    end = timer()
    
    elapsed = end - start
    
    
    
    return (res, elapsed)










# Example usage:

if __name__ == '__main__':
        
    from DeltaStrikes_pw import DeltaStrikes_pw
    
    # Global parameters
    S0 = 100
    V0 = 0.18       # V0 is initial volatility
    rd3 = 0.02
    rd2 = 0.03
    rd1 = 0.01
    rf3 = 0.00
    rf2 = 0.00
    rf1 = 0.00
    T = 1/12        # maturity
    dt3 = (1/2)*T
    dt2 = (1/4)*T
    dt1 = (1/4)*T
    
    rd_deque = deque([rd3, rd2, rd1])
    rf_deque = deque([rf3, rf2, rf1])
    dt = deque([dt3, dt2, dt1])
    
    # rd_deque = deque([rd3])
    # rf_deque = deque([rf3])
    # dt = deque([dt3])
    
    Delta = 0.5
    Strk = DeltaStrikes_pw(S0, V0, rd_deque, rf_deque, dt, Delta, 'Put')
    # Strk = S0*1.01
    
    # Model parameters
    lam3 = 0.4142
    lam2 = 0.4342
    lam1 = 0.3942
    rho3 = -0.391
    rho2 = -0.401
    rho1 = -0.381
    
    lam_ar = np.array([lam3, lam2, lam1])
    rho_ar = np.array([rho3, rho2, rho1])
    
    # lam_ar = np.array([lam3])
    # rho_ar = np.array([rho3])
    
    # mu specifies the power of V in dV. Take mu = 1 for the classical SABR model
    mu = 1

    global_params = [S0, V0, rd_deque, rf_deque, Strk, dt]
    model_params = [lam_ar, rho_ar]
    
    print(SABRmu_CFA_Malliavin(model_params, mu, global_params))










