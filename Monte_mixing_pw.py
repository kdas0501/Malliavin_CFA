# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Article: "Explicit approximations of option prices via Malliavin calculus in a
general stochastic volatility framework"

Description: Computes the Monte-Carlo price of a European put or call option where parameters 
are piecewise-constant using the log mixing solution method in a general model:

dS = (rd - rf)Sdt + S(V or sqrt{V})dW
dV = alpha(t,V)dt + beta(t,V) dB
dW dB = rho dt

"""


import numpy as np
from scipy.stats import norm
from collections import deque
import copy as cp
from timeit import default_timer as timer 


def Monte_mixing_pw(model_params, global_params, N_PATH, N_TIME, model, option):


    """
    Computes the Monte-Carlo price of a European put or call option where parameters 
    are piecewise-constant using the log mixing solution method in a general model:

    dS = (rd - rf)Sdt + S(V or sqrt{V})dW
    dV = alpha(t,V)dt + beta(t,V) dB
    dW dB = rho dt

    model_params (list): [kap_ar, the_ar, lam_ar, rho_ar] is a list of np.arrays, 
    where e.g. kap_ar = np.array([kap2, kap1]), piecewise parameters are 
    specified backward in time.
    globals_params (list): [S0, V0, rd_deque, rf_deque, Strk, dt].
        S0 (float): initial spot.
        V0 (float): initial variance/volatility.
        rd (deque): domestic interest rate, given backwards , e.g., _rd = deque([rd2, rd1]).
        rf (deque): foreign interest rate, given backwards, e.g., _rf = deque([rf2, rf1]).
        Strk (float): Strike of the contract.
        dt (deque): deque of time increments over which each parameter is 'alive' over,
          given backwards, e.g., _dt = deque([dt2, dt1]). Note sum(dt) gives option maturity T.
    N_PATH (int): # of Monte-Carlo paths.
    N_TIME (int): # of time discretisation points for SDE integration.
    model (str): specifies model, where
      t gives "test", volatility & dV = dB,
      o gives "Ornstein", volatility & dV = kap(the - V)dt + lam dB,
      g gives "GARCH", variance & dV = kap(the - V)dt + lam V dB,
      i gives "Inverse-Gamma", volatility & dV = kap(the - V)dt + lam V dB,
      h gives "Heston", variance & dV = kap(the - V)dt + lam sqrt{V} dB,
      v gives "Verhulst", volatility & dV = kap*V*(the - V)dt + lam V dB.
    option (str): 'Put' or 'Call'.

    """
    
    # Extra parameters
    kap_ar = model_params[0]
    the_ar = model_params[1]
    lam_ar = model_params[2]
    rho_ar = model_params[3]

    S0 = global_params[0]
    V0 = global_params[1]
    _rd_deque = global_params[2]
    _rf_deque = global_params[3]
    Strk = global_params[4]
    _dt = global_params[5]
    
    # Copy deques
    rd_deque = cp.copy(_rd_deque)
    rf_deque = cp.copy(_rf_deque)
    dt = cp.copy(_dt)
    
    
    
    
    # Start timing
    start = timer()
    
    
    

    # We now compute discretised versions of int (rd - rf)dt, e^(-int rd dt)
    # and e^(-int rf dt), as well as T.    
    rsumdt = 0
    expmrd = 1
    expmrf = 1
    T = 0
    dt_copy = cp.copy(_dt)
    
    lastlayer = deque([])
    while dt_copy != lastlayer:
        DT = dt_copy.popleft()
        RD = rd_deque.popleft()
        RF = rf_deque.popleft()
        R = RD - RF
        rsumdt += R*DT
        expmrd *= np.exp(-DT*RD)
        expmrf *= np.exp(-DT*RF)
        T += DT
    
    # Time parameters
    DT = T/N_TIME
    sqrtDT = np.sqrt(DT)
    
    
    # Initialise deques for the SDE parameters
    d_kap = deque([])
    d_kapthe = deque([])
    d_lam = deque([])
    d_rho = deque([])
    d_rhosq = deque([])
    d_omrhosq = deque([])
    
    
    # Do some manipulations
    kapthe_ar = kap_ar*the_ar
    rhosq_ar = rho_ar**2
    omrhosq_ar = 1.0 - rhosq_ar
    
    # Convert SDE parameter np.arrays to deques
    for i in range(0, len(kap_ar)):
        d_kap.append(kap_ar[i])
        d_kapthe.append(kapthe_ar[i])
        d_lam.append(lam_ar[i])
        d_rho.append(rho_ar[i])
        d_rhosq.append(rhosq_ar[i])
        d_omrhosq.append(omrhosq_ar[i])
    
    # N_TIME_running is the point in the time grid for which the current
    # SDE parameter value is 'alive' till. Its value gets updated in the for loop
    # below
    dt_running = dt.pop()
    N_TIME_running = int(dt_running/DT)
    
    kap = d_kap.pop()
    kapDT = kap*DT
    kapthe = d_kapthe.pop()
    kaptheDT = kapthe*DT
    lam = d_lam.pop()
    rho = d_rho.pop()
    rhosq = d_rhosq.pop()
    omrhosq = d_omrhosq.pop()
    
    
    
    
    
    
    
    
    
    
    V = np.ones([N_PATH])*V0
    lnS = np.ones([N_PATH])*np.log(S0)
    
    It = np.zeros([N_PATH])
    Rex = np.zeros([N_PATH])
    Rey = np.zeros([N_PATH])

# Test: Volatility and dV = dB         
    if model == "t":
        for tt in range(N_TIME):
            
            if tt > N_TIME_running: 
                dt_running += dt.pop()
                N_TIME_running = int(dt_running/DT)
            
            RND = np.random.normal(0,1,N_PATH)
            Vsqr = V**2
            Rex += rhosq*Vsqr*DT
            Rey += omrhosq*Vsqr*DT
            It += rho*V*sqrtDT*RND
            V += sqrtDT*RND
    
    # Ornstein Uhlenbeck: Volatility & dV = kap(the - V)dt + lam*dB_t     
    elif model == "o":
        for tt in range(N_TIME):
            
            if tt > N_TIME_running:
                kap = d_kap.pop()
                kapDT = kap*DT
                kapthe = d_kapthe.pop()
                kaptheDT = kapthe*DT
                lam = d_lam.pop()
                rho = d_rho.pop()
                rhosq = d_rhosq.pop()
                omrhosq = d_omrhosq.pop()
                       
                dt_running += dt.pop()
                N_TIME_running = int(dt_running/DT)
                                
            RND = np.random.normal(0,1,N_PATH)
            Vsqr = V**2
            Rex += rhosq*Vsqr*DT
            Rey += omrhosq*Vsqr*DT
            It += rho*V*sqrtDT*RND
            V += kaptheDT-kapDT*V + lam*sqrtDT*RND
            
    # GARCH: Variance & dV = kap(the - V)dt + lam*V*dB_t
    elif model == "g":
        for tt in range(N_TIME):
            
            if tt > N_TIME_running:
                kap = d_kap.pop()
                kapDT = kap*DT
                kapthe = d_kapthe.pop()
                kaptheDT = kapthe*DT
                lam = d_lam.pop()
                rho = d_rho.pop()
                rhosq = d_rhosq.pop()
                omrhosq = d_omrhosq.pop()
               
                dt_running += dt.pop()
                N_TIME_running = int(dt_running/DT)              
                
            RND = np.random.normal(0,1,N_PATH)
            mV = np.maximum(V,0)
            sqrtmV = np.sqrt(mV)  
            Rex += rhosq*V*DT
            Rey += omrhosq*V*DT
            It += rho*sqrtmV*sqrtDT*RND
            V += kaptheDT-mV*kapDT + lam*mV*sqrtDT*RND

    # Inverse Gamma: Volatility & dV = kap(the - V)dt + lam*V*dB_t
    elif model == "i":
        for tt in range(N_TIME):
            
            if tt > N_TIME_running:
                kap = d_kap.pop()
                kapDT = kap*DT
                kapthe = d_kapthe.pop()
                kaptheDT = kapthe*DT
                lam = d_lam.pop()
                rho = d_rho.pop()
                rhosq = d_rhosq.pop()
                omrhosq = d_omrhosq.pop()
                 
                dt_running += dt.pop()
                N_TIME_running = int(dt_running/DT)
                                
            RND = np.random.normal(0,1,N_PATH)
            Vsqr = V**2
            Rex += rhosq*Vsqr*DT
            Rey += omrhosq*Vsqr*DT
            It += rho*V*sqrtDT*RND
            V += kaptheDT-V*kapDT + lam*V*sqrtDT*RND
            
    # Heston: Variance & dV = kap(the - V)dt + lam*sqrt{V}*dB_t 
    elif model == "h":       
        if 2*kapthe < lam**2:
            print("\nFeller test fail!")
            
        for tt in range(N_TIME):
            
            if tt > N_TIME_running:
                kap = d_kap.pop()
                kapDT = kap*DT
                kapthe = d_kapthe.pop()
                kaptheDT = kapthe*DT
                lam = d_lam.pop()
                rho = d_rho.pop()
                rhosq = d_rhosq.pop()
                omrhosq = d_omrhosq.pop()
                if 2*kapthe < lam**2:
                    print("\nFeller test fail!")
                
                dt_running += dt.pop()
                N_TIME_running = int(dt_running/DT)
                
            RND = np.random.normal(0,1,N_PATH)
            mV = np.maximum(V,0)
            sqrtmV = np.sqrt(mV)  
            Rex += rhosq*V*DT
            Rey += omrhosq*V*DT
            It += rho*sqrtmV*sqrtDT*RND
            V += kaptheDT-mV*kapDT + lam*sqrtmV*sqrtDT*RND
            
    # Verhulst: Variance & dV = kap*V*(the - V)dt + lam*V*dB_t 
    elif model == "v":       
        for tt in range(N_TIME):
            
            if tt > N_TIME_running:
                kap = d_kap.pop()
                kapDT = kap*DT
                kapthe = d_kapthe.pop()
                kaptheDT = kapthe*DT
                lam = d_lam.pop()
                rho = d_rho.pop()
                rhosq = d_rhosq.pop()
                omrhosq = d_omrhosq.pop()
                
                dt_running += dt.pop()
                N_TIME_running = int(dt_running/DT)
                                
            RND = np.random.normal(0,1,N_PATH)
            Vsqr = V**2
            Rex += rhosq*Vsqr*DT
            Rey += omrhosq*Vsqr*DT

            It += rho*V*sqrtDT*RND
            V += kaptheDT*V-kapDT*Vsqr+ lam*V*sqrtDT*RND
        

        




            
    # Compute the price using the Mixing solution

    xarg = lnS + It - 0.5*Rex
    yarg = Rey
    sqrtyarg = np.sqrt(np.abs(yarg))
    dpl = (xarg - np.log(Strk) + rsumdt)/sqrtyarg + 0.5*sqrtyarg
    dm = dpl - sqrtyarg

    
    if option == "Put":
        
        PBS = Strk*expmrd*norm.cdf(-1.0*dm) - np.exp(xarg)*expmrf*norm.cdf(-1.0*dpl)
        H = np.mean(PBS)
        end = timer()
        
        Sigma = np.std(PBS)
        
    elif option == "Call":
        
        CBS = np.exp(xarg)*expmrf*norm.cdf(dpl) - Strk*np.expmrd*norm.cdf(dm)
        H = np.mean(CBS)
        end = timer()
        
        Sigma = np.std(CBS)
        
    # Standard Error
    SE = Sigma/np.sqrt(N_PATH)

    elapsed = end - start
    
    return(H, SE, elapsed)









# Example usage:

if __name__ == '__main__':

    from DeltaStrikes_pw import DeltaStrikes_pw

    # Global parameters
    S0 = 100
    V0 = 0.18       # V0 is initial variance for Heston and GARCH
    rd3 = 0.02
    rd2 = 0.03
    rd1 = 0.01
    rf3 = 0.00
    rf2 = 0.00
    rf1 = 0.00
    dt3 = 6/12
    dt2 = 3/12
    dt1 = 3/12
    
    rd_deque = deque([rd3, rd2, rd1])
    rf_deque = deque([rf3, rf2, rd1])
    dt = deque([dt3, dt2, dt1])

    rd_deque = deque([rd3])
    rf_deque = deque([rf3])
    dt = deque([dt3])
    
    Delta = 0.5
    sig = np.sqrt(V0)
    Strk = DeltaStrikes_pw(S0, sig, rd_deque, rf_deque, dt, Delta, 'Put')
    # Strk = S0*1.01
    
    # Model parameters
    kap3 = 5.0
    kap2 = 5.1
    kap1 = 4.9
    the3 = 0.15
    the2 = 0.16
    the1 = 0.14
    lam3 = 0.4142
    lam2 = 0.4342
    lam1 = 0.3942
    rho3 = -0.391
    rho2 = -0.401
    rho1 = -0.381
    
    kap_ar = np.array([kap3, kap2, kap1])
    the_ar = np.array([the3, the2, the1])
    lam_ar = np.array([lam3, lam2, lam1])
    rho_ar = np.array([rho3, rho2, rho1])

    kap_ar = np.array([kap3])
    the_ar = np.array([the3])
    lam_ar = np.array([lam3])
    rho_ar = np.array([rho3])
    
    
    global_params = [S0, V0, rd_deque, rf_deque, Strk, dt]
    model_params = [kap_ar, the_ar, lam_ar, rho_ar]
    
    # Monte-Carlo parameters
    N_PATH = 500000
    Steps_day_N = 24
    N_TIME = int(Steps_day_N*253*sum(dt)) 
    
    model = 'v'
    option = 'Put'
    
    
    print(Monte_mixing_pw(model_params, global_params, N_PATH, N_TIME, model, option))