# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Article: "Explicit approximations of option prices via Malliavin calculus in a
general stochastic volatility framework"

Description: Computes the price of a European put option in the XGBM (Verhulst/Logistic) model
with piecewise-constant parameters

dS = (rd - rf)S*dt + V*S*dW,
dV = kap(the - V)V*dt + lam*V*dB,
dW dB = rho dt

via the Malliavin closed-form approximation formula given by Corollary 6.1 in the article.

"""


from collections import deque
from ODE_Solver import ODE_Solver_Verhulst, ODE_Solution_Verhulst
from OmgG import OmgG1, OmgG2, OmgG3, OmgG4
from parPBS_pw import parPBS_pw
import numpy as np
import copy as cp
from timeit import default_timer as timer 



def Verhulst_CFA_Malliavin(model_params, global_params):

    """

    Computes the price of a European put option in the XGBM (Verhulst/Logistic) model
    with piecewise-constant parameters via the Malliavin closed-form approximation formula given by Corollary 6.1 in the article.

    model_params (list): [kap, the, lam, rho] is a list of np.arrays, 
        where e.g. kap_ar = np.array([kap2, kap1]), piecewise parameters are 
        specified backward in time.
    globals_params (list): [S0, V0, rd_deque, rf_deque, Strk, dt, N_dttilde].
        S0 (float): initial spot.
        V0 (float): initial variance/volatility.
        rd_deque (deque): domestic interest rate, given backwards , e.g., rd_deque = deque([rd2, rd1]).
        rf_deque (deque): foreign interest rate, given backwards, e.g., rf_deque = deque([rf2, rf1]).
        Strk (float): Strike of the contract.
        dt (deque): deque of time increments over which each parameter is 'alive',
            given backwards, e.g., dt = deque([dt2, dt1]). Note sum(dt) gives option maturity T.
        N_dttilde (int): Number of points for solving the ODE for v_{0,t}.

    """
    
    
    # kap, the, lam and rho are all arrays, but we have not put the subscript _ar
    # for simplicity
    kap = model_params[0]
    the = model_params[1]
    lam = model_params[2]
    rho = model_params[3]

    S0 = global_params[0]
    V0 = global_params[1]
    _rd_deque = global_params[2]
    _rf_deque = global_params[3]
    Strk = global_params[4]
    _dt = global_params[5]
    N_dttilde = global_params[6]
    
    
    # Start timer
    start = timer()
    
    x0 = np.log(S0)    
    
    # Copy deques
    rd_deque = cp.copy(_rd_deque)
    rf_deque = cp.copy(_rf_deque)
    # olddt = cp.copy(_dt)
    dt = cp.copy(_dt)
    
    T = sum(dt)


    # We need to solve ODE v_{0, t} over a fine grid (with N_dttilde points) that contains the maturities
    # (provided in dt). This requires a non-uniform mesh 
    dttilde = deque([])
    for DT in dt:
        Ni = int(np.ceil(DT/T*N_dttilde))
        DT_tildei = DT/Ni
        for j in range(Ni):
            dttilde.append(DT_tildei)
    
    olddttilde = cp.copy(dttilde)
    
    

    
    
    
    
    # Precompute the parameters to input into OmgG functions
    mkap = -1*kap
    mthe = -1*the
    kapthe = kap*the
    t2kapthe = 2*kapthe
    mkapthe = -1*kapthe
    mt2kapthe = -2*kapthe
    t2kap = 2*kap
    mt2kap = -1*t2kap
    t2the = 2*the
    mt2the = -1*t2the
    f4kap = 2*t2kap
    mf4kap = -1*f4kap
    mt2kap = -2*kap    
    lamsq = (lam**2)
    rholam = rho*lam
    
    # Initilise deques for SDE parameters
    d_kap = deque([])
    d_the = deque([])
    d_kapthe = deque([])
    d_t2kapthe = deque([])
    d_mkapthe = deque([])
    d_mt2kapthe = deque([])
    d_mkap = deque([])
    d_mthe = deque([])
    d_t2kap = deque([])
    d_mt2kap = deque([])
    d_t2the = deque([])
    d_mt2the = deque([])
    d_f4kap = deque([])
    d_mf4kap = deque([])
    d_mt2kap = deque([])
    d_lamsq = deque([])
    d_rholam = deque([])
    d_ones = deque([])
    d_zeros = deque([])
    
    
    # These will be the 'new' versions of rd and rf but over the time partition
    # governed by dttilde
    nrd_deque = deque([])
    nrf_deque = deque([])
    
    # Convert all SDE parameter arrays to deques and resize everything 
    # to associate all parameters with the time partition governed by 
    # dttilde rather than dt.
    for (i, DT) in enumerate(dt):

        rd_temp = rd_deque.popleft()
        rf_temp = rf_deque.popleft()
        
        kap_temp = kap[i]
        the_temp = the[i]
        kapthe_temp = kapthe[i]
        mkapthe_temp = mkapthe[i]
        mt2kapthe_temp = mt2kapthe[i]
        t2kapthe_temp = t2kapthe[i]
        mkap_temp = mkap[i] 
        mthe_temp = mthe[i]
        t2kap_temp = t2kap[i]
        mt2kap_temp = mt2kap[i]
        t2the_temp = t2the[i]
        mt2the_temp = mt2the[i]
        f4kap_temp = f4kap[i]
        mf4kap_temp = mf4kap[i]
        lamsq_temp = lamsq[i]
        rholam_temp = rholam[i]
        
        
        sumDTtilde = 0 
        while sumDTtilde < DT and olddttilde != deque([]):
            
            nrd_deque.append(rd_temp)
            nrf_deque.append(rf_temp)
            
            d_kap.append(kap_temp)
            d_the.append(the_temp)
            d_kapthe.append(kapthe_temp)
            d_mkapthe.append(mkapthe_temp)
            d_mt2kapthe.append(mt2kapthe_temp)
            d_t2kapthe.append(t2kapthe_temp)
            d_mkap.append(mkap_temp)
            d_mthe.append(mthe_temp)
            d_t2kap.append(t2kap_temp)
            d_mt2kap.append(mt2kap_temp)
            d_t2the.append(t2the_temp)
            d_mt2the.append(mt2the_temp)
            d_f4kap.append(f4kap_temp)
            d_mf4kap.append(mf4kap_temp)
            d_mt2kap.append(mt2kap_temp)
            d_lamsq.append(lamsq_temp)
            d_rholam.append(rholam_temp)
            d_ones.append(1.0)
            d_zeros.append(0.0)
            
            DTtilde = olddttilde.popleft()
            sumDTtilde += DTtilde
                                
            
    # Solve ODE for v_{0,t}
    oded_kap = cp.copy(d_kap)
    oded_the = cp.copy(d_the)
    ode_dttilde = cp.copy(dttilde)
    
    # v = ODE_Solver_Verhulst(oded_kap, oded_the, V0, ode_dttilde)
    v = ODE_Solution_Verhulst(oded_kap, oded_the, V0, ode_dttilde)

    
    
    
    # Compute \int_0^T v_{0,t}^2 dt
    intvsqrdttilde = cp.copy(dttilde)
    intvsqr = 0
    intv = cp.copy(v)
    
    while intvsqrdttilde != deque([]):
        DTtilde = intvsqrdttilde.pop()
        Vtemp = intv.pop()
        Vtempsqr = Vtemp**2
        intvsqr += Vtempsqr*DTtilde
 
    
    # Compute partial derivatives of PBS
    (PBS, yPBS, yyPBS, xyPBS, xxyPBS, xxyyPBS) = parPBS_pw(x0, intvsqr, Strk, nrd_deque, nrf_deque, dttilde)  
        
    # Compute coeffcients of the PBS partial derivatives
    yterm = OmgG3(d_mt2kapthe, d_f4kap, d_lamsq, d_kapthe, d_mt2kap, d_mt2kap, d_kapthe, d_mt2kap, d_ones, v, deque([2 ,0, 1]), dttilde)+\
    OmgG2(d_mt2kapthe, d_f4kap, d_lamsq, d_t2kapthe, d_mf4kap, d_ones, v,  deque([2, 0]), dttilde)
    
    xyterm = 2*OmgG2(d_mkapthe, d_t2kap, d_rholam, d_kapthe, d_mt2kap, d_ones, v, deque([2, 1]), dttilde)
        
    xxyterm =2*OmgG4(d_mkapthe, d_t2kap, d_rholam, d_mkapthe, d_t2kap, d_rholam, d_kapthe, d_mt2kap, d_mt2kap, d_kapthe, d_mt2kap, d_ones, v, deque([2, 2, 0, 1]), dttilde)+\
    4*OmgG3(d_mkapthe, d_t2kap, d_rholam, d_zeros, d_zeros, d_rholam, d_kapthe, d_mt2kap, d_ones, v, deque([2, 1, 1]), dttilde) 
    
    yyterm = 4*OmgG3(d_mt2kapthe, d_f4kap, d_lamsq, d_kapthe, d_mt2kap, d_ones, d_kapthe, d_mt2kap, d_ones, v, deque([2,1,1]), dttilde)
    
    xxyyterm = 2*(OmgG2(d_mkapthe, d_t2kap, d_rholam, d_kapthe, d_mt2kap, d_ones, v, deque([2, 1]), dttilde))**2
   
    
   


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
    
    
    # N_dttilde is the number of points in the grid for solving the ODE. 
    # Don't make it too fine or the Omg functions will have a bad time (try N_dttilde = 40 for example). 
    # Don't make it too coarse or the approximations to integrals involving 
    # v_{0,t} will be bad.

    N_dttilde = 35

    global_params = [S0, V0, rd_deque, rf_deque, Strk, dt, N_dttilde]
    model_params = [kap_ar, the_ar, lam_ar, rho_ar]
    
    print(Verhulst_CFA_Malliavin(model_params, global_params))










