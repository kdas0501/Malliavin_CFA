# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Article: "Explicit approximations of option prices via Malliavin calculus in a
general stochastic volatility framework"

Description: Computes the Strike associated with a European put or call option price
for a given Delta (unsigned).

"""


from scipy.stats import norm
from collections import deque
from math import sqrt, exp
import copy as cp

"""
Computes the Strike associated with a European put or call option price
for a given Delta (unsigned).


S0 (float): initial spot.
sig (float): implied vol guess.
rd_deque (deque): domestic interest rate, given backward, e.g., rd_deque = deque([rd2, rd1]).
rf_deque (deque): foreign interest rate, given backward, e.g., rf_deque = deque([rf2, rf1]).
dt (deque): deque of time increments over which each parameter is 'alive',
  given backward, e.g., dt = deque([dt2, dt1]). Note sum(dt) gives option maturity T.
Delta (float): Absolute value of Delta, e.g. Delta = 0.5 is ATM for either put or call option.
option (str): 'Put' or 'Call'.

"""


def DeltaStrikes_pw(S0, sig, _rd_deque, _rf_deque, _dt, Delta, option):
    
    
    # Copy deques
    rd_deque = cp.copy(_rd_deque)
    rf_deque = cp.copy(_rf_deque)
    dt = cp.copy(_dt)
    

    # We now compute discretised versions of int (rd - rf)dt, e^(-int rd dt)
    # and e^(-int rf dt), as well as T.
    rsumdt = 0
    exprf = 1
    T = 0
    
    
    lastlayer = deque([])
    while dt != lastlayer:
        DT = dt.popleft()
        RD = rd_deque.popleft()
        RF = rf_deque.popleft()
        R = RD - RF
        rsumdt += R*DT
        exprf *= exp(-DT*RF)
        T += DT
    
    
    
    if option == 'Put':
        Strk = S0*exp(sig*sqrt(T)*norm.ppf(Delta*exprf) + rsumdt + 0.5*sig**2*T)
        
    elif option == 'Call':
        Strk = S0*exp(-sig*sqrt(T)*norm.ppf(Delta*exprf) + rsumdt + 0.5*sig**2*T)
        
    return Strk
    








# Example usage:

if __name__ == '__main__':

    
    # Common parameters
    
    S0 = 100
    sig = 0.03
    rd3 = 0.05
    rd2 = 0.03
    rd1 = 0.02
    rf3 = 0.02
    rf2 = 0.01
    rf1 = 0.02
    dt3 = 3/12.0
    dt2 = 3/12.0
    dt1 = 3/12.0
    
    rd_deque = deque([rd3, rd2, rd1])
    rf_deque = deque([rf3, rf2, rf1])
    dt = deque([dt3, dt2, dt1])
    
    Delta = 0.5
    option = 'Put'
    
    
    print(DeltaStrikes_pw(S0, sig, rd_deque, rf_deque, dt, Delta, option))

