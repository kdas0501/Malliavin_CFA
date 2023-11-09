# -*- coding: utf-8 -*-
"""
@author: Dr Kaustav Das (kaustav.das@monash.edu)

Article: "Explicit approximations of option prices via Malliavin calculus in a
general stochastic volatility framework"

Description: Returns implied volatility for a given European Put or Call option price using 
Brent's method, where parameters are piecewise-constant.

"""

from scipy import optimize
from BSform_pw import BSform_pw
from collections import deque


def ImpVolBrent_pw(S0, Strk, rd_deque, rf_deque, dt, option, price):

    """
    Returns implied volatility for a given European Put or Call option price using 
    Brent's method, where parameters are piecewise-constant.

    S0 (float): initial spot.
    Strk (float): strike value of the contract.
    rd_deque (deque): domestic interest rate, given backward, e.g., rd_deque = deque([rd2, rd1]).
    rf_deque (deque): foreign interest rate, given backward, e.g., rf_deque = deque([rf2, rf1]).
    dt (deque): deque of time increments over which each parameter is 'alive',
        given backward, e.g., dt = deque([dt2, dt1]). Note sum(dt) gives option maturity T.
    option (str): 'Put' or 'Call'.
    price (float): Price of the Put or Call option.
    """
    
    def BSform_pw_sig(sig):
        
        return BSform_pw(S0, sig, Strk, rd_deque, rf_deque, dt, option) - price

    root = optimize.brentq(BSform_pw_sig, -1, 1)
    
    return root









# Example usage:

if __name__ == '__main__':
    
    
    S0 = 100
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
    price = 4.3416939168077135
    
    rd_deque = deque([rd3, rd2, rd1])
    rf_deque = deque([rf3, rf2, rf1])
    dt = deque([dt3, dt2, dt1])
    
    option = 'Put'
    
    
    print(ImpVolBrent_pw(S0, Strk, rd_deque, rf_deque, dt, option, price))
