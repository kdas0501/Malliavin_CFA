# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Article: "Explicit approximations of option prices via Malliavin calculus in a
general stochastic volatility framework"

Description: Computes the PBS functions and their partial derivatives as given in 
Appendix D of the article.
"""


import copy as cp
from math import exp, log, sqrt
from collections import deque
from scipy.stats import norm


def parPBS_pw(x, y, Strk, _rd_deque, _rf_deque, _dt):
    
    """
    Computes the PBS functions and their partial derivatives as given in 
    Appendix D of the article.

    x (float): First argument of PBS.
    y (float) Second argument of PBS.
    Strk (float): Strike of the contract.
    rd_deque (deque): domestic interest rate, given backward, e.g., rd_deque = deque([rd2, rd1]).
    rf_deque (deque): foreign interest rate, given backward, e.g., rf_deque = deque([rf2, rf1]).
    dt (deque): deque of time increments over which each parameter is 'alive',
        given backward, e.g., dt = deque([dt2, dt1]). Note sum(dt) gives option maturity T.

    """
    
    # Copy deques
    rd_deque = cp.copy(_rd_deque)
    rf_deque = cp.copy(_rf_deque)
    dt = cp.copy(_dt)
    
    sqrty = sqrt(y)
    
    # We now compute discretised versions of int (rd - rf)dt, e^(-int rd dt)
    # and e^(-int rf dt), as well as T.
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
    
    
    
    
    dpl = (x - log(Strk) + rsumdt)/sqrty + 0.5*sqrty
    dm = dpl - sqrty
    expx = exp(x)
    expx_expmrf = expx*expmrf
    expx_exprf_phidpl = expx_expmrf*norm.pdf(dpl)
    
    
    
    PBS  = Strk*expmrd*norm.cdf(-1*dm) - expx_expmrf*norm.cdf(-1*dpl)
    yPBS = expx_exprf_phidpl/(2*sqrty)
    yyPBS = expx_exprf_phidpl*(dpl*dm-1)/(4*y**(3/2))
    xyPBS = (-1)*expx_exprf_phidpl*dm/(2*y)
    xxyPBS = (-1)*expx_exprf_phidpl/(2*y)*(dm+1/(sqrty)*(1-dm*dpl))
    xxyyPBS = (-1)*expx_exprf_phidpl /2*(-1/(2*y**2)*(dpl+2*dm)-3/2*y**(-5/2)*(1-dm*dpl)\
            + y**(-3/2)*(y**(-1)*dm*dpl+(1/2))+1/(2*y)*dm*dpl*(dm*y**(-1)+y**(-3/2)*(1-dm*dpl)))
    
        
        
    return(PBS, yPBS, yyPBS, xyPBS, xxyPBS, xxyyPBS)
    


# Example usage:

if __name__ == '__main__':

    
    x = 3
    y = 10**(-1)
    Strk = 100
    
    rd2 = 0.02
    rd1 = 0.01
    rf2 = 0.00
    rf1 = 0.00
    dt2 = 1/12.0
    dt1 = 1/12.0
    
    rd_deque = deque([rd2, rd1])
    rf_deque = deque([rf2, rf1])
    dt = deque([dt2, dt1])
    

    print(parPBS_pw(x, y, Strk, rd_deque, rf_deque, dt))