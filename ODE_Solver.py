# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Article: "Explicit approximations of option prices via Malliavin calculus in a
general stochastic volatility framework"

Description: Computes the explicit solution to the ODE for v_{0,t} in Section 6 
for with piecewise-constant parameters which is:

dv_{0,t} = kap(the - v_{0,t})v_{0,t}*dt 

which pertains to the Verhulst model and is solved through the Methods:

- ODE_Solution_Verhulst(_kap_ar, _the_ar, v0, _dttilde): uses its explicit solution.
- ODE_Solver_Verhulst(_kap_ar, _the_ar, v0, _dttilde): use a numerical Euler scheme.

In addition, we also include methods for the ODE

dv_{0,t} = kap(the - v_{0,t})*dt 

which pertains to the Heston model, IGa model and generally any model with the usual linear mean reversion drift,
and is solved using the Methods:

- ODE_Solution_mr(_kap_ar, _the_ar, v0, _dttilde): uses its explicit solution.
- ODE_Solver_mr(_kap_ar, _the_ar, v0, _dttilde): uses a numerical Euler scheme.

"""


import copy as cp
from collections import deque
from math import exp
from timeit import default_timer as timer 

import matplotlib.pyplot as plt




def ODE_Solution_Verhulst(_kap_ar, _the_ar, v0, _dttilde):

    """
    Computes v_{0, t} in the Verhulst model using its explicit solution.

    kap_ar (np.array): kap_ar = np.array([kap3, kap2, kap1]), piecewise parameters are 
    specified backward in time.
    the_ar (np.array): the_ar = np.array([kap3, kap2, kap1]), piecewise parameters are 
    specified backward in time.
    v0 (float): initial value of v_{0,0}.
    dttilde (deque): A deque of time increments over which parameters are 'alive',  
    e.g., dttilde = deque([dttilde2, dttilde1]), piecewise parameters are specified backward in time.

    """
    
    
    # Copy deques
    kap_ar = cp.copy(_kap_ar)
    the_ar = cp.copy(_the_ar)
    dttilde = cp.copy(_dttilde)


    v = deque([])
    vtemp = v0

    ztemp = v0**(-1)
    
    lastlayer = deque([])
    while kap_ar != lastlayer:
        
        v.appendleft(vtemp)

        kap = kap_ar.pop()
        the = the_ar.pop()    
        DT = dttilde.pop() 
                
        ztemp = exp(-1.0*kap*the*DT)*(ztemp - 1/the) + 1/the
        
        vtemp = ztemp**(-1.0)
    
    
    return(v)




# Example usage:

if __name__ == '__main__':


    kap_ar = deque([3, 5, 4, 5])
    the_ar = deque([0.3, 0.5, 0.4, 0.3])
    dttilde = deque([0.1, 0.1, 0.1, 0.1])
    v0 = 0.4
    
    print(ODE_Solution_Verhulst(kap_ar, the_ar, v0, dttilde))
    







def ODE_Solver_Verhulst(_kap_ar, _the_ar, v0, _dttilde):
    
    """
    Computes v_{0, t} in the Verhulst model using an Euler scheme.

    kap_ar (np.array): kap_ar = np.array([kap3, kap2, kap1]), piecewise parameters are 
    specified backward in time.
    the_ar (np.array): the_ar = np.array([kap3, kap2, kap1]), piecewise parameters are 
    specified backward in time.
    v0 (float): initial value of v_{0,0}.
    dttilde (deque): A deque of time increments over which parameters are 'alive',  
    e.g., dttilde = deque([dttilde2, dttilde1]), piecewise parameters are specified backward in time.

    """
    
    
    # Copy deques
    kap_ar = cp.copy(_kap_ar)
    the_ar = cp.copy(_the_ar)
    dttilde = cp.copy(_dttilde)
    

    v = deque([])
    vtemp = v0
    
    lastlayer = deque([])
    while kap_ar != lastlayer:
                
        v.appendleft(vtemp)

        kap = kap_ar.pop()
        the = the_ar.pop()    
        DTtilde = dttilde.pop() 
        
        kapvtempDTtilde = kap*vtemp*DTtilde
        
        vtemp += kapvtempDTtilde*the - kapvtempDTtilde*vtemp
        
        
    
    return(v)




# Example usage:

if __name__ == '__main__':


    kap_ar = deque([3, 5, 4, 5])
    the_ar = deque([0.3, 0.5, 0.4, 0.3])
    dttilde = deque([0.1, 0.1, 0.1, 0.1])
          
    v0 = 0.4

    
    
    print(ODE_Solver_Verhulst(kap_ar, the_ar, v0, dttilde))









    
    
    

def ODE_Solution_mr(_kap_ar, _the_ar, v0, _dttilde):

    """
    Computes v_{0, t} in the usual mean reversion model using its explicit solution.

    kap_ar (np.array): kap_ar = np.array([kap3, kap2, kap1]), piecewise parameters are 
    specified backward in time.
    the_ar (np.array): the_ar = np.array([kap3, kap2, kap1]), piecewise parameters are 
    specified backward in time.
    v0 (float): initial value of v_{0,0}.
    dttilde (deque): A deque of time increments over which parameters are 'alive',  
    e.g., dttilde = deque([dttilde2, dttilde1]), piecewise parameters are specified backward in time.

    """
    
    
    # Copy deques
    kap_ar = cp.copy(_kap_ar)
    the_ar = cp.copy(_the_ar)
    dttilde = cp.copy(_dttilde)
    
    v = deque([])
    vtemp = v0
    
    
    lastlayer = deque([])
    while kap_ar != lastlayer:
        
        v.appendleft(vtemp)
        
        kap = kap_ar.pop()
        the = the_ar.pop() 
        DT = dttilde.pop() 
        
        vtemp = exp(-1.0*kap*DT)*(vtemp - the) + the



    return(v)




# Example usage:

if __name__ == '__main__':


    kap_ar = deque([3, 5, 4, 5])
    the_ar = deque([0.3, 0.5, 0.4, 0.3])
    dt = deque([0.1, 0.1, 0.1, 0.1])
    v0 = 0.4
    
    
    print(ODE_Solution_mr(kap_ar, the_ar, v0, dt))


    












def ODE_Solver_mr(_kap_ar, _the_ar, v0, _dttilde):

    """
    Computes v_{0, t} in the usual mean reversion model using an Euler scheme.

    kap_ar (np.array): kap_ar = np.array([kap3, kap2, kap1]), piecewise parameters are 
    specified backward in time.
    the_ar (np.array): the_ar = np.array([kap3, kap2, kap1]), piecewise parameters are 
    specified backward in time.
    v0 (float): initial value of v_{0,0}.
    dttilde (deque): A deque of time increments over which parameters are 'alive',  
    e.g., dttilde = deque([dttilde2, dttilde1]), piecewise parameters are specified backward in time.

    """
    
    
    # Copy deques
    kap_ar = cp.copy(_kap_ar)
    the_ar = cp.copy(_the_ar)
    dttilde = cp.copy(_dttilde)
    
    v = deque([])
    vtemp = v0
    #v.appendleft(v0)
    
    
    
    lastlayer = deque([])
    while kap_ar != lastlayer:
        
        v.appendleft(vtemp)
        
        kap = kap_ar.pop()
        the = the_ar.pop() 
        DTtilde = dttilde.pop() 
        
        kapDTtilde = kap*DTtilde
        kaptheDTtilde = kapDTtilde*the

        vtemp += kaptheDTtilde - kapDTtilde*vtemp
    
    

    return(v)




# Example usage:

if __name__ == '__main__':


    kap_ar = deque([3, 5, 4, 5])
    the_ar = deque([0.3, 0.5, 0.4, 0.3])
    dttilde = deque([0.1, 0.1, 0.1, 0.1])
    
    v0 = 0.4
    
    
    print(ODE_Solver_mr(kap_ar, the_ar, v0, dttilde))
    

    
