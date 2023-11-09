# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Article: "Explicit approximations of option prices via Malliavin calculus in a
general stochastic volatility framework"

Description: Computes the Phi functions from Section 6 of the article 
over \tilde T_i to \tilde T_{i+1}.

"""


import copy as cp
from math import exp
from collections import deque



def PhiGo(_ints_p, _params_k, _params_h, Vtemp, DTtilde, eps):

    """
    Computes the Phi functions from Section 6 of the article 
    over \tilde T_i to \tilde T_{i+1}.


    intsp (deque): [  pn,  p(n-1), ...,  p2, p1 ].
    paramsk (deque): [Kn, K(n-1), ..., K1].
    paramsh (deque): [Hn, H(n-1), ..., H1].
    Vtemp (float): v_{0, t} over \tilde T_i to \tilde T_{i+1}.
    DTtilde (float): the increment \tilde T_{i+1} - \tilde T_i.
    eps (float): constant numeric precision for Taylor expansions.
    """
    
     
    # Copy all deques    
    ints_p = cp.copy(_ints_p)
    params_k = cp.copy(_params_k)
    params_h = cp.copy(_params_h)
    
    # Create original copies so they won't get popped off
    origints_p = cp.copy(ints_p)
    origparams_k = cp.copy(params_k)
    origparams_h = cp.copy(params_h)
  
    p = ints_p.popleft()
    K = params_k.popleft()
    H = params_h.popleft()
    lastlayer = ints_p == deque([])
            
    Ktilde = K + H*Vtemp    
    
    
    
    # If parameter Ktilde approximately 0
    if abs(Ktilde) < 10**(-8):
        if lastlayer:
            res = DTtilde/(p+1.0)
        else:
            ints_p[0]+=p+1.0
            res = PhiGo(ints_p, params_k, params_h, Vtemp, DTtilde, eps)*DTtilde/(p+1.0)
    
    
    
    
    
    # If K + Hv_temp =/= 0   
     
    # If first int pn = 0, i.e., Phi(k,h,0)
    elif p == 0:
        # If Ktilde is small, then Taylor
        if abs(Ktilde) < 10**(-2):
            # print('taylor')
            if lastlayer:
                    res = PhiGo(deque([0]), deque([0]), deque([0]), Vtemp, DTtilde, eps)
                    aux = 1.0
                    m = 1
                    while aux > eps:
                        aux *= Ktilde*DTtilde/m
                        res += aux*PhiGo(deque([m]), deque([0]), deque([0]), Vtemp, DTtilde, eps)
                        m += 1
            else:
                    origparams_k[0] = 0.0
                    origparams_h[0] = 0.0
                    res = PhiGo(origints_p, origparams_k, origparams_h, Vtemp, DTtilde, eps)
                    aux = 1.0
                    m = 1
                    while aux > eps:
                        aux *= Ktilde*DTtilde/m
                        origints_p[0] += 1
                        res += aux*PhiGo(origints_p, origparams_k, origparams_h, Vtemp, DTtilde, eps)
                        m += 1
                        
                        
        # If Ktilde*DTtilde is not small!
        else:
            if lastlayer:
                    res = 1/Ktilde*(exp(Ktilde*DTtilde)-1) 
            else:
                newparams_k = cp.copy(params_k)
                newparams_h = cp.copy(params_h)
                newparams_k[0] += K
                newparams_h[0] += H
                res = ( PhiGo(ints_p, newparams_k, newparams_h, Vtemp, DTtilde, eps)-\
                PhiGo(ints_p, params_k, params_h, Vtemp, DTtilde, eps) )/Ktilde
      
            
                    
                
                    
    # If p>=0 and Ktilde > 0, i.e., Phi(k,h,p) 
    
    # If Ktilde small, Taylor
    elif abs((Ktilde)**(p+1.0)) < 10**(-2):
            if lastlayer:
                res = PhiGo(deque([p]), deque([0]), deque([0]), Vtemp, DTtilde, eps)
                aux = 1.0
                m = 1
                while aux > eps:
                    aux *= Ktilde*DTtilde/m
                    res += aux*PhiGo(deque([p+m]), deque([0]), deque([0]), Vtemp, DTtilde, eps)
                    m += 1
            else:
                origparams_k[0] = 0.0
                origparams_h[0] = 0.0
                res = PhiGo(origints_p, origparams_k, origparams_h, Vtemp, DTtilde, eps)
                aux = 1.0
                m = 1
                while aux > eps:
                    aux *= Ktilde*DTtilde/m
                    origints_p[0] += 1
                    res += aux*PhiGo(origints_p, origparams_k, origparams_k, Vtemp, DTtilde, eps)
                    m += 1
         
                    
          
                         
            
    # If Ktilde not small
    else:
        if lastlayer:
            res = (exp(Ktilde*DTtilde) - p/DTtilde*PhiGo(deque([p-1]), deque([K]), deque([H]), Vtemp, DTtilde, eps))/Ktilde
        else:
            ints_p[0] += p
            params_k[0] += K
            params_h[0] += H
            
            origints_p[0] += -1.0

            res = (PhiGo(ints_p, params_k, params_h, Vtemp, DTtilde, eps) -\
            p/DTtilde * PhiGo(origints_p, origparams_k, origparams_h, Vtemp, DTtilde, eps))/Ktilde
        
            
                
    return res




# Example usage:

if __name__ == '__main__':
    
    
    ints_p = deque([0,0])
    params_k = deque([0.5,0])
    params_h = deque([0.9,0])
    Vtemp = 0.01
    DTtilde = 0.1
    eps = 10**(-3.0)
    
    
    print(PhiGo(ints_p, params_k, params_h, Vtemp, DTtilde, eps))
    
    