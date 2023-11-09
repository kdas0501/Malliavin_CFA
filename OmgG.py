# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Article: "Explicit approximations of option prices via Malliavin calculus in a
general stochastic volatility framework"

Description: Computes the omega functions up to 4 dimensions from Section 6 of the article.

Methods: 

- Expo(_k, _dt)
- OmgG1(_k1, _h1, _l1, _v, q, _dttilde)
- OmgG2(_k2, _h2, _l2, _k1, _h1, _l1, _v, q, _dttilde)
- OmgG3(_k3, _h3, _l3, _k2, _h2, _l2, _k1, _h1, _l1, _v, q, _dttilde)
- OmgG4(_k4, _h4, _l4, _k3, _h3, _l3, _k2, _h2, _l2, _k1, _h1, _l1, _v, q, _dttilde)

"""


import copy as cp
from collections import deque
from math import exp
from PhiGo import PhiGo

def Expo(_k, _dt):

    """
    Computes e^{k dotproduct dt}.

    _k (deque)
    _dt (deque)
    """
    
    
    # Copy deques
    k = cp.copy(_k)
    dt = cp.copy(_dt)
        
    lastlayer = k == deque([])
    
    if lastlayer:
        return 1
    else:
        K = k.popleft()
        DT = dt.popleft()
    
    res = Expo(k, dt)*exp(DT*K)
    
    return res



# Example usage:

if __name__ == '__main__':
    
    
    dt = deque([1, 2])
    k = deque([0.5, 3])
    
    print(Expo(k, dt))

















def OmgG1(_k1, _h1, _l1, _v, q, _dttilde):

    """
    Computes the 1D Omega function.

    k1 (deque)
    h1 (deque)
    l1 (deque)
    dttilde (deque)
    v (deque): corresponds to solution of ODE v_{0, t}.
    q: Integer power of _v.

    """

    # Hard code the precision for Taylor expansions in PhiGo
    eps = 10**(-3.0)
        
    # Copy deques
    k1 = cp.copy(_k1)
    h1 = cp.copy(_h1)
    l1 = cp.copy(_l1)
    v = cp.copy(_v)
    dttilde = cp.copy(_dttilde)
    
    lastlayer = k1 == deque([])
    if lastlayer:
        return 0
    else:
        K1 = k1.popleft()
        H1 = h1.popleft()
        L1 = l1.popleft()
        Q1 = q.popleft()
        Vtemp = v.popleft()
        
        kbar = deque([])
        tempk1 = cp.copy(k1)
        temph1 = cp.copy(h1)
        
        tempv = cp.copy(v)
        
        DTtilde = dttilde.popleft()
        
        while tempk1 != deque([]):
            tempK1 = tempk1.popleft()
            tempH1 = temph1.popleft()
            tempV = tempv.popleft()
            kbar.append(tempK1 + tempH1*tempV)
            
        Exp1 = Expo(kbar, dttilde)
        
        VtempeQ1 = Vtemp**Q1
        
        res = OmgG1(k1, h1, l1, v, deque([Q1]), dttilde)+\
        L1*VtempeQ1*Exp1*PhiGo(deque([0]), deque([K1]), deque([H1]), Vtemp, DTtilde, eps)
        
    return res




# Example code.

if __name__ == '__main__':

    
    k1 = deque([.5])
    h1 = deque([.3])
    l1 = deque([.9])
    v = deque([.3])
    q = deque([0])
    dttilde = deque([0.5])
    
    print(OmgG1(k1, h1, l1, v, q, dttilde))














# 2D Omega function from General model Malliavin exp paper. 
# k1, h1 and l1 are deques of piecewise-constant parameters and dttilde is the 
# time discretisation deque

def OmgG2(_k2, _h2, _l2, _k1, _h1, _l1, _v, q, _dttilde):

    """
    Computes the 2D Omega function.

    k2, k1 (deque)
    h2, h1 (deque)
    l2, l1 (deque)
    v (deque): corresponds to solution of ODE v_{0, t}.
    q (deque): deque of integer powers of v.
    dttilde (deque): A deque of time increments over which parameters are 'alive',  
        e.g., dttilde = deque([dttilde2, dttilde1]), piecewise parameters are specified backward in time.

    """

    # Hard code the precision for Taylor expansions in PhiGo
    eps = 10**(-3.0)
    
    # Copy deques
    k2 = cp.copy(_k2)
    h2 = cp.copy(_h2)
    l2 = cp.copy(_l2)
    k1 = cp.copy(_k1)
    h1 = cp.copy(_h1)
    l1 = cp.copy(_l1)
    v = cp.copy(_v)
    dttilde = cp.copy(_dttilde)
    
    lastlayer = k1 == deque([])
    if lastlayer:
        return 0
    else:
        K2 = k2.popleft()
        H2 = h2.popleft()
        L2 = l2.popleft()
        K1 = k1.popleft()
        H1 = h1.popleft()
        L1 = l1.popleft()
        Q2 = q.popleft()
        Q1 = q.popleft()
        Vtemp = v.popleft()
        
        k2bar = deque([])
        tempk2 = cp.copy(k2)
        temph2 = cp.copy(h2)
        
        k1bar = deque([])
        tempk1 = cp.copy(k1)
        temph1 = cp.copy(h1)
        
        tempv = cp.copy(v)
        
        DTtilde = dttilde.popleft()
        
        while tempk1 != deque([]):
            tempK2 = tempk2.popleft()
            tempH2 = temph2.popleft()
            tempK1 = tempk1.popleft()
            tempH1 = temph1.popleft()
            tempV = tempv.popleft()
            k2bar.append(tempK2 + tempH2*tempV)
            k1bar.append(tempK1 + tempH1*tempV)
        
        L2L1 = L2*L1
        Exp1 = Expo(k1bar, dttilde)
        Exp21 = Expo(k2bar,dttilde)*Exp1
        
        VtempeQ1 = Vtemp**Q1
        VtempeQ2Q1 = (Vtemp**Q2)*VtempeQ1
        
        res = OmgG2(k2, h2, l2, k1, h1, l1, v, deque([Q2,Q1]), dttilde)+\
        L1*VtempeQ1*Exp1*PhiGo(deque([0]), deque([K1]), deque([H1]), Vtemp, DTtilde,eps)*OmgG1(k2, h2, l2, v, deque([Q2]), dttilde)+\
        L2L1*VtempeQ2Q1*Exp21*PhiGo(deque([0,0]),deque([K2,K1]), deque([H2,H1]), Vtemp, DTtilde, eps)
        
    return res




# Example code.

if __name__ == '__main__':


    k2 = deque([0.4,0.2])
    h2 = deque([0.4,0.2])
    l2 = deque([0.4,0.2])
    k1 = deque([0.4,0.2])
    h1 = deque([0.4,0.2])
    l1 = deque([0.4,0.2])
    v = deque([0.4,0.2])
    q = deque([3,5])
    dttilde = deque([0.6,0.3])
    
    print(OmgG2(k2, h2, l2, k1, h1, l1, v, q, dttilde))















def OmgG3(_k3, _h3, _l3, _k2, _h2, _l2, _k1, _h1, _l1, _v, q, _dttilde):

    """
    Computes the 3D Omega function.

    k3, k2, k1 (deque)
    h3, h2, h1 (deque)
    l3, l2, l1 (deque)
    v (deque): corresponds to solution of ODE v_{0, t}.
    q (deque): deque of integer powers of v.
    dttilde (deque): A deque of time increments over which parameters are 'alive',  
        e.g., dttilde = deque([dttilde2, dttilde1]), piecewise parameters are specified backward in time.

    """

    # Hard code the precision for Taylor expansions in PhiGo
    eps = 10**(-3.0)
            
    # Copy deques
    k3 = cp.copy(_k3)
    h3 = cp.copy(_h3)
    l3 = cp.copy(_l3)
    k2 = cp.copy(_k2)
    h2 = cp.copy(_h2)
    l2 = cp.copy(_l2)
    k1 = cp.copy(_k1)
    h1 = cp.copy(_h1)
    l1 = cp.copy(_l1)
    v = cp.copy(_v)
    dttilde = cp.copy(_dttilde)
    
    lastlayer = k1 == deque([])
    if lastlayer:
        return 0
    else:
        K3 = k3.popleft()
        H3 = h3.popleft()
        L3 = l3.popleft()
        K2 = k2.popleft()
        H2 = h2.popleft()
        L2 = l2.popleft()
        K1 = k1.popleft()
        H1 = h1.popleft()
        L1 = l1.popleft()
        Q3 = q.popleft()
        Q2 = q.popleft()
        Q1 = q.popleft()
        Vtemp = v.popleft()
        
        k3bar = deque([])
        tempk3 = cp.copy(k3)
        temph3 = cp.copy(h3)
        
        k2bar = deque([])
        tempk2 = cp.copy(k2)
        temph2 = cp.copy(h2)
        
        k1bar = deque([])
        tempk1 = cp.copy(k1)
        temph1 = cp.copy(h1)
        
        tempv = cp.copy(v)
        
        DTtilde = dttilde.popleft()
        
        while tempk1 != deque([]):
            tempK3 = tempk3.popleft()
            tempH3 = temph3.popleft()
            tempK2 = tempk2.popleft()
            tempH2 = temph2.popleft()
            tempK1 = tempk1.popleft()
            tempH1 = temph1.popleft()
            tempV = tempv.popleft()
            k3bar.append(tempK3 + tempH3*tempV)
            k2bar.append(tempK2 + tempH2*tempV)
            k1bar.append(tempK1 + tempH1*tempV)
            
        L2L1 = L2*L1
        L3L2L1 = L3*L2L1
        Exp1 = Expo(k1bar, dttilde)
        Exp21 = Expo(k2bar, dttilde)*Exp1
        Exp321 = Expo(k3bar, dttilde)*Exp21 
        
        VtempeQ1 = Vtemp**Q1
        VtempeQ2Q1 = (Vtemp**Q2)*VtempeQ1        
        VtempeQ3Q2Q1 = (Vtemp**Q3)*VtempeQ2Q1        
        
        res = OmgG3(k3, h3, l3, k2, h2, l2, k1, h1, l1, v, deque([Q3,Q2,Q1]), dttilde)+\
        L1*VtempeQ1*Exp1*PhiGo(deque([0]), deque([K1]), deque([H1]),Vtemp,DTtilde,eps)*OmgG2(k3, h3, l3, k2, h2, l2, v, deque([Q3, Q2]), dttilde)+\
        L2L1*VtempeQ2Q1*Exp21*PhiGo(deque([0,0]), deque([K2,K1]), deque([H2,H1]), Vtemp, DTtilde, eps)*OmgG1(k3, h3, l3, v, deque([Q3]), dttilde)+\
        L3L2L1*VtempeQ3Q2Q1*Exp321*PhiGo(deque([0,0,0]),deque([K3,K2,K1]),deque([H3,H2,H1]),Vtemp,DTtilde,eps)
        
        return res
    
    

# Example code.

if __name__ == '__main__':     
        
        
    
    k3 = deque([0.9,0.2])
    h3 = deque([0.4,0.2])
    l3 = deque([0.4,1.2])
    k2 = deque([0.4,0.2])
    h2 = deque([0.4,0.2])
    l2 = deque([0.4,0.2])
    k1 = deque([0.4,0.2])
    h1 = deque([0.4,0.2])
    l1 = deque([0.4,0.2])
    v = deque([0.4,0.2])
    q = deque([3,5,1])
    dttilde = deque([0.6,0.3])
    
    print(OmgG3(k3, h3, l3, k2, h2, l2, k1, h1, l1, v, q, dttilde))












def OmgG4(_k4, _h4, _l4, _k3, _h3, _l3, _k2, _h2, _l2, _k1, _h1, _l1, _v, q, _dttilde):

    """
    Computes the 4D Omega function.

    k4, k3, k2, k1 (deque)
    h4, h3, h2, h1 (deque)
    l4, l3, l2, l1 (deque)
    v (deque): corresponds to solution of ODE v_{0, t}.
    q (deque): deque of integer powers of v.
    dttilde (deque): A deque of time increments over which parameters are 'alive',  
        e.g., dttilde = deque([dttilde2, dttilde1]), piecewise parameters are specified backward in time.

    """

    # Hard code the precision for Taylor expansions in PhiGo
    eps = 10**(-3.0)
    
    # Copy deques
    k4 = cp.copy(_k4)
    h4 = cp.copy(_h4)
    l4 = cp.copy(_l4)
    k3 = cp.copy(_k3)
    h3 = cp.copy(_h3)
    l3 = cp.copy(_l3)
    k2 = cp.copy(_k2)
    h2 = cp.copy(_h2)
    l2 = cp.copy(_l2)
    k1 = cp.copy(_k1)
    h1 = cp.copy(_h1)
    l1 = cp.copy(_l1)
    v = cp.copy(_v)
    dttilde = cp.copy(_dttilde)
    
    lastlayer = k1 == deque([])
    if lastlayer:
        return 0
    else:
        K4 = k4.popleft()
        H4 = h4.popleft()
        L4 = l4.popleft()
        K3 = k3.popleft()
        H3 = h3.popleft()
        L3 = l3.popleft()
        K2 = k2.popleft()
        H2 = h2.popleft()
        L2 = l2.popleft()
        K1 = k1.popleft()
        H1 = h1.popleft()
        L1 = l1.popleft()
        Q4 = q.popleft()
        Q3 = q.popleft()
        Q2 = q.popleft()
        Q1 = q.popleft()
        Vtemp = v.popleft()
        
        k4bar = deque([])
        tempk4 = cp.copy(k4)
        temph4 = cp.copy(h4)
        
        k3bar = deque([])
        tempk3 = cp.copy(k3)
        temph3 = cp.copy(h3)
        
        k2bar = deque([])
        tempk2 = cp.copy(k2)
        temph2 = cp.copy(h2)
        
        k1bar = deque([])
        tempk1 = cp.copy(k1)
        temph1 = cp.copy(h1)
        
        tempv = cp.copy(v)
        
        DTtilde = dttilde.popleft()
        
        while tempk1 != deque([]):
            tempK4 = tempk4.popleft()
            tempH4 = temph4.popleft()
            tempK3 = tempk3.popleft()
            tempH3 = temph3.popleft()
            tempK2 = tempk2.popleft()
            tempH2 = temph2.popleft()
            tempK1 = tempk1.popleft()
            tempH1 = temph1.popleft()
            tempV = tempv.popleft()
            k4bar.append(tempK4 + tempH4*tempV)
            k3bar.append(tempK3 + tempH3*tempV)
            k2bar.append(tempK2 + tempH2*tempV)
            k1bar.append(tempK1 + tempH1*tempV)
            
            
        L2L1 = L2*L1
        L3L2L1 = L3*L2L1
        L4L3L2L1 = L4*L3L2L1
        Exp1 = Expo(k1bar, dttilde)
        Exp21 = Expo(k2bar, dttilde)*Exp1
        Exp321 = Expo(k3bar, dttilde)*Exp21
        Exp4321 = Expo(k4bar, dttilde)*Exp321
        
        VtempeQ1 = Vtemp**Q1
        VtempeQ2Q1 = (Vtemp**Q2)*VtempeQ1
        VtempeQ3Q2Q1 = (Vtemp**Q3)*VtempeQ2Q1
        VtempeQ4Q3Q2Q1 = (Vtemp**Q4)*VtempeQ3Q2Q1
        
        res = OmgG4(k4, h4, l4, k3, h3, l3, k2, h2, l2, k1, h1, l1, v, deque([Q4,Q3,Q2,Q1]), dttilde)+\
        L1*VtempeQ1*Exp1*PhiGo(deque([0]), deque([K1]), deque([H1]),Vtemp,DTtilde,eps)*OmgG3(k4, h4, l4, k3, h3, l3, k2, h2, l2, v, deque([Q4, Q3, Q2]), dttilde)+\
        L2L1*VtempeQ2Q1*Exp21*PhiGo(deque([0,0]), deque([K2,K1]), deque([H2,H1]), Vtemp, DTtilde, eps)*OmgG2(k4, h4, l4, k3, h3, l3, v,  deque([Q4, Q3]),dttilde)+\
        L3L2L1*VtempeQ3Q2Q1*Exp321*PhiGo(deque([0,0,0]),deque([K3,K2,K1]),deque([H3,H2,H1]),Vtemp,DTtilde,eps)*OmgG1(k4, h4, l4, v, deque([Q4]),dttilde)
        L4L3L2L1*VtempeQ4Q3Q2Q1*Exp4321*PhiGo(deque([0,0,0,0]),deque([K4,K3,K2,K1]),deque([H4,H3,H2,H1]),Vtemp, DTtilde,eps)
        
        return res
    
    


# Example code.

if __name__ == '__main__':



    k4 = deque([0.9,0.2])
    h4 = deque([0.4,0.2])
    l4 = deque([0.4,1.2])  
    k3 = deque([0.9,0.2])
    h3 = deque([0.4,0.2])
    l3 = deque([0.4,1.2])
    k2 = deque([0.4,0.2])
    h2 = deque([0.4,0.2])
    l2 = deque([0.4,0.2])
    k1 = deque([0.4,0.2])
    h1 = deque([0.4,0.2])
    l1 = deque([0.4,0.2])
    v = deque([0.4,0.2])
    q = deque([3,5,1,5])
    dttilde = deque([0.6,0.3])
    
    print(OmgG4(k4, h4, l4, k3, h3, l3, k2, h2, l2, k1, h1, l1, v, q, dttilde))

