# Malliavin_CFA
This repository contains working code for pricing European put options with piecewise-constant parameters in the Verhulst and SABR-mu models using the Malliavin calculus closed-form approximation formula detailed in Section 6 and Appendix F of the article 'Explicit approximations of option prices via Malliavin calculus in a general stochastic volatility framework'. 

Title: Explicit approximations of option prices via Malliavin calculus in a general stochastic volatility framework

Authors: Kaustav Das (Monash University) and Nicolas Langrené (BNU-HKBU United International College).

email addresses: kaustav.das@monash.edu and nicolaslangrene@uic.edu.cn




### Quickstart for readers of article

Simply run Verhulst_compare_pw.py, which computes and compares the price and implied volatility of a European put option in the Verhulst model with piecewise-constant parameter inputs, where the price is obtained via the Malliavin calculus closed-form approximation method and mixing solution Monte-Carlo method.

### Main files 

The following .py files are required in order to compute the closed-form approximation formula.

- **Verhulst_CFA_Malliavin.py:**
  Computes the price of a European put option in the Verhulst model with piecewise-constant parameter inputs via the Malliavin calculus closed-form approximation method.

- **SABRmu_CFA_Malliavin.py:**
  Computes the price of a European put option in the SABR-mu model with piecewise-constant parameter inputs via the Malliavin calculus closed-form approximation method.
  
- **ODE_Solver.py:**
  Computes the ODE for v_{0, t} in the Verhulst model with piecewise-constant parameter inputs.
    
- **OmgG.py:**
  Contains methods OmgG1, OmgG2, OmgG3, OmgG4, methods which compute the integral operator (1 to 4 fold) for piecewise-constant parameter inputs.
  
- **PhiGo.py:**
  Computes the n-fold function Phi for any n.
  
- **parPBS_pw.py:**
  Computes the second-order partial derivatives of the function P_BS for piecewise-constant parameter inputs.
  




### Auxiliary files
The rest of the .py files are auxiliary files that are not required for the closed-form approximation formula.

- **Monte_mixing_pw.py:**
    Computes the price of a European put/call option in the Heston, GARCH diffusion, Ornstein-Uhlenbeck, Inverse-Gamma, SABR-mu, and Verhulst models with piecewise-constant parameter inputs via the mixing solution Monte-Carlo method.
  
- **Verhulst_compare_pw.py:**
    Compares the price and implied volatility of a European put option in the Verhulst model with piecewise-constant parameter inputs, where the price is obtained via the Malliavin calculus closed-form approximation method and mixing solution Monte-Carlo method.

- **SABRmu_compare_pw.py:**
    Compares the price and implied volatility of a European put option in the SABR-mu model with piecewise-constant parameter inputs, where the price is obtained via the Malliavin calculus closed-form approximation method and mixing solution Monte-Carlo method.
    
- **BSform_pw.py:** 
  Computes the usual Black-Scholes price of a European put/call option for piecewise-constant parameter inputs.
  
- **ImpVolBrent_pw.py:** 
  Computes the implied volatility of a European put/call option with piecewise-constant parameter inputs via Brent's method. 
  
- **DeltaStrikes_pw.py:**
  Computes the strike of an option contract corresponding to a given European put/call option Delta with piecewise-constant parameter inputs.
