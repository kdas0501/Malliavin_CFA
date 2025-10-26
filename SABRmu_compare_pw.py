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

via both the Monte-Carlo and Malliavin closed-form approximation methodology, 
and compares the prices, implied volatilities and runtimes.

"""


import numpy as np
from scipy.stats import norm
from collections import deque
from SABRmu_CFA_Malliavin import SABRmu_CFA_Malliavin
from Monte_mixing_pw import Monte_mixing_pw
from ImpVolBrent_pw import ImpVolBrent_pw
import copy as cp


def SABRmu_compare_pw(model_params, mu, global_params, N_PATH, N_TIME): 
    
      """
      Computes the price of a European put option in the SABR-mu model 
      with piecewise-constant parameters via both the Monte-Carlo and Malliavin closed-form approximation methodology, 
      and compares the prices, implied volatilities and runtimes.
      
      model_params (list): [lam_ar, rho_ar] is a list of np.arrays, 
        where e.g. lam_ar = np.array([lam2, lam1]), piecewise parameters are 
        specified backward in time.
      mu (float): Power of V in the SDE for dV. Should be chosen in the interval [1/2, 1]. mu = 1 gives the classical SABR model.
      globals_params (list): [S0, V0, rd_deque, rf_deque, Strk, dt].
            S0 (float): initial spot.
            V0 (float): initial volatility.
            rd_deque (deque): domestic interest rate, given backward, e.g., rd_deque = deque([rd2, rd1]).
            rf_deque (deque): foreign interest rate, given backward, e.g., rf_deque = deque([rf2, rf1]).
            Strk (float): Strike of the contract.
            dt (deque): deque of time increments over which each parameter is 'alive',
            given backward, e.g., dt = deque([dt2, dt1]). Note sum(dt) gives option maturity T.
      N_PATH (int): # of Monte-Carlo paths.
      N_TIME (int): # of time discretisation points for SDE integration.

      """
        
    
      S0 = global_params[0]
      # V0 = global_params[1]
      rd_deque = global_params[2]
      rf_deque = global_params[3]
      Strk = global_params[4]
      dt = global_params[5]

      
    
      global_params_Monte = cp.copy(global_params)
      global_params_Monte.pop


      # We need to adjust the SABR model_params to input into Monte_mixing_pw
      mu_array = np.full(len(model_params[0]), mu)
      model_params_sabr = np.vstack([mu_array, mu_array, model_params])

      # Compute prices
      # Compute the closed-form approximation several times to get an average of the run-time
      elapsed_temp = 0
      num_its_CF = 10
      for i in range(num_its_CF):
            (SPrice_approx, elapsed_approx) = SABRmu_CFA_Malliavin(model_params, mu, global_params)
            elapsed_temp += elapsed_approx 
      
      elapsed_approx = elapsed_temp/num_its_CF

      # Compute MC price
      (SPrice_Monte, SStdErr_Monte, elapsed_MC)  = Monte_mixing_pw(model_params_sabr, global_params_Monte, N_PATH, N_TIME, 's', 'Put')
      
      SStdErr_Monte_cent = SStdErr_Monte*100

      # Compute implied volatilities  
      ImpVolS_Monte = ImpVolBrent_pw(S0, Strk, rd_deque, rf_deque, dt, 'Put', SPrice_Monte)
      ImpVolS_approx = ImpVolBrent_pw(S0, Strk, rd_deque, rf_deque, dt, 'Put', SPrice_approx) 
      
      # Compute implied volatilities in cent
      ImpVolS_approx_cent = 100*ImpVolS_approx
      ImpVolS_Monte_cent = 100*ImpVolS_Monte
      
      # Absolute and relative error of option price
      Abs_Err = abs(SPrice_approx - SPrice_Monte)
      Abs_Err_cent = 100*Abs_Err
      Rel_Err_cent = 100*Abs_Err/SPrice_Monte
            
      # Absolute and relative error of Implied vols
      ImpVol_AbsErr = abs(ImpVolS_approx - ImpVolS_Monte)
      ImpVol_AbsErr_bp = 10000*ImpVol_AbsErr
      ImpVol_RelErr_cent = 100*ImpVol_AbsErr/ImpVolS_Monte if ImpVolS_Monte != 0 else 0

      rsumdt = 0
      T = 0
      lastlayer = deque([])
      while dt != lastlayer:
        DT = dt.popleft()
        RD = rd_deque.popleft()
        RF = rf_deque.popleft()
        R = RD - RF
        rsumdt += R*DT
        T += DT

      # Compute impvol 'standard error'              
      dpl_Monte = (np.log(S0/Strk) + rsumdt)/(ImpVolS_Monte*np.sqrt(T)) + 0.5*ImpVolS_Monte*np.sqrt(T)
      Deriv_vega_Monte = S0*np.sqrt(T)*norm.pdf(dpl_Monte)
      ImpVol_Monte_StdErr_bp = 10000*(Deriv_vega_Monte**(-1))*SStdErr_Monte

      # speedup
      speedup = elapsed_MC/elapsed_approx
      
      
      
      
      
      # Print some things!
      print(f"\nModel: SABR-{mu:.2f} \n")

      print("Put prices")
      print(f'MC {SPrice_Monte:.5f} CF {SPrice_approx:.5f} AbsErr {Abs_Err_cent:.3f}% RelErr {Rel_Err_cent:.3f}% StdErr {SStdErr_Monte_cent:.3f}%\n')
      
      print("Implied Vols")
      print(f'MC {ImpVolS_Monte_cent:.3f}% CF {ImpVolS_approx_cent:.3f}% AbsErr {ImpVol_AbsErr_bp:.3f}bp RelErr {ImpVol_RelErr_cent:.3f}% StdErr {ImpVol_Monte_StdErr_bp:.3f}bp \n')

      print("Elapsed time")
      print(f'MC {elapsed_MC:.3f}s CF {elapsed_approx:.3f}s Speed up {speedup:.2f}x')
      

      return

    







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
      T = 1/12           # maturity
      dt3 = 1/2*T
      dt2 = 1/4*T
      dt1 = 1/4*T
      
      rd_deque = deque([rd3, rd2, rd1])
      rf_deque = deque([rf3, rf2, rf1])
      dt = deque([dt3, dt2, dt1])
      
      # rd_deque = deque([rd3])
      # rf_deque = deque([rf3])
      # dt = deque([dt3])
      
      Delta = 0.5
      sig = np.sqrt(V0)
      Strk = DeltaStrikes_pw(S0, sig, rd_deque, rf_deque, dt, Delta, 'Put')
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
      
      
      # Monte-Carlo parameters
      # T = sum(dt)
      N_PATH = 250000
      Steps_day_N = 24
      N_TIME = int(Steps_day_N*253*T)
      
      
      
      global_params = [S0, V0, rd_deque, rf_deque, Strk, dt]
      model_params = [lam_ar, rho_ar]
       
      SABRmu_compare_pw(model_params, mu, global_params, N_PATH, N_TIME)
            
