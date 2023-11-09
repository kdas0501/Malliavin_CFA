# -*- coding: utf-8 -*-
"""

@author: Dr Kaustav Das (kaustav.das@monash.edu)

Article: "Explicit approximations of option prices via Malliavin calculus in a
general stochastic volatility framework"

Description: Computes the price of a European put option in the Verhulst model 
with piecewise-constant parameters:
    
dS = (rd - rf)S*dt + V*S*dW,
dV = kap(the - V)V*dt + lam*V*dB,
dW dB = rho dt

via both the Monte-Carlo and Malliavin closed-form approximation methodology, 
and compares the prices, implied volatilities and runtimes.

"""


import numpy as np
from collections import deque
from Verhulst_CFA_Malliavin import Verhulst_CFA_Malliavin
from Monte_mixing_pw import Monte_mixing_pw
from ImpVolBrent_pw import ImpVolBrent_pw
import copy as cp


def Verhulst_compare_pw(model_params, global_params, N_PATH, N_TIME): 
    
      """
      Computes the price of a European put option in the Verhulst model 
      with piecewise-constant parameters via both the Monte-Carlo and Malliavin closed-form approximation methodology, 
      and compares the prices, implied volatilities and runtimes.
      
      model_params (list): [kap_ar, the_ar, lam_ar, rho_ar] is a list of np.arrays, 
        where e.g. kap_ar = np.array([kap2, kap1]), piecewise parameters are 
        specified backward in time.
      globals_params (list): [S0, V0, rd_deque, rf_deque, Strk, dt, dttilde].
            S0 (float): initial spot.
            V0 (float): initial variance/volatility.
            rd_deque (deque): domestic interest rate, given backward, e.g., rd_deque = deque([rd2, rd1]).
            rf_deque (deque): foreign interest rate, given backward, e.g., rf_deque = deque([rf2, rf1]).
            Strk (float): Strike of the contract.
            dt (deque): deque of time increments over which each parameter is 'alive',
            given backward, e.g., dt = deque([dt2, dt1]). Note sum(dt) gives option maturity T.
            N_dttilde (int): Number of points in time grid to solve the ODE for v_{0, t}.
      N_PATH (int): # of Monte-Carlo paths.
      N_TIME (int): # of time discretisation points for SDE integration.

      """
        
    
      S0 = global_params[0]
      # V0 = global_params[1]
      rd_deque = global_params[2]
      rf_deque = global_params[3]
      Strk = global_params[4]
      dt = global_params[5]
      # dttilde = global_params[6]
      
      global_params_Monte = cp.copy(global_params)
      global_params_Monte.pop
      
      # Compute prices
      (VPrice_approx, elapsed_approx) = Verhulst_CFA_Malliavin(model_params, global_params) 
      (VPrice_Monte, __, elapsed_MC)  = Monte_mixing_pw(model_params, global_params_Monte, N_PATH, N_TIME, 'v', 'Put')
      
      # Compute implied volatilities  
      ImpVolV_Monte = ImpVolBrent_pw(S0, Strk, rd_deque, rf_deque, dt, 'Put', VPrice_Monte)
      ImpVolV_approx = ImpVolBrent_pw(S0, Strk, rd_deque, rf_deque, dt, 'Put', VPrice_approx) 
      
      # Compute implied volatilities in cent
      ImpVolV_approx_cent = 100*ImpVolV_approx
      ImpVolV_Monte_cent = 100*ImpVolV_Monte
      
      # Absolute and relative error of option price
      Abs_Err = abs(VPrice_approx - VPrice_Monte)
      Abs_Err_cent = 100*Abs_Err
      Rel_Err_cent = 100*Abs_Err/VPrice_Monte
            
      # Absolute and relative error of Implied vols
      ImpVol_AbsErr = abs(ImpVolV_approx - ImpVolV_Monte)
      ImpVol_AbsErr_bp = 10000*ImpVol_AbsErr
      ImpVol_RelErr_cent = 100*ImpVol_AbsErr/ImpVolV_Monte if ImpVolV_Monte != 0 else 0
      
      # speedup
      speedup = elapsed_MC/elapsed_approx
      
      
      
      
      
      # Print some things!
      print("\nModel: Verhulst \n")

      print("Put prices")
      print(f'MC {VPrice_Monte:.5f} CF {VPrice_approx:.5f} AbsErr {Abs_Err_cent:.3f}% RelErr {Rel_Err_cent:.3f} \n')
      
      print("Implied Vols")
      print(f'MC {ImpVolV_Monte_cent:.3f}% CF {ImpVolV_approx_cent:.3f}% AbsErr {ImpVol_AbsErr_bp:.3f}bp RelErr {ImpVol_RelErr_cent:.3f}% \n')

            
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
      dt3 = 6/12
      dt2 = 3/12
      dt1 = 3/12
      
      rd_deque = deque([rd3, rd2, rd1])
      rf_deque = deque([rf3, rf2, rd1])
      dt = deque([dt3, dt2, dt1])
      
      # rd_deque = deque([rd3])
      # rf_deque = deque([rf3])
      # dt = deque([dt3])
      
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
      
      # kap_ar = np.array([kap3])
      # the_ar = np.array([the3])
      # lam_ar = np.array([lam3])
      # rho_ar = np.array([rho3])
      
      # N_dttilde is the number of points in the grid for solving the ODE for v_{0,t}. 

      N_dttilde = 35
      
      
      
      # Monte-Carlo parameters
      T = sum(dt)
      N_PATH = 1000000
      Steps_day_N = 24
      N_TIME = int(Steps_day_N*253*T)
      
      
      
      global_params = [S0, V0, rd_deque, rf_deque, Strk, dt, N_dttilde]
      model_params = [kap_ar, the_ar, lam_ar, rho_ar]
       
      Verhulst_compare_pw(model_params, global_params, N_PATH, N_TIME)
            