"""
This module computes and an analytic aproximation for the 
matter power spectrum (without wiggles) from Eisenstein and Hu, 1998.
"""
import numpy as np
c = 299792458
ns = 0.9681
h = 0.6790
deltah_squared = 1.868e-9
Omega_lambda = 0.6935
Omega_b = 0.022270/h**2
T_CMB = 2.72548

Omega0 = 1 - Omega_lambda
theta27 = T_CMB/2.7
c_over_H0 = c/(100*1000*h)
alpha_gamma = 1 - 0.328*np.log(431*Omega0*h**2)*Omega_b/Omega0 + 0.38*np.log(22.3*Omega0*h**2)*(Omega_b/Omega0)**2 
s = 44.5*np.log(9.83/(Omega0*h**2))/np.sqrt(1+10*(Omega_b*h**2)**0.75)

class AnalyticPower:
    def Transfer(self,k):
        Gamma_eff = Omega0*h*(alpha_gamma + (1-alpha_gamma)/(1+(0.43*k*s)**4))
        q = (k*theta27)/(h*Gamma_eff)
        L0 = np.log(2*np.e + 1.8*q)
        C0 = 14.2 + 731/(1+62.5*q)
        return L0/(L0+C0*q**2)


    def Phs(self,k):
       Phs = 2*np.pi**2/k**3*deltah_squared*(c_over_H0*k)**(3+ns)*self.Transfer(k)**2
       return Phs

