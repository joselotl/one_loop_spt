"""
This code takes the linear power spectrum of matter (e. g. from a Boltzman solver like CAMB)
and convolves it into its one loop correction by integrating in log k 
"""

import numpy as np
from scipy.integrate import simps
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy import interpolate
import analyticpower
from multiprocessing import Pool

# Definiendo una funcion que sea el espectro lineal
def linearpower(q):
    if q<0:
        print 'k negativa'
        power = 0.
    elif q < qmin:
        power = power_rate_ini*ana.Phs(q)
        #power = power_ini
    elif q> qmax:
        #power = power_end
        power = power_rate_end*ana.Phs(q)
    else:
        power = interpolate.splev(q,tck,der=0)
    return power


def xargument(x,q,k):
    r = q/k
    den = 1. + r**2 - 2.*r*x
    factorx = (3.*r + 7.*x - 10.*r*x**2)**2/den**2
    xargument = linearpower(k*np.sqrt(den))*factorx
    return xargument

def p22_argument(q,k):
    r = q/k
    # The next is to integrate only over the numerical part of P
    #rmin = qmin/k
    #rmax = qmax/k
    #x_bis_min = max(-1.,(1.+r**2-rmax**2)/(2.*r))
    #x_bis_max = min(0.9999999,(1.+r**2-rmin**2)/(2.*r))
    # I use this limits if I trust the analytical extension of P
    x_bis_min = -1.
    x_bis_max = min(0.999999,1/(r*2))
    # We make the integral over x using the interpolated powerspectrum
    xintegral = quad(xargument,x_bis_min,x_bis_max,args=(q,k))[0]
    return xintegral
    
def p22(k):
    aux0 = p22_argument(ktabla,k)
    p22_arg = aux0*ptabla
    p22_integral = simps(p22_arg,ktabla)
    print 'Computing P22 for k = ',k
    p22 = 2./98.*k**2/(4.*np.pi**2)*p22_integral
    # This last step is to put everything back in CAMB like units
    p22 = p22*h**3
    return p22

def p13_argument(q,k):
    """
    Kernels of the P13 integral with their respective aproximations around particular values
    """
    r = q/k
    if r < 5e-3:
        factor = -2./3. + 232./315.*r**2 - 376./735.*r**4
        #-168. + 928./5.*r**2 - 4512./35.*r**4 + 416./21.*r**6
    elif abs(r-1.) < 3e-3:
        factor = (-22. + 2.*(r-1.) - 29.*(r-1.)**2)/63.
        #factor = -88. + 8.*(r-1.)
    elif r > 500.:
        s = 1./r
        factor = -122./315. + 8./105.*s**2 - 40./1323.*s**4
        #factor = -488./5. + 96./5./r**2 - 160./21./r**4 - 1376./1155./r**6
    else:
        factor = (12./r**2 - 158. + 100.*r**2 - 42.*r**4 + 3./r**3*(r**2-1.)**3*(7.*r**2+2.)*np.log(abs((1.+r)/(1.-r))))/252.
    return factor


def p13(k):
    aux1 = p13_argument(ktabla,k)
    p13_arg = aux1*ptabla
    second_integral = simps(p13_arg,ktabla)
    print 'Computing P13 for k =',k
    p13 = k**2/(2.*np.pi)**2*linearpower(k)*second_integral
    # This last step is to put everything in CAMB like units
    p13 = p13*h**3
    return p13

"""
Computing one loop correciton of the matter large scale structure power spectrum.
Several cosmological constants are defined in analyticpower.py, go there if you want to change them.
"""
h = 0.6790

# Initializing both numerical and analyic linear powerspectrum
# the function linearpower uses both at appropiate regions

ana = analyticpower.AnalyticPower()
tabla = np.loadtxt("./planck_linear.dat")
ktabla = tabla[:,0]*h
ptabla = tabla[:,1]*h**(-3)
tck=interpolate.splrep(ktabla,ptabla,s=0)

# Adjusting the limits between numerical and analytical
qmin = ktabla[0]
qmax = ktabla[-1]
power_ini = ptabla[0]
power_end = ptabla[-1]
power_rate_ini = ptabla[0]/ana.Phs(ktabla[0])
power_rate_end = ptabla[-1]/ana.Phs(ktabla[-1])

# Computing the one-loop corrections
p13_argument = np.vectorize(p13_argument)
p22_argument = np.vectorize(p22_argument)
    
if __name__ == '__main__':

    p = Pool(4)
    p22_tabla = p.map(p22,ktabla)
    p13_tabla = p.map(p13,ktabla)

    # This last step is to put everything back in CAMB-like units
    ptabla = ptabla*h**3
    ktabla = ktabla/h
    np.savetxt("./oneloop_Pk.dat",np.transpose([ktabla,ptabla,p22_tabla,p13_tabla]))


