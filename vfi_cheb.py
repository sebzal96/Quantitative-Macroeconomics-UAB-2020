### VFI with Chebyshev approximation ###

## Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time

## Declaration of parameters etc.:
beta = 0.988 #discount factor
delta = 0.013 #depreciation rate
theta = 0.679 #labour share
niu = 2 
kappa = 5.24
h = 1 #inelastic labour supply
conv = 1e-6 #convergence criterion
n = 20  #number nodes
m = 3 #order of polynomial 
g = np.zeros((n)) #container for kprime
progress = 1


## Steady State:
k_ss = ((beta*(1-theta)*(h**theta))/(1-beta*(1-beta*delta)))**(1/theta)
c_ss = (k_ss**(1-theta))*(h**theta) - delta*k_ss

## Set upper and lowerbound for capital space:
k_lo = 35
k_up = 50

## Construct Chebyshev nodes:
ks = np.zeros((n,1))
z = np.zeros((n,1))
for j in range(0,n):
    z[j] = -np.cos(np.pi*(2*j-1)/(2*n)) #from interval -1 - 1
    ks[j] = (z[j]+1)*((k_up-k_lo)/2)+k_lo #adjusted to k_lo and k_up
  
## Basis functions:
psi = np.zeros((n,m+1))
psi[:,0] = 1
psi[:,1] = ks[:,0]
psiZ = np.zeros((n,m+1))
psiZ[:,0] = 1
psiZ[:,1] = z[:,0]
for i in range(1,m):
    psi[:,i+1] = 2*ks[:,0]*psi[:,i]-psi[:,i-1]
    psiZ[:,i+1] = 2*z[:,0]*psiZ[:,i]-psiZ[:,i-1]

## Initial guess for coefficients:
coef = np.zeros([m+1])
for i in range(0,m+1):
    coef[i] = 0.000000001
    
## Initial value function:
vf = np.matmul(psi,coef)

## Chebyshev regression:
for k in range(0,m+1):
    coef[k] = (np.matmul(vf,psiZ[:,k]))/(np.matmul(psiZ[:,k],psiZ[:,k]))
    
## Iterate until convergence criterion is satisfied:    
while progress > conv:
    coef_old = coef
    
    ## approximate Vf:
    aprox_vf = np.matmul(psi,coef_old)
        
    ## Solve for k prime:
    for j in range(0,n):
        def focvalfun(kprim):
            return 1 - beta*(coef[0]-coef[2]+coef[1]*kprim+2*coef[2]*(kprim**2))*((ks[j]**(1-theta))*(h**theta)+(1-delta)*ks[j]-kprim)
        g[j] = fsolve(focvalfun,k_ss)
        
    ## Evaluate VF:
    for j in range(0,n):
        if (ks[j]**(1-theta))*(h**theta)+(1-delta)*ks[j]-g[j]>0:
            vf[j] = np.log((ks[j]**(1-theta))*(h**theta)+(1-delta)*ks[j]-g[j])-kappa*(1+(1/niu))*(h**(1+(1/niu)))+beta*(coef_old[0]+coef_old[1]*g[j]+coef_old[2]*(2*g[j]**2 - 1))
        else:
            vf[j] = -10**10
            
    ## Obtain coefficients:
    for k in range(0,m+1):
        coef[k] = (np.matmul(vf[:],psiZ[:,k]))/(np.matmul(psiZ[:,k],psiZ[:,k]))
        
    ## obtain progress:
    progress = np.max( abs(np.subtract(coef,coef_old)))    
    
## Back out consumption
cons = np.zeros((n,1))
for i in range(0,n):
    cons[i] = (ks[i]**(1-theta))*(h**theta)+(1-delta)*ks[i]-g[i]

## Plot kp vs ks
plt.plot(ks, g)
plt.show()

## Plot cons vs ks in retation to steady state values
plt.plot(ks,cons)
plt.title('Consumption policy function', fontsize = 18) # title with fontsize 20
plt.ylabel('Value', fontsize = 12) # x-axis label with fontsize 15
plt.xlabel('Capital grid values', fontsize = 12) # y-axis label with fontsize 15
plt.savefig('VFI_cheb.png')
plt.show()

## Plot Value function vs ks
plt.plot(ks, vf)
plt.title('Value funcion vs capital grid.', fontsize = 18) # title with fontsize 20
plt.ylabel('Value', fontsize = 12) # x-axis label with fontsize 15
plt.xlabel('Capital grid values', fontsize = 12) # y-axis label with fontsize 15
plt.savefig('VFI_cheb.png')
plt.show()    
    
    
    
    
    
    
    
    
    