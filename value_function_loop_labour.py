### Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time

### Value function iteration ###########################
### Brute - force version with Howard improvement ######

## Declaration of parameters etc.
beta = 0.988 #discount factor
delta = 0.013 #depreciation rate
theta = 0.679 #capital share
niu = 2
kappa = 5.24
S = 100  #number or discrete states between (and including) k_lo and k_up
Howard = 100 #steps in Howard improvement

conv = np.zeros((1,S)) + 1e-6 #convergence criterion
progress = np.ones((1,S)) #vector for stroing differences between value and value old
comp = np.zeros((1,S)) + 100 #vector for comparison of progress and conv criterion
cont = np.zeros((1,S)) - 1 #vector storing optimal decision
chi = np.zeros((S,S)) 
ones = np.ones([1,S])

## Steady state
h_ss = ((theta*(beta**(-1)-1+delta))/(kappa*(beta**(-1)-1+theta*delta)))**(niu/(1+niu))
k_ss = ((beta*(1-theta)*(h_ss**theta))/(1-beta*(1-beta*delta)))**(1/theta)
c_ss = (k_ss**(1-theta))*(h_ss**theta) -delta*k_ss

## Step 1: set up grid for capital
#k_up = 2*k_ss #upper bound for capital
#k_lo = 0.5*k_ss #lower bound for capital
k_lo = 0.01 #lower bound for capital
k_up = 20 #upper bound for capital
ks = np.arange(k_lo, k_up, ((k_up-k_lo)/(S)) ) #capital grid

## Step 2: initial guess for value function
value = np.zeros((S,1))

## Steps 3 & 4: define matrix M and fill it; care for nonnegativity
start = time.time()
print("VFI + labour time:")
M = np.ones((S,S))*np.inf*(-1)
index = np.arange(0, S, 1).tolist()
for j in index: #for every state today
    for i in index: #for every state tomorrow
        #Here account for labour choice:
        def labfun(h):
            return (ks[j]**(1-theta))*(h**theta)+(1-delta)*ks[j]-ks[i]-(ks[j]**(1-theta))*(h**(theta-1-(1/niu)))*(theta/kappa) #labour implicit formula
        hopt = fsolve(labfun,0.01) #solve for optimal labour
        cij = (ks[j]**(1-theta))*(hopt**(theta))+(1-delta)*ks[j]-ks[i] #consumption        
        if cij>=0:  #consumption and labor must be non-negative
            M[i,j] = np.log(cij)-kappa*(niu/(1+niu))*(hopt**(1+(1/niu)))

    
## Step 5 & 6: Iterate over value funtion until conv criterion is met
#here start time measuring
while np.any(comp) == 1:
    valueold = value
    for i in index:
        for j in index:
            chi[i,j] = M[i,j] + beta*valueold[i]
    value = np.array([np.max( chi, axis = 0)]).reshape((S,1))
    cont = np.argmax(chi, axis = 0) #max index row
    progress = np.max(abs(value-valueold), axis = 0 )
    np.greater(progress, conv, out = comp)
#here end time measuring
end = time.time()
print(end - start) 

## Next period capital
kp = np.zeros((S,1))
for i in index:
    kp[i] = ks[cont[i]]

#Back out labour
hours = np.zeros((S,1))
for j in index:
    def labfun2(h):
        return (ks[j]**(1-theta))*(h**theta)+(1-delta)*ks[j]-kp[j]-(ks[j]**(1-theta))*(h**(theta-1-(1/niu)))*(theta/kappa) #labour implicit formula
    hours[j] = fsolve(labfun2, 0.1) #solve for optimal labour

## Back out consumption
cons = np.zeros((S,1))
for i in index:
    cons[i] = (ks[i]**(1-theta))*(hours[i]**theta)+(1-delta)*ks[i]-kp[i]

## Plot kp vs ks
plt.plot(ks, kp)
plt.show()

## Plot cons vs ks in retation to steady state values
plt.plot(ks,cons)
plt.title('Consumption policy function', fontsize = 18) # title with fontsize 20
plt.ylabel('Value', fontsize = 12) # x-axis label with fontsize 15
plt.xlabel('Capital grid values', fontsize = 12) # y-axis label with fontsize 15
plt.savefig('cons_2a_l.png')
plt.show()

## Plot cons vs ks in retation to steady state values
plt.plot(ks,hours)
plt.title('Labour policy function', fontsize = 18) # title with fontsize 20
plt.ylabel('Value', fontsize = 12) # x-axis label with fontsize 15
plt.xlabel('Capital grid values', fontsize = 12) # y-axis label with fontsize 15
plt.savefig('lab_2a_l.png')
plt.show()

## Plot Value function vs ks
plt.plot(ks, value)
plt.title('Value funcion vs capital grid.', fontsize = 18) # title with fontsize 20
plt.ylabel('Value', fontsize = 12) # x-axis label with fontsize 15
plt.xlabel('Capital grid values', fontsize = 12) # y-axis label with fontsize 15
plt.savefig('VFI_2a_l.png')
plt.show()

## Step 7: play with parameters to be thorough.