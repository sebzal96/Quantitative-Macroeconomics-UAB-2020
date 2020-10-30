### Packages
import numpy as np
import matplotlib.pyplot as plt
import time

### Value function iteration ###
### Brute - force version ######

## Declaration of parameters etc.
beta = 0.988 #discount factor
delta = 0.013 #depreciation rate
theta = 0.679 #capital share
niu = 2
kappa = 5.24
S = 300  #number or discrete states between (and including) k_lo and k_up
h = 1 #inelastic labour supply
k_lo = 10 #lower bound for capital
Howard = 100 #steps in Howard improvement

conv = np.zeros((1,S)) + 1e-6 #convergence criterion
progress = np.ones((1,S)) #vector for stroing differences between value and value old
comp = np.zeros((1,S)) + 100 #vector for comparison of progress and conv criterion
cont = np.zeros((1,S)) - 1 #vector storing optimal decision
chi = np.zeros((S,S)) 

## Steady state
k_ss = ((beta*(1-theta)*(h**theta))/(1-beta*(1-beta*delta)))**(1/theta)
c_ss = k_ss**(1-theta)-delta*k_ss

start = time.time()
print("VFI with concavity time:")
## Step 1: set up grid for capital
k_up = 50 #upper bound for capital
ks = np.arange(k_lo, k_up, ((k_up-k_lo)/(S)) ) #capital grid

## Step 2: initial guess for value function
value = np.zeros((S,1))

## Steps 3 & 4: define matrix M and fill it; care for nonnegativity
M = np.ones((S,S))*np.inf*(-1)
index = np.arange(0, S, 1).tolist()
for j in index: #for every state today
    for i in index: #for every state tomorrow
        if (ks[j]**(1-theta))*(h**theta)+(1-delta)*ks[j]-ks[i] >= 0: #consumption must be non-negative
            M[i,j] = np.log((ks[j]**(1-theta))*(h**theta)+(1-delta)*ks[j]-ks[i])-kappa*(1+(1/niu))*(h**(1+(1/niu)))

## Step 5 & 6: Iterate over value funtion until conv criterion is met
#here start time measuring
itr=1
while np.any(comp) == 1:
    valueold = value
    for i in index:
        for j in index:
            cell = M[i][j]+beta*valueold[i]
            if i>0 and cell < chi[i-1][j]:
                chi[i][j]=chi[i-1][j]
            else:
                chi[i][j]=cell
    value = np.array([np.max( chi, axis = 0)]).reshape((S,1))
    cont = np.argmax(chi, axis = 0) #max index row
    progress = np.max(abs(value-valueold), axis = 0 )
    np.greater(progress, conv, out = comp)
    itr = itr + 1
#here end time measuring
end = time.time()
print(end - start) 

## Next period capital
kp = np.zeros((S,1))
for i in index:
    kp[i] = ks[cont[i]] 

## Back out consumption
cons = np.zeros((S,1))
for i in index:
    cons[i] = (ks[i]**(1-theta))*(h**theta)+(1-delta)*ks[i]-kp[i]

## Plot kp vs ks
plt.plot(ks, kp)
plt.show()

## Plot cons vs ks in retation to steady state values
plt.plot(ks,cons)
plt.title('Consumption policy function', fontsize = 18) # title with fontsize 20
plt.ylabel('Value', fontsize = 12) # x-axis label with fontsize 15
plt.xlabel('Capital grid values', fontsize = 12) # y-axis label with fontsize 15
plt.savefig('cons_c.png')
plt.show()

## Plot Value function vs ks
plt.plot(ks, value)
plt.title('Value funcion vs capital grid.', fontsize = 18) # title with fontsize 20
plt.ylabel('Value', fontsize = 12) # x-axis label with fontsize 15
plt.xlabel('Capital grid values', fontsize = 12) # y-axis label with fontsize 15
plt.savefig('VFI_c.png')
plt.show()

## Step 7: play with parameters to be thorough.