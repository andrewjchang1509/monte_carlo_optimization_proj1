import numpy as np
import matplotlib.pyplot as plt
def pif(x,y):
    return 1/(2*np.pi)*np.exp(-1/2 *((x-2)**2+(y-2)**2))
def gif(x,y,sigma):
    return 1/(2*np.pi*sigma**2)*np.exp(-1/(2*sigma**2)*(x**2+y**2))
    
sizes = sorted(np.round(np.power(10, np.arange(1, 7.2, 0.2))).astype(int))

n_sizes = len(sizes)
theta1 = np.zeros(n_sizes)
theta2 = np.zeros(n_sizes)
theta3 = np.zeros(n_sizes)
ess2 = np.zeros(n_sizes)
ess3 = np.zeros(n_sizes)

for i in range(n_sizes):
    #sample from pi(x,y)
    x =  np.random.normal(size = sizes[i],loc = 2, scale = 1)
    y =  np.random.normal(size = sizes[i],loc = 2, scale = 1)
    theta1[i] = np.mean(np.sqrt(x**2+y**2))
    
    #sample from g(x,y) with sd = 1
    x =  np.random.normal(size = sizes[i],loc = 0, scale = 1)
    y =  np.random.normal(size = sizes[i],loc = 0, scale = 1)
    w2 = pif(x,y)/gif(x,y,1)
    theta2[i] = np.mean(np.sqrt(x**2+y**2)*w2)
    ess2[i] = sizes[i]/(1+np.var(w2,ddof=1))
    #sample from g(x,y) with sd = 4
    x =  np.random.normal(size = sizes[i],loc = 0, scale = 4)
    y =  np.random.normal(size = sizes[i],loc = 0, scale = 4)
    w3 = pif(x,y)/gif(x,y,4)
    theta3[i] = np.mean(np.sqrt(x**2+y**2)*w3)
    ess3[i] = sizes[i]/(1+np.var(w3,ddof=1))
# a) Plot the alternatives against log scaled sample size
fig,ax = plt.subplots(figsize = (8,5))
plt.plot(sizes, theta1, color='blue')
plt.plot(sizes, theta2, color='red')
plt.plot(sizes, theta3, color='green')
plt.xscale('log')
plt.grid()
plt.xlabel('n samples')
plt.ylabel(r'$\hat\theta$')
plt.title(r'$\hat\theta$ over n samples')
plt.legend([r'$\hat\theta_1$', r'$\hat\theta_2$', r'$\hat\theta_3$'])
fig.savefig('1a.png', dpi=300)
plt.show()

# b) True vs. Estimated Effective Sample Sizes
# We define vals this way so that we have many iterations for smaller sample sizes
# but less interations towards the end. This is because as sample size gets very large
# the error will not change as much, so it is unnecessary to run through small increments
# of sample size.
vals = list(range(1, int(10e3))) + list(range(int(10e3), int(10e5), int(10e3)))

#The following 3 variables hold mean error for each sample size
error1 = np.zeros(len(vals))
error2 = np.zeros(len(sizes))
error3 = np.zeros(len(sizes))
theta = theta1[-1] #Suppost this is the true value of the integral

#Find mean error for theta_1
for i in range(len(vals)):
    error = [] # Hold error for each run
    for j in range(50): # We only run 50 times for each sample size
        x =  np.random.normal(size = vals[i],loc = 2, scale = 1)
        y =  np.random.normal(size = vals[i],loc = 2, scale = 1)
        error.append(abs(np.mean(np.sqrt(x**2+y**2))-theta))
    error1[i] = np.mean(error)

#Find mean error for theta_2 and theta_3
for i in range(n_sizes): # We don't use the sample sizes vals here
    #The following two variables hold error for each run
    error_2 = []
    error_3 = []
    for j in range(100): # We run 100 times for each sample size
    #sample from g(x,y) with sd = 1
        x =  np.random.normal(size = sizes[i],loc = 0, scale = 1)
        y =  np.random.normal(size = sizes[i],loc = 0, scale = 1)
        error_2.append(abs(np.mean(np.sqrt(x**2+y**2)*pif(x,y)/gif(x,y,1))-theta))
    #sample from g(x,y) with sd = 4
        x =  np.random.normal(size = sizes[i],loc = 0, scale = 4)
        y =  np.random.normal(size = sizes[i],loc = 0, scale = 4)
        error_3.append(abs(np.mean(np.sqrt(x**2+y**2)*pif(x,y)/gif(x,y,4))-theta))
    error2[i]= np.mean(error_2)
    error3[i]= np.mean(error_3)    
    
#Find true ess for theta_2 and theta_3
ess2_star = np.zeros(len(sizes))
ess3_star = np.zeros(len(sizes))

for i in range(len(sizes)):
# The variable index is the index of the smallest sample size using the original distribution to get 
# the same error if we use g(x,y), for each sample size of theta_2, theta_3
    index = np.min(np.where(error1 <= error2[i])) 
    ess2_star[i] = vals[index]
    index = np.min(np.where(error1 <= error3[i])) 
    ess3_star[i] = vals[index]
fig,ax = plt.subplots(figsize = (8,5))
plt.plot(sizes, ess2_star)
plt.plot(sizes, ess2)
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.xlabel('n samples')
plt.ylabel('Estimated Effective Size')
plt.title(r'Estimated Effective Size of $\hat\theta_2$ over n samples')
plt.legend([r'$ess_2^*$',r'$ess_2$'])
fig.savefig('1b1.png', dpi=300)
plt.show()

fig,ax = plt.subplots(figsize = (8,5))
plt.plot(sizes, ess3_star)
plt.plot(sizes, ess3)
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.xlabel('n samples')
plt.ylabel('Estimated Effective Size')
plt.title(r'Estimated Effective Size of $\hat\theta_3$ over n samples')
plt.legend([r'$ess_3^*$',r'$ess_3$'])
fig.savefig('1b2.png', dpi=300)
plt.show()

fig,ax = plt.subplots(figsize = (8,5))
indices = np.argsort(ess2_star)
plt.plot(ess2_star[indices],ess2[indices])
plt.grid()
plt.xlabel(r'$ess_2$')
plt.ylabel(r'$ess_2^*$')
plt.title(r'$ess_2^*$ against $ess_2$')
fig.savefig('1b3.png', dpi=300)
plt.show()

fig,ax = plt.subplots(figsize = (8,5))
indices = np.argsort(ess3_star)
plt.plot(ess3_star[indices],ess3[indices])
plt.grid()
plt.xlabel(r'$ess_3$')
plt.ylabel(r'$ess_3^*$')
plt.title(r'$ess_3^*$ against $ess_3$')
fig.savefig('1b4.png', dpi=300)
plt.show()