import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice
def SAW_1 (n, x= None ,y = None, positions = None):
    '''
    Generate Simple Self Avoiding Walk in grid of size (n+1)x(n+1)
    '''   
    if x is None and y is None and positions is None:
        x = [0]
        y = [0]
        positions = set([(0,0)])
    step = 0                          # a set only contains distinct items
    pr = 1 # naive sampling distribution
    deltas = [(1,0), (0,1), (-1,0), (0,-1)] # possible directions for 2D lattice
    deltas_feasible = []  # deltas_feasible stores the available directions at each step t
    while True:      
        for dx, dy in deltas: 
            if x[-1] + dx > n or y[-1] + dy > n or x[-1] + dx < 0 or y[-1] + dy < 0:
                continue
            if ((x[-1] + dx, y[-1] + dy) not in positions) :  #c hecks if direction leads to a site not visited before
                deltas_feasible.append((dx,dy)) # if not visited, add to the feasible directions (this is self-avoiding steps)
              
        if deltas_feasible:  #checks if there is at least one direction available
            dx, dy = deltas_feasible[np.random.randint(0,len(deltas_feasible))]  # choose a direction at random among available ones
            pr *= 1/len(deltas_feasible)
            xnew, ynew = x[-1] + dx, y[-1] + dy
            positions.add((xnew, ynew))
            x.append(xnew)
            y.append(ynew)
            deltas_feasible = []
            step += 1
        else:
            break
    return x, y, pr, step
def SAW_2 (n):
    '''
    Generate Self Avoiding Walk that favors shorter walks in grid of size (n+1)x(n+1)
    '''
    x, y = [0], [0] # Start at (0,0)
    positions = set([(0,0)])  # positions is a set that stores all sites visited by the walk
    step = 0                          # a set only contains distinct items
    pr = 1 # naive sampling distribution
    deltas = [(1,0), (0,1), (-1,0), (0,-1)] # possible directions for 2D lattice
    deltas_feasible = []  # deltas_feasible stores the available directions at each step t
    esp = 0.1 # esp is the  probability that we terminate the walk early.
            # and the possible directions share 1-esp probability equally
    while True:      
        for dx, dy in deltas: 
            if x[-1] + dx > n or y[-1] + dy > n or x[-1] + dx < 0 or y[-1] + dy < 0:
                continue
            if ((x[-1] + dx, y[-1] + dy) not in positions) :  #c hecks if direction leads to a site not visited before
                deltas_feasible.append((dx,dy)) # if not visited, add to the feasible directions (this is self-avoiding steps)
              
        if deltas_feasible:  #checks if there is at least one direction available
            if step > 75:
                prob = np.array([(1-esp)/len(deltas_feasible),esp])
                index = choice([*range(len(deltas_feasible)),5],1,[*np.repeat(prob,[len(deltas_feasible),1])])[0] 
                if index == 5:
                    pr *= esp
                    return x, y, pr, step
                else :
                    dx, dy = deltas_feasible[index] 
                    pr *= (1-esp)/len(deltas_feasible)
                    xnew, ynew = x[-1] + dx, y[-1] + dy
                    positions.add((xnew, ynew))
                    x.append(xnew)
                    y.append(ynew)
                    deltas_feasible = []
                    step += 1
            else:
                dx, dy = deltas_feasible[np.random.randint(0,len(deltas_feasible))]  # choose a direction at random among available ones
                pr *= 1/len(deltas_feasible)
                xnew, ynew = x[-1] + dx, y[-1] + dy
                positions.add((xnew, ynew))
                x.append(xnew)
                y.append(ynew)
                deltas_feasible = []
                step += 1
                
        else:
            break
    return x, y, pr, step

def SAW_3 (n, currentM = 0):
    '''
    Generate Self Avoiding Walk that favors walks longer than 50 steps in grid of size (n+1)x(n+1), this function will 
    return 1/(p) instead of p because we have to add weights to the p obtained from the children
    of each walk that passes 50 steps, and returning 1/p makes it easier to compute K.
    '''
    x, y = [0], [0] # Start at (0,0)
    positions = set([(0,0)])  # positions is a set that stores all sites visited by the walk
    step = 0                          # a set only contains distinct items
    pr = 1 # naive sampling distribution
    cutpoint = 90 # where we start generating child walks
    deltas = [(1,0), (0,1), (-1,0), (0,-1)] # possible directions for 2D lattice
    deltas_feasible = []  # deltas_feasible stores the available directions at each step t
    walk_num = 0 # number of walks SAW_3 returns, excluding the parent walk, we account for the parent walk in compute_k function
    while True:      
        for dx, dy in deltas: 
            if x[-1] + dx > n or y[-1] + dy > n or x[-1] + dx < 0 or y[-1] + dy < 0:
                continue
            if ((x[-1] + dx, y[-1] + dy) not in positions) :  #c hecks if direction leads to a site not visited before
                deltas_feasible.append((dx,dy)) # if not visited, add to the feasible directions (this is self-avoiding steps)
              
        if deltas_feasible:  #checks if there is at least one direction available
           if step >= cutpoint and currentM < M:
                weight = np.array([1/pr])
                steps = [step]
                for _ in range(5):
                    updated_x = x.copy()
                    updated_y = y.copy()
                    x_temp,y_temp, pr_temp, step_temp = SAW_1(n,[x[-1]],[y[-1]], positions)
                    updated_x.extend(x_temp[1:])
                    updated_y.extend(y_temp[1:])
                    currentM += 1
                    walk_num += 1
                    weight = np.append(weight,pr*1/pr_temp/5)
                    steps.append(cutpoint + step_temp)
                    if len(updated_x) > len(x): #or we can use len(y) instead
                        x = updated_x #we only return the longest child walk
                        y = updated_y
                    if currentM == M:
                        break
                return x,y,[*weight], steps, walk_num
           else:
                dx, dy = deltas_feasible[np.random.randint(0,len(deltas_feasible))]  # choose a direction at random among available ones
                pr *= 1/len(deltas_feasible)
                xnew, ynew = x[-1] + dx, y[-1] + dy
                positions.add((xnew, ynew))
                x.append(xnew)
                y.append(ynew)
                deltas_feasible = []
                step += 1
        else:
            break
    return x, y, 1/pr, step, walk_num

def compute_K(M,n, method, end_at_nn = False):
    K = [0]
    x = []
    y = []
    steps = []
    weights = []
    if end_at_nn:
        for _ in range(M):
            xnew,ynew,pr,step = method(n)
            if (xnew[len(xnew)-1],ynew[len(ynew)-1]) == (n,n):
                K.append(K[-1] + 1/pr)
                weights.append(1/pr)
                steps.append(step)
                if step > len(x): # or len(y)
                    x = xnew
                    y = ynew
    else:
        if method == SAW_3:
            currentM = 0
            for _ in range(M):
                currentM+=1
                xnew,ynew,weight,step,walk_num = SAW_3(n,currentM)
                if type(weight) is list:
                    K.extend(weight)
                    weights.extend(weight)
                    steps.extend(step)
                else:
                    K.append(weight)
                    steps.append(step)
                    weights.append(weight)
                currentM += walk_num
                if np.max(step) > len(x): # or len(y)
                    x = xnew
                    y = ynew
                if currentM == M:
                    del K[0]
                    return x, y, steps, weights, np.cumsum(K)/range(1,len(K)+1)
        else:
            for _ in range(M):
                xnew,ynew,pr,step = method(n)
                K.append(K[-1] + 1/pr)
                steps.append(step)
                weights.append(1/pr)
                if step > len(x): # or len(y)
                    x = xnew
                    y = ynew
    del K[0]
    return x, y, steps, weights, np.array(K)/range(1,len(K)+1)
def plot_saw(x, y, design):
    """
    Plots the output of the myopic algorithm
    
    Args:
        n (int): the length of the walk
    Returns:
        Plot of the output of the myopic algorithm
    """
    fig,ax = plt.subplots(figsize = (5,5))
    plt.plot(x, y, 'bo-', linewidth = 1,alpha=0.9)
    plt.plot(0, 0, 'go', ms = 12, label = 'Start', alpha=0.5)
    plt.plot(x[-1], y[-1], 'ro', ms = 12, label = 'End', alpha=0.5)
    plt.axis('equal')
    ax.set_xticks(np.arange(0,11,1))
    ax.set_yticks(np.arange(0,11,1))
    plt.legend()
    ax.set_title('Longest Walk of Design {a}: {b} steps'.format(a = design,b = len(x)))
    ax.grid(which = 'both')
    fig.savefig('longest{a}.png'.format(a=design),dpi = 300)
    plt.show()
    
    
    
    
M = int(10e4)
n = 10


#Compute the estimated K with each design:
    
x1, y1, steps1, weight_1, K1 = compute_K(M,n, SAW_1)
x2, y2, steps2, weight_2 , K2 = compute_K(M,n, SAW_2)
x3, y3, steps3, weight_3, K3 = compute_K(M,n, SAW_3)
x1_nn, y1_nn, steps1_nn, weight_1_nn, K1_nn = compute_K(M,n, SAW_1,True)

# Plot the performance
fig,ax = plt.subplots(figsize = (8,5))
plt.plot(range(M), K1, color='blue')
plt.plot(range(M), K2, color='red')
plt.plot(range(M), K3, color='green')
plt.xscale('log')
plt.yscale('log')
ax.set_xlabel("M")
ax.set_ylabel("Estimated K")
ax.grid(b=True, which='major', linestyle='-')
plt.title('Estimated Number of Self-Avoiding Walks for M Samples')
plt.legend(['Design 1', 'Design 2', 'Design 3'])
fig.savefig('2a.png',dpi=300)
plt.show()

# Plot the weighted histograms of all 4 designs
fig,ax = plt.subplots(figsize = (8,5))
plt.hist(steps1, bins=50, weights=weight_1, alpha=0.5, label='Design 1', color='blue')
plt.hist(steps2, bins=50, weights=weight_2, alpha=0.5, label='Design 2', color='red')
plt.hist(steps3, bins=50, weights=weight_3, alpha=0.5, label='Design 3', color='green')
plt.hist(steps1_nn, bins=50, weights=weight_1_nn, alpha=0.5, label='Design 1', color='purple')
ax.set_xlabel("N-Length")
ax.set_ylabel("Density")
plt.title('Distribution of SAW lengths')
plt.legend(['Design 1', 'Design 2', 'Design 3','Design 4'])
fig.savefig('2c.png',dpi=300)
plt.show()

# Plot the longest walks of all 4 designs
plot_saw(x1, y1,1)
plot_saw(x2, y2,2)
plot_saw(x3, y3,3)
plot_saw(x1_nn, y1_nn,1)

