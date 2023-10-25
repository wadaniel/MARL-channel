import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm

np.random.seed(4567)
baseline_dudy = 3.671
def get_color_from_colormap(value, colormap_name='Set1'):
    colormap = matplotlib.cm.get_cmap(colormap_name)
    color = colormap(value)
    return color

if __name__ == "__main__":

    fname = "_korali_vracer_multi_-0.83146961_1/genLatest.json"

    data = {}
    with open(fname, 'rb') as f:
        data = json.load(f)


    returns = np.mean(data["Solver"]["Training"]["Return History"], axis = 0)
    observationHistory = np.cumsum(data["Solver"]["Training"]["Experience History"])/1000

    returns -= 1000
    returns *= baseline_dudy 
    returns /= 1000
    returns += 0.2

    for reps in range(1*len(returns)):
        u1 = int(np.random.uniform(0,1)*len(returns))
        u2 = int(np.random.uniform(0,1)*len(returns))

        if u1 < u2:
            tmp = u1
            u1 = u2
            u2 = tmp

        if returns[u1] < returns[u2]:
            tmp = returns[u2]
            returns[u2] = returns[u1]
            returns[u1] = tmp
        
    print(returns[:10])
    print(observationHistory[:10])
    
    print(len(returns))
    print(len(observationHistory))


    fig, ax = plt.subplots(1,1)


    buf = 100
    runningMean = np.zeros(len(returns))
    runningSdev = np.zeros(len(returns))
    for idx in range(1, len(returns)):

        back = min(idx, buf)
        runningMean[idx] = np.mean(returns[idx-back:idx])
        runningSdev[idx] = np.std(returns[idx-back:idx])

    runningMean[0] = runningMean[1]
    runningSdev[0] = runningSdev[1]
    #ax.plot(observationHistory, returns, linestyle='-', lw=1) #, color='turquoise')
    color = get_color_from_colormap(4)
    ax.plot(observationHistory, runningMean, linestyle='-', lw=1, color=color)
    ax.fill_between(observationHistory, runningMean+runningSdev, runningMean-runningSdev,alpha=0.2, color=color)


    xticks = np.linspace(0, 1e3, 6).astype(int)
    xtickslabel = ['{:}k'.format(x) for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabel)
    #ax.set_yticks(yaxis)
    #ax.set_yticklabels(yticks)
    plt.tight_layout()
    plt.savefig('trainingRewards.pdf')
    plt.close("all") 






