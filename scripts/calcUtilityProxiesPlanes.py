
import re
import os
import pickle
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool

import numpy as np
from scipy.stats import norm

nx = 16
ny = 65
nz = 16
npl = 3

h=1.
Lx = 2.67*h
Lz = 0.8*h


wbci = 7
nctrlx = 16
nctrlz = 16

dy = 1.-np.cos(np.pi*1/(ny-1)) 
ndrl = 28 #400 #80
nst = 4

retau = 180
nxs_ = 1
nzs_ = 1
xchange = 1
zchange = 1

version = 4
alpha = 1.0

maxv = 0.04285714285714286

def calcUtilityProxies(workDir):

    #if os.path.isfile(f'{workDir}/stress.pickle') == False:
    #    print(f'skipping dir {workDir}, stress.pickle doesnt exist')
    #    return

    with open(f'{workDir}/fieldU.pickle', 'rb') as f:
        allDataUplane = pickle.load(f)

    with open(f'{workDir}/fieldV.pickle', 'rb') as f:
        allDataVplane = pickle.load(f)

    allDataU = allDataUplane.flatten()
    allDataV = allDataVplane.flatten()

    print(f'data loaded in workdir {workDir}')

    start = time.time()
    
    utilityProxyU1 = np.std(allDataU)
    utilityProxyV1 = np.std(allDataV)

    utilityProxyU2 = np.std(allDataU)/np.abs(np.mean(allDataU))
    utilityProxyV2 = np.std(allDataV)/np.abs(np.mean(allDataV))

    utilityProxyU3 = np.random.normal(size=utilityProxyU1.shape)
    utilityProxyV3 = np.random.normal(size=utilityProxyV1.shape)

    end = time.time()
    print(f'Workdir: {workDir}')
    print(f'Utility Proxy U 1: {utilityProxyU1}')
    print(f'Utility Proxy V 2: {utilityProxyV1}')
    print(f'Utility Proxy U 1: {utilityProxyU2}')
    print(f'Utility Proxy V 2: {utilityProxyV2}')
    print(f'Took: {end-start}s')
    return np.array([[utilityProxyU1, utilityProxyU2, utilityProxyU3],[utilityProxyV1, utilityProxyV2, utilityProxyV3]])


if __name__ == "__main__":

    wdir = 'data_base_h'
    allWorkDirs = [d for d in os.listdir('./../') if os.path.isdir(f'./../{d}')]
    allPlaneDirs = [f'./../{d}/' for d in allWorkDirs if wdir in d]

    heights = [ float(re.findall("\d+\.\d+",d.replace(wdir,''))[0]) for d in allPlaneDirs ]
    print(heights)
    print(f'Processing {allPlaneDirs}')
    proxies = np.array([calcUtilityProxies(d) for d in allPlaneDirs])
    
    fName = f'{wdir}_proxy.png'

    fig, ax = plt.subplots(2,1)
    ax[0].plot(heights, proxies[:,0,0], linestyle='-', marker='x', color='lightsteelblue')
    ax[0].set_title("Statistics U")
    ax[0].set_xticks([]) 

    ax01 = ax[0].twinx()
    ax01.plot(heights, proxies[:,0,2], linestyle='-', marker='x', color='lightsteelblue')

    ax[1].plot(heights, proxies[:,1,0], linestyle='-', marker='x', color='lightsteelblue')
    ax[1].set_title("Statistics V")
    ax[1].set_xticks(np.arange(0,100,5)) 

    ax11 = ax[1].twinx()
    ax11.plot(heights, proxies[:,1,2], linestyle='-', marker='x', color='lightsteelblue')
   
    plt.tight_layout()
    plt.savefig(fName)
    plt.close('all')
    print(f'figure {fName} saved!')
