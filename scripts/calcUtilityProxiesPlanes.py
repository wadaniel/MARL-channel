
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

    with open(f'{workDir}/control.pickle', 'rb') as f:
        allDataControl = pickle.load(f)

    with open(f'{workDir}/fieldU.pickle', 'rb') as f:
        allDataUplane = pickle.load(f)

    with open(f'{workDir}/fieldV.pickle', 'rb') as f:
        allDataVplane = pickle.load(f)

    with open(f'{workDir}/covU.pickle', 'rb') as f:
        uCov = pickle.load(f)

    with open(f'{workDir}/covV.pickle', 'rb') as f:
        vCov = pickle.load(f)

    with open(f'{workDir}/dataU.pickle', 'rb') as f:
        yU = pickle.load(f)

    with open(f'{workDir}/dataV.pickle', 'rb') as f:
        yV = pickle.load(f)


    uFlat = np.reshape(allDataUplane[1:,:,:],(-1,nz*nx))
    vFlat = np.reshape(allDataVplane[1:,:,:],(-1,nz*nx))

    
    allDataControl = allDataControl[:-1,:,:].flatten()
    allDataU = allDataUplane.flatten()
    allDataV = allDataVplane.flatten()

    print(f'data loaded in workdir {workDir}')

    start = time.time()
    
    utilityProxyU1 = np.std(allDataU)
    utilityProxyV1 = np.std(allDataV)

    utilityProxyU2 = np.std(allDataU)/np.abs(np.mean(allDataU))
    utilityProxyV2 = np.std(allDataV)/np.abs(np.mean(allDataV))

    end = time.time()
    print(f'Workdir: {workDir}')
    print(f'Utility Proxy U 1: {utilityProxyU1}')
    print(f'Utility Proxy V 2: {utilityProxyV1}')
    print(f'Utility Proxy U 1: {utilityProxyU2}')
    print(f'Utility Proxy V 2: {utilityProxyV2}')
    print(f'Took: {end-start}s')


if __name__ == "__main__":

    allWorkDirs = [d for d in os.listdir('./../') if os.path.isdir(f'./../{d}')]
    allPlaneDirs = [f'./../{d}/' for d in allWorkDirs if 'data_h' in d]

    print(f'Processing {allPlaneDirs}')
    for d in allPlaneDirs:
        calcUtilityProxies(d)

    #with Pool(8) as p:
    #    p.map(calcUtility, allPlaneDirs)
