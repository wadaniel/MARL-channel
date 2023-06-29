import numpy as np
import os
import shutil
import pickle
from helpers import fieldToState, calcControl, distribute_field
import matplotlib
matplotlib.use('Agg')
import time
import matplotlib.pyplot as plt
import seaborn as sns
#sns.color_palette("vlag", as_cmap=True)
#sns.set_palette("vlag")

from scipy.stats import norm

srcDir = './../bin/'
workDir = './../data3/'
maxProc = 1

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

maxSteps = 100 #1000
saveFrqncy = 100 #1000
Ny = maxSteps #500 

stepfac = 1
maxv = 0.04285714285714286

if __name__ == "__main__":


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

    uFlat = np.reshape(allDataUplane,(-1,nz*nx))
    vFlat = np.reshape(allDataVplane,(-1,nz*nx))

    
    allDataControl = allDataControl.flatten()
    allDataU = allDataUplane.flatten()
    allDataV = allDataVplane.flatten()

    allDataU = allDataU[:100*256]
    allDataV = allDataV[:100*256]
    #allDataU = allDataU[:10]
    #allDataV = allDataV[:10]

    print(allDataU.shape)
    print(allDataV.shape)
    stdU = np.std(allDataU)
    stdV = np.std(allDataV)


    for k in range(11):
    #for k in range(1):

        Ny = 2**k
        Na = len(allDataU)

        allYU = allDataU + np.random.normal(0, stdU, size=(Ny,Na))
        allYV = allDataU + np.random.normal(0, stdU, size=(Ny,Na))

        print(allYU.shape)
        print(allYV.shape)


        start = time.time()


        utility = 0.

        for j in range(Ny):
            for i in range(Na):
                
                utility += norm.logpdf(allYU[j,i], allDataU[i], stdU)

                tmp = norm.pdf(allYU[j,i], allDataU[:], stdU)
                #print(tmp.shape)
                #print(tmp)
                innerSum = np.mean(norm.pdf(allYU[j,i], allDataU[:], stdU))
                #print(innerSum)
                utility -= np.log(innerSum)


        utility /= (Ny*Na)

        end = time.time()
        print(f'Utility: {utility} (Na={Na}, Ny={Ny})')
        print(f'Took: {end-start}s')

        start = time.time()

        utility = 0.

        for j in range(Ny):
            
            utility += np.sum(norm.logpdf(allYU[j,:], allDataU[:], stdU))

            pdf = lambda mu : norm.pdf(allYU[j,:], mu, stdU)
            tmp = np.array([pdf(xi) for xi in allDataU])
            utility -= np.sum(np.log(np.mean(tmp,axis=0)))


        utility /= (Ny*Na)

        end = time.time()
        print(f'Utility: {utility} (Na={Na}, Ny={Ny})')
        print(f'Took: {end-start}s')


