
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

def calcUtility(workDir):

    if os.path.isfile(f'{workDir}/control.pickle') == False:
        print(f'skipping dir {workDir}, control.pickle doesnt exist')


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


    #uFlat = np.reshape(allDataUplane[:,:,:],(-1,nz*nx)) # unrolls inner plane to 2d array
    #vFlat = np.reshape(allDataVplane[:,:,:],(-1,nz*nx))
    
    allDataControl = allDataControl[:,:,:].flatten() # unrolls planes to 1d array
    allDataU = allDataUplane.flatten()
    allDataV = allDataVplane.flatten()

    nstep = 100
    ndat = nstep*nx*nz
    allDataU = allDataU[:ndat] 
    allDataV = allDataV[:ndat] 
    allDataControl = allDataControl[:ndat] 
 
    stdU = 0.2 #np.std(allDataU)/3
    stdV = 0.03 #np.std(allDataV)/3

    print(f'data loaded in workdir {workDir}')
    print(f'mean and sdev U')
    print(np.mean(allDataU))
    print(np.std(allDataU))
    print(f'mean and sdev V')
    print(np.mean(allDataV))
    print(np.std(allDataV))

    Na = len(allDataU)
    Ny = 16

    allYU = allDataU + np.random.normal(0, stdU, size=(Ny,Na))
    allYV = allDataU + np.random.normal(0, stdU, size=(Ny,Na))
 
    print(allYU.shape)
    print(allYV.shape)

    start = time.time()
    
    """
    utility = 0.
    utility2 = 0.

    for j in range(Ny):
        for i in range(Na):
            
            utmp = norm.logpdf(allYU[j,i], allDataU[i], stdU)
            innerSum = np.mean(norm.pdf(allYU[j,i], allDataU[:], stdU))
            utmp -= np.log(innerSum)

        utility += utmp
        utility2 += utmp**2
    """

    utility = 0.
    utility2 = 0.

    vtility = 0.
    vtility2 = 0.

    for j in range(Ny):
        
        utmp = np.sum(norm.logpdf(allYU[j,:], allDataU[:], stdU))
        vtmp = np.sum(norm.logpdf(allYV[j,:], allDataV[:], stdV))

        pdfu = lambda mu : norm.pdf(allYU[j,:], mu, stdU)
        pdfv = lambda mu : norm.pdf(allYV[j,:], mu, stdV)

        tmp1 = np.array([pdfu(xi) for xi in allDataU])
        tmp2 = np.array([pdfv(xi) for xi in allDataV])

        utmp -= np.sum(np.log(np.mean(tmp1,axis=0)))
        vtmp -= np.sum(np.log(np.mean(tmp2,axis=0)))

        utility += utmp
        utility2 += utmp**2

        vtility += vtmp
        vtility2 += vtmp**2

    utility /= (Ny*Na)
    utility2 /= (Ny*Na)
    utilitySdev = np.sqrt(utility2 - utility**2)

    vtility /= (Ny*Na)
    vtility2 /= (Ny*Na)
    vtilitySdev = np.sqrt(vtility2 - vtility**2)

    end = time.time()
    print(f'Workdir: {workDir}')
    print(f'Utility (sdev): {utility} ({utilitySdev}) ({workDir})')
    print(f'Vtility (sdev): {vtility} ({vtilitySdev}) ({workDir})')
    print(f'Took: {end-start}s (Na={Na}, Ny={Ny})')

    f = open(f'{workDir}/result.txt', 'a')
    f.write(f'Utility\n')
    f.write(f'{utility}\n')
    f.write(f'{utilitySdev}\n')
    f.write(f'{vtility}\n')
    f.write(f'{vtilitySdev}\n')
    f.write(f'{stdU}\n')
    f.write(f'{stdV}\n')
    f.write(f'{Na}\n')
    f.write(f'{Ny}\n')
    f.close()

if __name__ == "__main__":

    wdir = 'data1_h'
    allWorkDirs = [d for d in os.listdir('./../') if os.path.isdir(f'./../{d}')]
    allPlaneDirs = [f'./../{d}/' for d in allWorkDirs if wdir in d]

    print(f'Processing {allPlaneDirs}')
    #for d in allPlaneDirs:
    #    calcUtility(d)

    nproc = (len(allPlaneDirs)+1)//2
    with Pool(nproc) as p:
        p.map(calcUtility, allPlaneDirs)
