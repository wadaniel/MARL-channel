import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

workDir = './../runOpposition/'

def loadData(workdir, version):
    file = open(f'{workDir}/control_v{version}.pickle', 'rb')
    allDataControl = pickle.load(file)
    file.close()
    controlA = allDataControl[:,0,0]
    controlB = allDataControl[:,5,5]
    controlC = allDataControl[:,10,10]
    controlD = allDataControl[:,15,15]

    file = open(f'{workDir}/stress_v{version}.pickle', 'rb')
    allDataStress = pickle.load(file)
    file.close()
    stress = np.mean(allDataStress, axis=(1,2))
    
    return controlA, controlB, controlC, controlD, stress


if __name__ == "__main__":

    versions = [0,1,2,3,4,5,6,7,8]
    stresses = []


    for v in versions:
        controlA, controlB, controlC, controlD, _ = loadData(workDir, v)
        
        T = len(controlA)
    
        fName = f'{workDir}/control_v{v}.png'
        fig, ax = plt.subplots(4,1)
        ax[0].plot(np.arange(T), controlA, linestyle='-', lw=1, color='turquoise')
        ax[0].set_xticks(np.arange(0,T,500)) 
        ax[0].set_xticklabels([]) 
        ax[0].set_ylim([-.05,0.05]) 

        ax[1].plot(np.arange(T), controlB, linestyle='-', lw=1, color='mediumturquoise')
        ax[1].set_xticks(np.arange(0,T,500)) 
        ax[1].set_xticklabels([]) 
        ax[1].set_ylim([-.05,0.05]) 

        ax[2].plot(np.arange(T), controlC, linestyle='-', lw=1, color='cadetblue')
        ax[2].set_xticks(np.arange(0,T,500)) 
        ax[2].set_xticklabels([]) 
        ax[2].set_ylim([-.05,0.05]) 

        ax[3].plot(np.arange(T), controlD, linestyle='-', lw=1, color='darkcyan')
        ax[3].set_xticks(np.arange(0,T,500)) 
        ax[3].set_xticklabels([]) 
        ax[3].set_ylim([-.05,0.05]) 

        plt.tight_layout()
        plt.savefig(fName)
        print(f'figure {fName} saved!')

    fName = f'{workdir}/stresses.pdf'
    fig, ax = plt.subplots(1,1)

    for v in versions:
        _, _, _, _, stress = loadData(workDir, v)
        ax.plot(np.arange(T), stress, linestyle='-', lw=1, label=f'v{v}')
 
    ax.legend()
    plt.tight_layout()
    plt.savefig(fName)
    print(f'figure {fName} saved!')
