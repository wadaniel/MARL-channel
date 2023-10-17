import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from helpers import versionNames

import numpy as np

#workDir = './../runOpposition/'
#workDir = './../runOpposition3000_10/'
#workDir = './../runOpposition3000_10p/'
#workDir = './../runOpposition3000_10tm/'
workDir = './../runControl/'

def loadData(workdir, version):
    file = open(f'{workDir}control_v{version}.pickle', 'rb')
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

def loadStress(fileName):
    file = open(fileName, 'rb')
    allDataStress = pickle.load(file)
    file.close()
    stress = np.mean(allDataStress, axis=(1,2))
    return stress


def get_color_from_colormap(value, colormap_name='viridis'):
    colormap = cm.get_cmap(colormap_name)
    color = colormap(value)
    return color


#files = [ ['./../runControl0/stress_v9.pickle', './../runControl1/stress_v9.pickle', './../runControl2/stress_v9.pickle', './../runControl3/stress_v9.pickle', './../runControl4/stress_v9.pickle'],
#        ['./../runControl0/stress_v7.pickle', './../runControl1/stress_v7.pickle', './../runControl2/stress_v7.pickle', './../runControl3/stress_v7.pickle', './../runControl4/stress_v7.pickle'],
#        ['./../_korali_vracer_multi_2/sample0/stress_r0.pickle', './../_korali_vracer_multi_2/sample1/stress_r1.pickle', './../_korali_vracer_multi_2/sample2/stress_r2.pickle', './../_korali_vracer_multi_2/sample3/stress_r3.pickle', './../_korali_vracer_multi_2/sample4/stress_r4.pickle'],
#        ['./../_korali_vracer_multi_3/sample0/stress_r0.pickle', './../_korali_vracer_multi_3/sample1/stress_r1.pickle', './../_korali_vracer_multi_3/sample2/stress_r2.pickle', './../_korali_vracer_multi_3/sample3/stress_r3.pickle', './../_korali_vracer_multi_3/sample4/stress_r4.pickle'],
#        ['./../_falcon_korali_vracer_multi_2/sample0/stress_r0.pickle', './../_falcon_korali_vracer_multi_2/sample1/stress_r1.pickle', './../_falcon_korali_vracer_multi_2/sample2/stress_r2.pickle', './../_falcon_korali_vracer_multi_2/sample3/stress_r3.pickle', './../_falcon_korali_vracer_multi_2/sample4/stress_r4.pickle'],
#        ['./../_falcon_korali_vracer_multi_3/sample0/stress_r0.pickle', './../_falcon_korali_vracer_multi_3/sample1/stress_r1.pickle', './../_falcon_korali_vracer_multi_3/sample2/stress_r2.pickle', './../_falcon_korali_vracer_multi_3/sample3/stress_r3.pickle', './../_falcon_korali_vracer_multi_3/sample4/stress_r4.pickle'] ]


files = [ ['./../_korali_vracer_multi_-0.9988_1/sample0/stress_r0.pickle', './../_korali_vracer_multi_-0.9988_1/sample2/stress_r2.pickle' ], ['./../_korali_vracer_multi_-0.9988_2/sample0/stress_r0.pickle', './../_korali_vracer_multi_-0.9988_2/sample2/stress_r2.pickle' ]

numsteps = 1000

if __name__ == "__main__":

    fName = f'stress.png'
    fig, ax = plt.subplots(1,1)

    for fs in files:
        stress = np.zeros((len(fs),numsteps))
        for idx, f in enumerate(fs):
            stress[idx,:] = loadStress(f)

        meanStress = np.mean(stress,axis=0)
        stdStress = np.std(stress,axis=0)


        ax.plot(np.arange(numsteps), meanStress, linestyle='-', lw=1) #, color='turquoise')
        ax.fill_between(np.arange(numsteps), meanStress+stdStress, meanStress-stdStress,alpha=0.2)
        ax.set_xticks(np.linspace(0,numsteps,5)) 
        ax.set_ylim([0.,7.]) 


    plt.tight_layout()
    plt.savefig(fName)
