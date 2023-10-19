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


baseLine = [ './../runControl-0.88192126_u0/stress_v9.pickle', './../runControl-0.88192126_u1/stress_v9.pickle', './../runControl-0.88192126_u2/stress_v9.pickle', './../runControl-0.88192126_u3/stress_v9.pickle', './../runControl-0.88192126_u4/stress_v9.pickle']

files = [ 
        [ './../runControl-0.88192126_u0/stress_v7.pickle', './../runControl-0.88192126_u1/stress_v7.pickle', './../runControl-0.88192126_u2/stress_v7.pickle', './../runControl-0.88192126_u3/stress_v7.pickle', './../runControl-0.88192126_u4/stress_v7.pickle'], 
        [ './../_korali_vracer_multi_-0.9988_1/sample0/stress_train_r0.pickle', './../_korali_vracer_multi_-0.9988_1/sample1/stress_train_r1.pickle', './../_korali_vracer_multi_-0.9988_1/sample2/stress_train_r2.pickle', './../_korali_vracer_multi_-0.9988_1/sample3/stress_train_r3.pickle', './../_korali_vracer_multi_-0.9988_1/sample4/stress_train_r4.pickle'],
        [ './../_korali_vracer_multi_-0.9988_2/sample0/stress_train_r0.pickle', './../_korali_vracer_multi_-0.9988_2/sample1/stress_train_r1.pickle', './../_korali_vracer_multi_-0.9988_2/sample2/stress_train_r2.pickle', './../_korali_vracer_multi_-0.9988_2/sample3/stress_train_r3.pickle', './../_korali_vracer_multi_-0.9988_2/sample4/stress_train_r4.pickle'],
        [ './../_korali_vracer_multi_-0.9988_3/sample0/stress_train_r0.pickle', './../_korali_vracer_multi_-0.9988_3/sample1/stress_train_r1.pickle', './../_korali_vracer_multi_-0.9988_3/sample2/stress_train_r2.pickle', './../_korali_vracer_multi_-0.9988_3/sample3/stress_train_r3.pickle', './../_korali_vracer_multi_-0.9988_3/sample4/stress_train_r4.pickle'],
#        [ './../_korali_vracer_multi_-0.9988_4a/sample0/stress_train_r0.pickle', './../_korali_vracer_multi_-0.9988_4a/sample1/stress_train_r1.pickle', './../_korali_vracer_multi_-0.9988_4a/sample2/stress_train_r2.pickle', './../_korali_vracer_multi_-0.9988_4a/sample3/stress_train_r3.pickle', './../_korali_vracer_multi_-0.9988_4a/sample4/stress_train_r4.pickle'],
        [ './../_korali_vracer_multi_-0.9988_5/sample0/stress_train_r0.pickle', './../_korali_vracer_multi_-0.9988_5/sample1/stress_train_r1.pickle', './../_korali_vracer_multi_-0.9988_5/sample2/stress_train_r2.pickle', './../_korali_vracer_multi_-0.9988_5/sample3/stress_train_r3.pickle', './../_korali_vracer_multi_-0.9988_5/sample4/stress_train_r4.pickle'] 
        ]
    
    
numsteps = 1000

if __name__ == "__main__":

    baseStress = np.zeros((len(baseLine), numsteps))
    for idx, f in enumerate(baseLine):
        baseStress[idx,:] = loadStress(f)
        baseMeanStress = np.mean(baseStress,axis=0)
        baseStdStress = np.std(baseStress,axis=0)

    fName = f'stressResults.pdf'
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

    fName = f'dragReductionResults.pdf'
    fig, ax = plt.subplots(1,1)

    for fs in files:
        reduction = np.zeros((len(fs),numsteps))
        for idx, f in enumerate(fs):
            stress = loadStress(f)
            #reduction[idx,:] = stress
            reduction [idx,:] = 100.*(1.-stress/baseStress[idx,:])

        #meanReduction = 100.*np.mean(1.-reduction/baseStress,axis=0)
        meanReduction = np.mean(reduction,axis=0) #100.*np.mean(1.-reduction/baseStress,axis=0)
        #stdReduction = 100.*np.std(1.-reduction/baseMeanStress,axis=0)
        stdReduction = np.std(reduction,axis=0) #100.*np.std(1.-reduction/baseMeanStress,axis=0)

        print(meanReduction)
        ax.plot(np.arange(numsteps), meanReduction, linestyle='-', lw=1) #, color='turquoise')
        ax.fill_between(np.arange(numsteps), meanReduction+stdReduction, meanReduction-stdReduction,alpha=0.2)
        ax.set_xticks(np.linspace(0,numsteps,5)) 
        ax.set_ylim([-75.,75.]) 
        ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(fName)
