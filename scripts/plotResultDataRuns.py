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

def loadControl(fileName):
    file = open(fileName, 'rb')
    control = pickle.load(file)
    file.close()
    return control

def get_color_from_colormap(value, colormap_name='viridis'):
    colormap = cm.get_cmap(colormap_name)
    color = colormap(value)
    return color


baseLine = [ './../runControl-0.88192126_u0/stress_v9.pickle', './../runControl-0.88192126_u1/stress_v9.pickle', './../runControl-0.88192126_u2/stress_v9.pickle', './../runControl-0.88192126_u3/stress_v9.pickle', './../runControl-0.88192126_u4/stress_v9.pickle']
baseLineV = [ './../runControl-0.88192126_u0/fieldV_v9.pickle', './../runControl-0.88192126_u1/fieldV_v9.pickle', './../runControl-0.88192126_u2/fieldV_v9.pickle', './../runControl-0.88192126_u3/fieldV_v9.pickle', './../runControl-0.88192126_u4/fieldV_v9.pickle']

#ycoord = -0.9988
ycoord = -0.83146961
legend = False

files = [ 
        [ './../runControl-0.88192126_u0/stress_v7.pickle', './../runControl-0.88192126_u1/stress_v7.pickle', './../runControl-0.88192126_u2/stress_v7.pickle', './../runControl-0.88192126_u3/stress_v7.pickle', './../runControl-0.88192126_u4/stress_v7.pickle'], 
        [ f'./../_korali_vracer_multi_{ycoord}_1/sample0/stress_train_r0.pickle', f'./../_korali_vracer_multi_{ycoord}_1/sample1/stress_train_r1.pickle', f'./../_korali_vracer_multi_{ycoord}_1/sample2/stress_train_r2.pickle', f'./../_korali_vracer_multi_{ycoord}_1/sample3/stress_train_r3.pickle', f'./../_korali_vracer_multi_{ycoord}_1/sample4/stress_train_r4.pickle'],
        [ f'./../_korali_vracer_multi_{ycoord}_2/sample0/stress_train_r0.pickle', f'./../_korali_vracer_multi_{ycoord}_2/sample1/stress_train_r1.pickle', f'./../_korali_vracer_multi_{ycoord}_2/sample2/stress_train_r2.pickle', f'./../_korali_vracer_multi_{ycoord}_2/sample3/stress_train_r3.pickle', f'./../_korali_vracer_multi_{ycoord}_2/sample4/stress_train_r4.pickle'],
        [ f'./../_korali_vracer_multi_{ycoord}_3/sample0/stress_train_r0.pickle', f'./../_korali_vracer_multi_{ycoord}_3/sample1/stress_train_r1.pickle', f'./../_korali_vracer_multi_{ycoord}_3/sample2/stress_train_r2.pickle', f'./../_korali_vracer_multi_{ycoord}_3/sample3/stress_train_r3.pickle', f'./../_korali_vracer_multi_{ycoord}_3/sample4/stress_train_r4.pickle'],
#        [ './../_korali_vracer_multi_{ycoord}_4a/sample0/stress_train_r0.pickle', './../_korali_vracer_multi_{ycoord}_4a/sample1/stress_train_r1.pickle', './../_korali_vracer_multi_{ycoord}_4a/sample2/stress_train_r2.pickle', './../_korali_vracer_multi_{ycoord}_4a/sample3/stress_train_r3.pickle', './../_korali_vracer_multi_{ycoord}_4a/sample4/stress_train_r4.pickle'],
        [ f'./../_korali_vracer_multi_{ycoord}_5/sample0/stress_train_r0.pickle', f'./../_korali_vracer_multi_{ycoord}_5/sample1/stress_train_r1.pickle', f'./../_korali_vracer_multi_{ycoord}_5/sample2/stress_train_r2.pickle', f'./../_korali_vracer_multi_{ycoord}_5/sample3/stress_train_r3.pickle', f'./../_korali_vracer_multi_{ycoord}_5/sample4/stress_train_r4.pickle'] 
        ]

filesControl = [ 
        [ './../runControl-0.88192126_u0/control_v7.pickle', './../runControl-0.88192126_u1/control_v7.pickle', './../runControl-0.88192126_u2/control_v7.pickle', './../runControl-0.88192126_u3/control_v7.pickle', './../runControl-0.88192126_u4/control_v7.pickle'], 
        [ f'./../_korali_vracer_multi_{ycoord}_1/sample0/control_train_r0.pickle', f'./../_korali_vracer_multi_{ycoord}_1/sample1/control_train_r1.pickle', f'./../_korali_vracer_multi_{ycoord}_1/sample2/control_train_r2.pickle', f'./../_korali_vracer_multi_{ycoord}_1/sample3/control_train_r3.pickle', f'./../_korali_vracer_multi_{ycoord}_1/sample4/control_train_r4.pickle'],
        [ f'./../_korali_vracer_multi_{ycoord}_2/sample0/control_train_r0.pickle', f'./../_korali_vracer_multi_{ycoord}_2/sample1/control_train_r1.pickle', f'./../_korali_vracer_multi_{ycoord}_2/sample2/control_train_r2.pickle', f'./../_korali_vracer_multi_{ycoord}_2/sample3/control_train_r3.pickle', f'./../_korali_vracer_multi_{ycoord}_2/sample4/control_train_r4.pickle'],
        [ f'./../_korali_vracer_multi_{ycoord}_3/sample0/control_train_r0.pickle', f'./../_korali_vracer_multi_{ycoord}_3/sample1/control_train_r1.pickle', f'./../_korali_vracer_multi_{ycoord}_3/sample2/control_train_r2.pickle', f'./../_korali_vracer_multi_{ycoord}_3/sample3/control_train_r3.pickle', f'./../_korali_vracer_multi_{ycoord}_3/sample4/control_train_r4.pickle'],
#        [ './../_korali_vracer_multi_{ycoord}_4a/sample0/control_train_r0.pickle', './../_korali_vracer_multi_{ycoord}_4a/sample1/control_train_r1.pickle', './../_korali_vracer_multi_{ycoord}_4a/sample2/control_train_r2.pickle', './../_korali_vracer_multi_{ycoord}_4a/sample3/control_train_r3.pickle', './../_korali_vracer_multi_{ycoord}_4a/sample4/control_train_r4.pickle'],
        [ f'./../_korali_vracer_multi_{ycoord}_5/sample0/control_train_r0.pickle', f'./../_korali_vracer_multi_{ycoord}_5/sample1/control_train_r1.pickle', f'./../_korali_vracer_multi_{ycoord}_5/sample2/control_train_r2.pickle', f'./../_korali_vracer_multi_{ycoord}_5/sample3/control_train_r3.pickle', f'./../_korali_vracer_multi_{ycoord}_5/sample4/control_train_r4.pickle'] 
        ]


labels = [ 'Opposition Control', 'DRL 1 Agent', 'DRL 2 Agents', 'DRL 4 Agents', 'DRL 8 Agents' ]
    
nx = 16
ny = 16
numsteps = 1000

if __name__ == "__main__":

    baseStress = np.zeros((len(baseLine), numsteps))
    for idx, f in enumerate(baseLine):
        baseStress[idx,:] = loadStress(f)

    baseV = np.zeros((len(baseLineV), numsteps, nx, ny))
    for idx, f in enumerate(baseLineV):
        baseV[idx,:, :, :] = loadControl(f)

    fName = f'stressResults.pdf'
    fig, ax = plt.subplots(1,1)

    for fs in files:
        print(fs)
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
    plt.close("all")

    ###########################################################################
    if legend:
        fName = f'dragReductionResults_{ycoord}.pdf'
    else:
        fName = f'dragReductionResultsNoLeg_{ycoord}.pdf'

    fig, ax = plt.subplots(1,1)

    for fidx, fs in enumerate(files):
        print(fs)
        reduction = np.zeros((len(fs),numsteps))
        for idx, f in enumerate(fs):
            stress = loadStress(f)
            reduction [idx,:] = 100.*(1.-stress/baseStress[idx,:])

        meanReduction = np.mean(reduction,axis=0) 
        stdReduction = np.std(reduction,axis=0) 

        ax.plot(np.arange(numsteps), meanReduction, linestyle='-', lw=1) #, color='turquoise')
        ax.fill_between(np.arange(numsteps), meanReduction+stdReduction, meanReduction-stdReduction,alpha=0.2, label=labels[fidx])
    
    ax.set_xticks(np.linspace(0,numsteps,5)) 
    ax.set_ylim([-75.,75.])
    #ax.set_aspect('box')
    #ax.set_box_aspect(1)
    if legend:
        ax.legend()
    #plt.axis("square")

    plt.tight_layout()
    plt.savefig(fName)
    plt.close("all") 

    ###########################################################################
    fig, ax = plt.subplots(1,1)

    for fidx, fs in enumerate(files):
        print(fs)
        reduction = np.zeros((len(fs),numsteps))
        for idx, f in enumerate(fs):
            control = loadControl(f)
            controlA = control[:,0,0]
            controlB = control[:,5,5]
            controlC = control[:,10,10]
            controlD = control[:,15,15]

            fName = f'control_s{idx}_v{fidx}.pdf'
            fig, ax = plt.subplots(4,1)
            ax[0].plot(np.arange(numsteps), controlA, linestyle='-', lw=1, color='mediumturquoise')
            ax[0].set_xticks(np.arange(0,numsteps,500)) 
            ax[0].set_xticklabels([]) 
            #ax[0].set_ylim([-0.1,0.1]) 

            ax[1].plot(np.arange(numsteps), controlB, linestyle='-', lw=1, color='mediumturquoise')
            ax[1].set_xticks(np.arange(0,numsteps,500)) 
            ax[1].set_xticklabels([]) 
            #ax[1].set_ylim([-0.1,0.1]) 

            ax[2].plot(np.arange(numsteps), controlC, linestyle='-', lw=1, color='mediumturquoise')
            ax[2].set_xticks(np.arange(0,numsteps,500)) 
            ax[2].set_xticklabels([]) 
            #ax[1].set_ylim([-0.1,0.1]) 

            ax[3].plot(np.arange(numsteps), controlC, linestyle='-', lw=1, color='mediumturquoise')
            ax[3].set_xticks(np.arange(0,numsteps,500)) 
            ax[3].set_xticklabels([]) 
            #ax[1].set_ylim([-0.1,0.1]) 

            plt.tight_layout()
            plt.savefig(fName)
            print(f'figure {fName} saved!')
            plt.close("all") 
