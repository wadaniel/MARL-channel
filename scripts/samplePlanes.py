import numpy as np
import os
import sys
import shutil
import pickle
from mpi4py import MPI
from helpers import fieldToState, calcControl, distribute_field, getHeights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
#sns.color_palette("vlag", as_cmap=True)
#sns.set_palette("vlag")

launchCommand = './bla_16x65x16_1'
#launchCommand = './bla_16x65x16_1_debug'
srcDir = './../bin/'
workDirTmp = './../dataNew_h'
maxProc = 1

requestState = b'STATE'
requestControl = b'CNTRL'
requestEvolve = b'EVOLV'
requestTerm = b'TERMN'

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

rew_mode = 'Instantaneous'
#rew_mode = 'MovingAverage'
partial_reward = False
agents = [f"jet_z{zi}_x{xi}" for xi in range(nctrlx) for zi in range(nctrlz)]

retau = 180
nxs_ = 1
nzs_ = 1
xchange = 1
zchange = 1

# Load the reference average du/dy at the wall to use as baseline (for wbci 6 or 7)
baseline_dudy_dict = {"180_16x65x16"   : 3.7398798426242075,
                      "180_32x33x32"   : 3.909412638928125,
                      "180_32x65x32"   : 3.7350180468974763,#
                      "180_64x65x64"   : 3.82829465265046,
                      "180_128x65x128" : 3.82829465265046}

baseline_dudy = baseline_dudy_dict[f"{int(retau)}_{nx}x{ny}x{nz}"]

seed = 1337
version = 0
alpha = 1.0

maxSteps = 1500
plotFrequency = 500
Ny = maxSteps #500 

stepfac = 1
maxv = 0.04285714285714286

def rollout(heightTuple):

    ycoords, height = heightTuple
    workDir = f"{workDirTmp}{height:.1f}/"

    allDataUplane = np.empty((maxSteps, nz, nx))
    allDataVplane = np.empty((maxSteps, nz, nx))
    allDataControl = np.empty((maxSteps, nz, nx))

    os.makedirs(workDir, exist_ok=True)
    os.system(f"sed 's/SAMPLINGHEIGHT/{ycoords}/' {srcDir}bla_macro.i > {workDir}/bla.i")
    shutil.copy(srcDir + "bla_16x65x16_1", workDir + "bla_16x65x16_1")

    print(f"Launching SIMSON from workdir {workDir}",flush=True)
    mpi_info = MPI.Info.Create()
    mpi_info.Set('wdir',workDir)
    mpi_info.Set('bind_to','none')
    subComm = MPI.COMM_SELF.Spawn(launchCommand,maxprocs=maxProc,info=mpi_info)

    #print("Python receiving state from Fortran", flush=True)
    subComm.Send([requestState, MPI.CHARACTER], dest=0, tag=maxProc+100)
    field = np.ndarray((npl-1,nz,nx),dtype=np.double)
    for plidx in range(1,npl):
        for zidx in range(nz):
            subComm.Recv([field[plidx-1,zidx,:], MPI.DOUBLE], source=0, tag=maxProc+10+plidx+zidx+1)

    #state = fieldToState(field)

    step = 0
    done = False
    stresses = []
    rewards = []

    # Receiving initial time of the simulation
    initTime = np.array(0,dtype=np.double)
    subComm.Recv([initTime, MPI.DOUBLE], source=0, tag=maxProc+960)
    currentTime = initTime

    while not done and step < maxSteps:

        control = calcControl(nctrlz, nctrlx, step//stepfac, maxv, version, seed)
        control -= np.mean(control)

        #print(control)
        #print("Python sending control to Fortran")
        subComm.Send([requestControl, MPI.CHARACTER], dest=0, tag=maxProc+100)
        if ((wbci == 6) or (wbci == 7)):
            for j in range(nctrlx):
                for i in range(nctrlz):
                    subComm.Send([control[i,j], MPI.DOUBLE], dest=0, tag=300+i+1+(j+1)*nctrlz)

        uxz = np.ndarray((nz, nx),dtype=np.double)
        uxzAvg = np.zeros((nz, nx),dtype=float)

        #print("Python sending evolve message to Fortran", flush=True)
        subComm.Send([requestEvolve, MPI.CHARACTER], dest=0, tag=maxProc+100)

        #print(f'Receiving the (non-averaged) rewards from Fortran', flush=True)
        i_evolv = 1
        while i_evolv <= (ndrl // nst)-1:
            for i in range(1):
                for zidx in range(nz):
                    subComm.Recv([uxz[zidx,:], MPI.DOUBLE], source=0, tag=maxProc+10+i+zidx+1)
            
            uxzAvg += uxz.astype(float)
            #uxzAvg = (uxzAvg*i_evolv + uxz.astype(float)/dy)/(i_evolv+1)
            i_evolv += 1

        uxzAvg /= (i_evolv*dy)

        # Distributing the field to compute the individual reward per each agent
        wallStresses = distribute_field(np.expand_dims(uxzAvg,0),agents,nctrlx,nctrlz,partial_reward,reward=True)

        if plotFrequency > 0 and step % plotFrequency == 0:
            fieldName = f"{workDir}uvplane_v{version}_s{step}.png"
            print(f"saving field {fieldName}")
            fig, ax = plt.subplots(1,2)
            x, z = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Lz, nz))
            c = ax[0].pcolormesh(z, x, field[1,:,:], cmap='RdBu', vmin=field[1,:,:].min(), vmax=field[1,:,:].max())
            ax[0].set_xticks(np.linspace(z.min(), z.max(), 3), minor=True)
            ax[0].set_yticks(np.linspace(x.min(), x.max(), 3), minor=True)
            ax[0].set_aspect('equal')
            ax[0].set_title('u')
            fig.colorbar(c, ax=ax[0])

            c = ax[1].pcolormesh(z, x, field[0,:,:], cmap='RdBu', vmin=field[0,:,:].min(), vmax=field[0,:,:].max())
            ax[1].set_xticks(np.linspace(z.min(), z.max(), 3), minor=True)
            ax[1].set_yticks([], minor=True)
            ax[1].set_yticklabels( () )
            ax[1].set_aspect('equal')
            ax[1].set_title('v')
            fig.colorbar(c, ax=ax[1])
            plt.tight_layout()
            plt.savefig(fieldName)

            cfieldName = f"{workDir}contol_v{version}_s{step}.png"
            print(f"saving control {cfieldName}")
            fig, ax = plt.subplots()
            x, z = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Lz, nz))
            c_min = control.min()
            c_max = control.max()
            c = ax.pcolormesh(z, x, control, cmap='RdBu', vmin=c_min, vmax=c_max)
            fig.colorbar(c, ax=ax)
            plt.axis('scaled')
            plt.xticks(np.linspace(z.min(), z.max(), 3))
            plt.yticks(np.linspace(x.min(), x.max(), 3))
            plt.savefig(cfieldName)

            wfieldName = f"{workDir}stress_v{version}_s{step}.png"
            print(f"saving stresses {wfieldName}")
            fig, ax = plt.subplots()
            x, z = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Lz, nz))
            c = ax.pcolormesh(z, x, uxzAvg, cmap='RdBu', vmin=uxzAvg.min(), vmax=uxzAvg.max())
            fig.colorbar(c, ax=ax)
            plt.axis('scaled')
            plt.xticks(np.linspace(z.min(), z.max(), 3))
            plt.yticks(np.linspace(x.min(), x.max(), 3))
            plt.savefig(wfieldName)

        # Computing wall stresses
        avgStress = 0.
        avgReward = 0.
        for ws in wallStresses.values():
            avgStress += np.mean(ws)
            avgReward += 1-np.mean(ws)/baseline_dudy
        avgStress /= len(agents)
        avgReward /= len(agents)
        stresses.append(avgStress)
        rewards.append(avgReward)

        #print("Python receiving state from Fortran", flush=True)
        subComm.Send([requestState, MPI.CHARACTER], dest=0, tag=maxProc+100)
        field = np.ndarray((npl-1,nz,nx),dtype=np.double)
        for plidx in range(1,npl):
            for zidx in range(nz):
                subComm.Recv([field[plidx-1,zidx,:], MPI.DOUBLE], source=0, tag=maxProc+10+plidx+zidx+1)

        #state = fieldToState(field)
        prevTime = currentTime
        currentTime = np.array(0,dtype=np.double)
        subComm.Recv([currentTime, MPI.DOUBLE], source=0, tag=maxProc+960)

        # store data
        allDataControl[step, :, :] = control
        allDataVplane[step, :, :] = field[0, :, :]
        allDataUplane[step, :, :] = field[1, :, :]

        step = step + 1
        if (step % 250 == 0):
            print(f"Step {step}, t={currentTime:.3f} (dt={(currentTime-prevTime):.3}), avg stress {avgStress:.3f}, stress mean {np.mean(stresses):.3f} sdev {np.std(stresses):.3f}",flush=True)
            print(f"Rewards {avgReward:3f} mean {np.mean(rewards):.3f} sdev {np.std(stresses):.3f}")
            print(f"Max / min control {np.max(control):.3f} {np.min(control):.3f}")

        # Terminate if max simulation time reached
        if currentTime - initTime > 200000:
            done = True

    print("Python sending terminate message to Fortran")
    subComm.Send([requestTerm, MPI.CHARACTER], dest=0, tag=maxProc+100)
    subComm.Disconnect()


    fName = f"{workDir}rew_v{version}.png"
    print(f"saving field {fName}")
    fig, ax = plt.subplots(1,2)
    ax[0].plot(rewards, linestyle=':', color='darkgreen')
    ax[0].plot(np.cumsum(rewards)/np.arange(1,len(stresses)+1), linestyle='-', color='darkgreen', lw=2)
    ax[0].set_title("Reward")
    ax[1].plot(stresses, linestyle=':', color='darkred')
    ax[1].plot(np.cumsum(stresses)/np.arange(1,len(stresses)+1), linestyle='-', color='darkred', lw=2)
    ax[1].set_title("Stress")
    plt.tight_layout()
    plt.savefig(fName)
    plt.close('all')

    with open(f'{workDir}/control.pickle', 'wb') as f:
        pickle.dump(allDataControl, f)

    with open(f'{workDir}/fieldU.pickle', 'wb') as f:
        pickle.dump(allDataUplane, f)

    with open(f'{workDir}/fieldV.pickle', 'wb') as f:
        pickle.dump(allDataVplane, f)

    uFlat = np.reshape(allDataUplane,(-1,nz*nx))
    vFlat = np.reshape(allDataVplane,(-1,nz*nx))

    uCov = np.cov(uFlat, rowvar=False)
    #print(np.diagonal(uCov))

    ucovplot = uCov[:32,:32]
    plt.figure()
    ax = sns.heatmap(ucovplot, linewidth=0.1, vmax=ucovplot.max(), vmin=ucovplot.min(), cmap='vlag')
    #plt.xticks(np.arange(13)+0.5,['2','3','4','5','6','7','8','9','10','J','Q','K','A'])
    #plt.yticks(np.arange(13)+0.5,['2','3','4','5','6','7','8','9','10','J','Q','K','A'])
    fname = f"{workDir}/uCov.png"
    plt.savefig(fname)
    plt.close()

    with open(f'{workDir}/covU.pickle', 'wb') as f:
        pickle.dump(uCov, f)

    vCov = np.cov(vFlat, rowvar=False)
    
    vcovplot = vCov[:32,:32]

    plt.figure()
    ax = sns.heatmap(vcovplot, linewidth=0.1, vmax=vcovplot.max(), vmin=vcovplot.min(), cmap='vlag')
    #plt.xticks(np.arange(13)+0.5,['2','3','4','5','6','7','8','9','10','J','Q','K','A'])
    #plt.yticks(np.arange(13)+0.5,['2','3','4','5','6','7','8','9','10','J','Q','K','A'])
    fname = f"{workDir}/vCov.png"
    plt.savefig(fname)
    plt.close()

    with open(f'{workDir}/covV.pickle', 'wb') as f:
        pickle.dump(vCov, f)

    randU = np.random.multivariate_normal(np.zeros(nz*nx), uCov, Ny)
    randV = np.random.multivariate_normal(np.zeros(nz*nx), vCov, Ny)

    yU = uFlat + randU
    yV = vFlat + randV

    with open(f'{workDir}/dataU.pickle', 'wb') as f:
        pickle.dump(yU, f)

    with open(f'{workDir}/dataV.pickle', 'wb') as f:
        pickle.dump(yV, f)

    return np.mean(stresses), np.mean(rewards), allDataControl, allDataVplane, allDataUplane


    
if __name__ == "__main__":

    lower = 0.
    upper = 100
    heights = getHeights(lower, upper)
    heights = heights[1::3]
    print(heights)
    print(len(heights))
    for idx in range(len(heights)):
        s, r, allDataControl, allDataVplane, allDataUplane = rollout(heights[idx,:])
        sys.exit()
