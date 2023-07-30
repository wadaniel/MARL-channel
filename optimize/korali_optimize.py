import numpy as np
import os
import shutil
from mpi4py import MPI
from helpers import distribute_field
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

launchCommand = './bla_16x65x16_1'
#launchCommand = './bla_16x65x16_1_debug'
srcDir = './../bin/'
maxProc = 1

requestState = b'STATE'
requestControl = b'CNTRL'
requestEvolve = b'EVOLV'
requestTerm = b'TERMN'

seed = 1337

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

alpha = 1.0
maxSteps = 3000

saveFrqncy = 500
printFrqncy = 1000

stepfac = 1
maxv = 0.04285714285714286

def rollout(s, workDir):
  
    np.random.seed(seed)

    bsdev = s["Parameters"][0]
    asdev = s["Parameters"][1]
    amu   = s["Parameters"][2]

    #global version
    #print(version)

    #print(f"Launching SIMSON from workdir {workDir}",flush=True)
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


        # Calculating control action
        sig = np.clip(asdev*np.abs(field[0,:,:]) + bsdev, a_min=0.01, a_max=999)
        mu  = amu*field[0,:,:]
        control = np.random.normal(mu, sig)
        control = np.clip(control,a_min=-maxv,a_max=maxv)
        control -= np.mean(control)

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

        """
        if saveFrqncy > 0 and step % saveFrqncy == 0:
            fieldName = f"{workDir}ux_v{version}_s{step}.png"
            print(f"saving field {fieldName}")
            fig, ax = plt.subplots(1,2)
            x, z = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Lz, nz))
            c = ax[0].pcolormesh(z, x, field[0,:,:], cmap='RdBu', vmin=field[0,:,:].min(), vmax=field[0,:,:].max())
            fig.colorbar(c, ax=ax[0])
            c = ax[1].pcolormesh(z, x, field[1,:,:], cmap='RdBu', vmin=field[1,:,:].min(), vmax=field[1,:,:].max())
            fig.colorbar(c, ax=ax[1])
            plt.savefig(fieldName)
            print("done")

            cfieldName = f"{workDir}contol_v{version}_s{step}.png"
            print(f"saving control {cfieldName}")
            fig, ax = plt.subplots()
            x, z = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Lz, nz))
            c_min = control.min()
            c_max = control.max()
            c = ax.pcolormesh(z, x, control, cmap='RdBu', vmin=c_min, vmax=c_max)
            fig.colorbar(c, ax=ax)
            plt.savefig(cfieldName)
            print("done")

            wfieldName = f"{workDir}stress_v{version}_s{step}.png"
            print(f"saving stresses {wfieldName}")
            fig, ax = plt.subplots()
            x, z = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Lz, nz))
            c = ax.pcolormesh(z, x, uxzAvg, cmap='RdBu', vmin=uxzAvg.min(), vmax=uxzAvg.max())
            fig.colorbar(c, ax=ax)
            plt.savefig(wfieldName)
            print("done")
            plt.close("all")
        """

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

        step = step + 1
        if (step % printFrqncy == 0):
            print(f"Step {step}, t={currentTime:.3f} (dt={(currentTime-prevTime):.3}), avg stress {avgStress:.3f}, stress mean {np.mean(stresses):.3f} sdev {np.std(stresses):.3f}",flush=True)
            print(f"Rewards {avgReward:3f} mean {np.mean(rewards):.3f} sdev {np.std(stresses):.3f}")
            print(f"Max / min control {np.max(control):.3f} {np.min(control):.3f}")

        # Terminate if max simulation time reached
        if currentTime - initTime >= 200000:
            done = True

    print("Python sending terminate message to Fortran")
    subComm.Send([requestTerm, MPI.CHARACTER], dest=0, tag=maxProc+100)
    subComm.Disconnect()


    """
    fName = f"{workDir}rew_v{version}.png"
    print(f"saving field {fName}")
    fig, ax = plt.subplots(1,2)
    ax[0].plot(rewards, linestyle='--', color='b')
    ax[0].plot(np.cumsum(rewards)/np.arange(1,len(stresses)+1), linestyle='-', color='k')
    ax[0].set_title("Rewards")
    ax[1].plot(stresses, linestyle='--', color='b')
    ax[1].plot(np.cumsum(stresses)/np.arange(1,len(stresses)+1), linestyle='-', color='k')
    ax[1].set_title("Stresses")
    plt.savefig(fName)
    print("done")
    plt.close('all')
    """

    print(f"Params bsdev {bsdev} asdev {asdev} amu {amu}")
    print(f"Final Reward mean {np.mean(rewards):.3f} sdev {np.std(stresses):.3f}")
    s["F(x)"] = np.mean(rewards)


    
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--maxGenerations',
        help='Maximum Number of generations to run',
        default=100,
        type=int,
        required=False)    
    parser.add_argument(
        '--run',
        help='Run tag',
        default=0,
        type=int,
        required=False)    
    parser.add_argument(
        '--pop',
        help='Population size',
        default=8,
        type=int,
        required=False)    

    args = parser.parse_args()

    workDir = f'./../cmaes{args.run}/'
    os.makedirs(workDir, exist_ok=True)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print(f'[korali_optimize] rank 0 copying files to {workDir}')
        shutil.copy(srcDir + "bla.i", workDir)
        shutil.copy(srcDir + "bla_16x65x16_1", workDir)
        shutil.copy(srcDir + "bla_16x65x16_1_debug", workDir)
    MPI.COMM_WORLD.Barrier()

    # Importing computational model
    import sys
    import math

    # Starting Korali's Engine
    import korali
    k = korali.Engine()

    # Creating new experiment
    e = korali.Experiment()

    # Configuring Problem
    e["Random Seed"] = 0xC0FEE
    e["Problem"]["Type"] = "Optimization"
    e["Problem"]["Objective Function"] = lambda s : rollout(s,workDir)

    # Defining the problem's variables (max/min +/-0.04)
    e["Variables"][0]["Name"] = "bsdev"
    e["Variables"][0]["Lower Bound"] = -5.0
    e["Variables"][0]["Upper Bound"] = 5.0
    e["Variables"][0]["Initial Mean"] = 0.0
    e["Variables"][0]["Initial Standard Deviation"] = 0.5

    e["Variables"][1]["Name"] = "asdev"
    e["Variables"][1]["Lower Bound"] = -5.0
    e["Variables"][1]["Upper Bound"] = 5.
    e["Variables"][1]["Initial Mean"] = 0.
    e["Variables"][1]["Initial Standard Deviation"] = 0.5

    e["Variables"][2]["Name"] = "amu"
    e["Variables"][2]["Lower Bound"] = -5.
    e["Variables"][2]["Upper Bound"] = 5.0
    e["Variables"][2]["Initial Mean"] = 0.
    e["Variables"][2]["Initial Standard Deviation"] = 0.5

    # Configuring CMA-ES parameters
    e["Solver"]["Type"] = "Optimizer/CMAES"
    e["Solver"]["Population Size"] = args.pop
    e["Solver"]["Mu Value"] = args.pop//4
    e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-32
    e["Solver"]["Termination Criteria"]["Max Generations"] = args.maxGenerations

    # Configuring results path
    e["File Output"]["Enabled"] = True
    e["File Output"]["Path"] = f'_korali_result_cmaes_{args.run}'
    e["File Output"]["Frequency"] = 1

    k.setMPIComm(MPI.COMM_WORLD)
    k["Conduit"]["Type"] = "Distributed"
    k["Conduit"]["Ranks Per Worker"] = 1

    # Running Korali
    k.run(e)
