import time
import numpy as np
import pickle
from mpi4py import MPI
from helpers import field_to_state, action_to_control, field_to_reward
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

maxProc = 1

requestState = b'STATE'
requestControl = b'CNTRL'
requestEvolve = b'EVOLV'
requestTerm = b'TERMN'

nx = 16
ny = 65
nz = 16
npl = 3

# Load the reference average du/dy at the wall to use as baseline (for wbci 6 or 7)
opposition_dudy_dict = {"180_16x65x16"   : 3.7398798426242075,
                      "180_32x33x32"   : 3.909412638928125,
                      "180_32x65x32"   : 3.7350180468974763,#
                      "180_64x65x64"   : 3.82829465265046,
                      "180_128x65x128" : 3.82829465265046}

retau = 180
#baseline_dudy = opposition_dudy_dict[f"{int(retau)}_{nx}x{ny}x{nz}"]
baseline_dudy = 3.671
maxv = 0.04285714285714286

wbci = 7
nctrlx = 16
nctrlz = 16

dy = 1.-np.cos(np.pi*1/(ny-1)) 
ndrl = 80
nst = 4

rew_mode = 'Instantaneous'
#rew_mode = 'MovingAverage'
partial_reward = False
agents = [f"jet_z{zi}_x{xi}" for xi in range(nctrlx) for zi in range(nctrlz)]

h=1.
Lx = 2.67*h
Lz = 0.8*h

nxs_ = 1
nzs_ = 1
xchange = 1
zchange = 1

printFrequency = 100

version = 9 # V-RACER tag for plotting
saveFrqncy = 500

def env(s, args):

    start = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()-1

    compression = args.compression
    wdir = f"{args.resDir}/sample{rank}/"

    mode = int(s["Custom Settings"]["Mode"])
    testing = True if s["Mode"] == "Testing" else False

    bestCumReward = -999.

    if testing:
        if os.path.exists(f"{wdir}/cumReward.dat"):
            rewardfile = open(f"{wdir}/cumReward.dat","r")
            bestCumReward = float(rewardfile.read())
            rewardfile.close()

        print(f"[env] Run testing generation in {wdir} (current best reward {bestCumReward})")

    else:
        if os.path.exists(f"{wdir}/cumRewardTrain.dat"):
            rewardfile = open(f"{wdir}/cumRewardTrain.dat","r")
            bestCumReward = float(rewardfile.read())
            rewardfile.close()

        print(f"[env] Run training generation in {wdir} (current best reward {bestCumReward})")
    
       
    os.chdir(wdir)
    wdir = os.getcwd()

    mpi_info = MPI.Info.Create()
    mpi_info.Set('wdir',wdir)
    mpi_info.Set('bind_to','none')

    launchCommand = f'./bla_16x65x16_1'
    print(f"[env] Python rank {rank}/{size} sending launch message to Fortran")
    subComm = MPI.COMM_SELF.Spawn(launchCommand,maxprocs=maxProc,info=mpi_info)

    rewards = []

    allDataUplane = np.empty((args.episodeLength, nz, nx))
    allDataVplane = np.empty((args.episodeLength, nz, nx))
    allDataControl = np.empty((args.episodeLength, nz, nx))
    allDataStress = np.empty((args.episodeLength, nz, nx))

    try:

        #print("Python receiving state from Fortran", flush=True)
        subComm.Send([requestState, MPI.CHARACTER], dest=0, tag=maxProc+100)
        field = np.ndarray((npl-1,nz,nx),dtype=np.double)
        for plidx in range(1,npl):
            for zidx in range(nz):
                subComm.Recv([field[plidx-1,zidx,:], MPI.DOUBLE], source=0, tag=maxProc+10+plidx+zidx+1)

        s["State"] = field_to_state(field, nagx=args.nagx, nagz=args.nagz, compression=args.compression)

        step = 0
        done = False
        cumReward = 0.

        # Receiving initial time of the simulation
        initTime = np.array(0,dtype=np.double)
        subComm.Recv([initTime, MPI.DOUBLE], source=0, tag=maxProc+960)
        currentTime = initTime

        while not done and step < args.episodeLength:

            # Getting new action from korali
            s.update()
        
            # Retrieve control from korali
            action = s["Action"] 
            control = action_to_control(action, field, args.nagx, args.nagz, nctrlx, nctrlz, compression, mode)
            control -= np.mean(control)
            control = np.clip(control, a_min=-maxv, a_max=maxv)
            #assert (np.abs(control) < 3.*maxv).all(), f"sth wrong with action magnitude {control}" 

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
            reward = field_to_reward(uxzAvg,args.nagx,args.nagz,nz,nx,baseline_dudy)
            s["Reward"] = reward
            cumReward += np.mean(reward)
            rewards.append(np.mean(reward))
            #print(f"Reward {reward}",flush=True)

            #print("Python receiving state from Fortran", flush=True)
            subComm.Send([requestState, MPI.CHARACTER], dest=0, tag=maxProc+100)
            field = np.ndarray((npl-1,nz,nx),dtype=np.double)
            for plidx in range(1,npl):
                for zidx in range(nz):
                    subComm.Recv([field[plidx-1,zidx,:], MPI.DOUBLE], source=0, tag=maxProc+10+plidx+zidx+1)

            s["State"] = field_to_state(field, nagx=args.nagx, nagz=args.nagz, compression=args.compression)

            prevTime = currentTime
            currentTime = np.array(0,dtype=np.double)
            subComm.Recv([currentTime, MPI.DOUBLE], source=0, tag=maxProc+960)

            step = step + 1

            # store data
            allDataControl[step-1, :, :] = control.copy()
            allDataVplane[step-1, :, :] = field[0, :, :].copy()
            allDataUplane[step-1, :, :] = field[1, :, :].copy()
            allDataStress[step-1, :, :] = uxzAvg.copy()

            """
            # console output
            if (step % printFrequency == 0):
                print(f"[env] Step {step}, t={currentTime} (dt={(currentTime-prevTime):.3}), reward {np.mean(reward):.3f}, reward mean {(cumReward/step):.3f}",flush=True)
            """

        s["Termination"] = "Terminal"
     
        # write data
        if cumReward > bestCumReward:

            bestCumReward = cumReward
            print(f"[env] Storing generation with cumulative reward {cumReward} (rank {rank}) in {wdir}")

            if testing:
                rewardfile = open(f"{wdir}/cumReward.dat","w")
                rewardfile.write(f'{bestCumReward}')
                rewardfile.close()

                with open(f'{wdir}/control_r{rank}.pickle', 'wb') as f:
                    pickle.dump(allDataControl, f)

                with open(f'{wdir}/fieldU_r{rank}.pickle', 'wb') as f:
                    pickle.dump(allDataUplane, f)

                with open(f'{wdir}/fieldV_r{rank}.pickle', 'wb') as f:
                    pickle.dump(allDataVplane, f)

                with open(f'{wdir}/stress_r{rank}.pickle', 'wb') as f:
                    pickle.dump(allDataStress, f)

                with open(f'{wdir}/rewards_r{rank}.pickle', 'wb') as f:
                    pickle.dump(np.array(rewards), f)

            else:
                rewardfile = open(f"{wdir}/cumRewardTrain.dat","w")
                rewardfile.write(f'{bestCumReward}')
                rewardfile.close()

                with open(f'{wdir}/control_train_r{rank}.pickle', 'wb') as f:
                    pickle.dump(allDataControl, f)

                with open(f'{wdir}/fieldU_train_r{rank}.pickle', 'wb') as f:
                    pickle.dump(allDataUplane, f)

                with open(f'{wdir}/fieldV_train_r{rank}.pickle', 'wb') as f:
                    pickle.dump(allDataVplane, f)

                with open(f'{wdir}/stress_train_r{rank}.pickle', 'wb') as f:
                    pickle.dump(allDataStress, f)

                with open(f'{wdir}/rewards_train_r{rank}.pickle', 'wb') as f:
                    pickle.dump(np.array(rewards), f)

    except Exception as e:

        print(f"Exception occured {e}")
        s["Reward"] = -20.
        s["Termination"] = "Truncated"
        cumReward -= 20

    finally:

        print(f"[env] Python rank {rank}/{size} sending terminate message to Fortran")
        subComm.Send([requestTerm, MPI.CHARACTER], dest=0, tag=maxProc+100)
        subComm.Disconnect()

    end = time.time()

    if testing:
        print(f"[env] Cumulative testing reward rank {rank}/{size}: {cumReward}, Took {end-start}s")
    else:
        print(f"[env] Cumulative training reward rank {rank}/{size}: {cumReward}, Took {end-start}s")
