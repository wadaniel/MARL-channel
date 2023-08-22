import time
import numpy as np
from mpi4py import MPI
from helpers import fieldToState, distribute_field

launchCommand = './bla_16x65x16_1'
maxProc = 1

requestState = b'STATE'
requestControl = b'CNTRL'
requestEvolve = b'EVOLV'
requestTerm = b'TERMN'

nx = 16
ny = 65
nz = 16
npl = 3

wbci = 7
nctrlx = 16
nctrlz = 16

dy = 1.-np.cos(np.pi*1/(ny-1)) 
ndrl = 28
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

printFrequency = 100
# Load the reference average du/dy at the wall to use as baseline (for wbci 6 or 7)
opposition_dudy_dict = {"180_16x65x16"   : 3.7398798426242075,
                      "180_32x33x32"   : 3.909412638928125,
                      "180_32x65x32"   : 3.7350180468974763,#
                      "180_64x65x64"   : 3.82829465265046,
                      "180_128x65x128" : 3.82829465265046}

version = 9 # V-RACER tag for plotting
baseline_dudy = opposition_dudy_dict[f"{int(retau)}_{nx}x{ny}x{nz}"]

def env(s, args):

    start = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()-1

    mpi_info = MPI.Info.Create()
    mpi_info.Set('wdir',args.workDir)
    mpi_info.Set('bind_to','none')

    print(f"[env] Python rank {rank}/{size} sending launch message to Fortran")
    subComm = MPI.COMM_SELF.Spawn(launchCommand,maxprocs=maxProc,info=mpi_info)

    if args.test:
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

        state = fieldToState(field, compression=args.compression)
        s["State"] = state

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
            control = s["Action"] 
            control = np.reshape(control,(nctrlz, nctrlx))
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

            uxzAvg /= ((i_evolv-1)*dy)

            # Distributing the field to compute the individual reward per each agent
            wallStresses = distribute_field(np.expand_dims(uxzAvg,0),agents,nctrlx,nctrlz,partial_reward,reward=True)

            # Computing the reward
            reward = 0.
            for agent in agents:
                reward += 1-np.mean(wallStresses[agent])/baseline_dudy
            reward /= len(agents)
            cumReward += reward
            #print(f"Reward {reward}, Stress {wallStresses[agent[0]]}",flush=True)
            s["Reward"] = reward

            #print("Python receiving state from Fortran", flush=True)
            subComm.Send([requestState, MPI.CHARACTER], dest=0, tag=maxProc+100)
            field = np.ndarray((npl-1,nz,nx),dtype=np.double)
            for plidx in range(1,npl):
                for zidx in range(nz):
                    subComm.Recv([field[plidx-1,zidx,:], MPI.DOUBLE], source=0, tag=maxProc+10+plidx+zidx+1)

            state = fieldToState(field, compression=args.compression)
            s["State"] = state

            prevTime = currentTime
            currentTime = np.array(0,dtype=np.double)
            subComm.Recv([currentTime, MPI.DOUBLE], source=0, tag=maxProc+960)

            # store data
            if args.test:
                allDataControl[step, :, :] = control
                allDataVplane[step, :, :] = field[0, :, :]
                allDataUplane[step, :, :] = field[1, :, :]
                allDataStress[step, :, :] = uxzAvg

            step = step + 1

            # console output
            if (args.test and step % printFrequency == 0):
                print(f"[env] Step {step}, t={currentTime} (dt={(currentTime-prevTime):.3}), reward {reward:.3f}, reward mean {(cumReward/step):.3f}",flush=True)

            # Terminate if max simulation time reached
            if currentTime - initTime > 200000:
                done = True

        s["Termination"] = "Terminal"
     
        # write data
        if args.test:
            with open(f'{args.workDir}/control_v{version}.pickle', 'wb') as f:
                pickle.dump(allDataControl, f)

            with open(f'{args.workDir}/fieldU_v{version}.pickle', 'wb') as f:
                pickle.dump(allDataUplane, f)

            with open(f'{args.workDir}/fieldV_v{version}.pickle', 'wb') as f:
                pickle.dump(allDataVplane, f)

            with open(f'{args.workDir}/stress_v{version}.pickle', 'wb') as f:
                pickle.dump(allDataStress, f)

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
    print(f"[env] Cumulative reward rank {rank}/{size}: {cumReward}, Took {end-start}s")
