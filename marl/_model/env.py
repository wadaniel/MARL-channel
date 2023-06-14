import numpy as np
from mpi4py import MPI
from helpers import fieldToState, distribute_field

launchCommand = './bla_16x65x16_1'
workDir = './../bin/'
maxProc = 1

requestState = b'STATE'
requestControl = b'CNTRL'
requestEvolve = b'EVOLV'
requestTerm = b'TERMN'

nx = 16
ny = 65
nz = 16
npl = 2

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


maxSteps = 3000

def env(s):

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

    state = fieldToState(field)
    s["State"] = state

    step = 0
    done = False

    # Receiving initial time of the simulation
    initTime = np.array(0,dtype=np.double)
    subComm.Recv([initTime, MPI.DOUBLE], source=0, tag=maxProc+960)
    currentTime = initTime

    while not done and step < maxSteps:

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
        agent = agents[0]
        reward = 1-np.mean(wallStresses[agent])/baseline_dudy
        #print(f"Reward {reward}, Stress {wallStresses[agent]}",flush=True)
        s["Reward"] = reward 

        #print("Python receiving state from Fortran", flush=True)
        subComm.Send([requestState, MPI.CHARACTER], dest=0, tag=maxProc+100)
        field = np.ndarray((npl-1,nz,nx),dtype=np.double)
        for plidx in range(1,npl):
            for zidx in range(nz):
                subComm.Recv([field[plidx-1,zidx,:], MPI.DOUBLE], source=0, tag=maxProc+10+plidx+zidx+1)

        state = fieldToState(field)
        s["State"] = state

        prevTime = currentTime
        currentTime = np.array(0,dtype=np.double)
        subComm.Recv([currentTime, MPI.DOUBLE], source=0, tag=maxProc+960)

        step = step + 1
        print(f"Step {step}, t={currentTime} (dt={currentTime-prevTime}), reward {reward}",flush=True)

        # Terminate if max simulation time reached
        if currentTime - initTime > 200000:
            done = True

    s["Termination"] = "Terminal"
    print("Python sending terminate message to Fortran")
    subComm.Send([requestTerm, MPI.CHARACTER], dest=0, tag=maxProc+100)
    subComm.Disconnect()

if __name__ == "__main__":

    ## START
    mpi_info = MPI.Info.Create()
    mpi_info.Set('wdir',workDir)
    mpi_info.Set('bind_to','none')
    sub_comm = MPI.COMM_SELF.Spawn(launchCommand,maxprocs=maxProc,info=mpi_info)

    ## STATE
    request = b'STATE'
            
    print("Python sending state message to Fortran")
    sub_comm.Send([request, MPI.CHARACTER], dest=0, tag=maxProc+100)
      
    buf = np.ndarray((nx,),dtype=np.double)
    print("Python receiving state from Fortran")
    for i in range(1,npl):
        for i_z in range(nz):
            sub_comm.Recv([buf, MPI.DOUBLE], source=0, tag=maxProc+10+i+i_z+1)
            print(buf)

    # Receiving current time of the simulation
    current_time = np.array(0,dtype=np.double)
    sub_comm.Recv([current_time, MPI.DOUBLE], source=0,
                    tag=maxProc+960)
    print(f"Simulation time: {current_time}")

    ## CONTROL
    request = b'CNTRL'

    print("Python sending control message to Fortran")
    sub_comm.Send([request, MPI.CHARACTER], dest=0,
                    tag=maxProc+100)

    ctrl_value = np.zeros((nctrlz, nctrlx))
    print("Python sending control to Fortran")
    if ((wbci == 6) or (wbci == 7)):
        for j in range(nctrlx):
            for i in range(nctrlz):
                sub_comm.Send([ctrl_value[i,j], MPI.DOUBLE], 
                                    dest=0,
                                    tag=300+i+1+(j+1)*nctrlz)

    ## EVOLVE
    request = b'EVOLV'

    print("Python sending evolve message to Fortran")
    sub_comm.Send([request, MPI.CHARACTER], dest=0,
                    tag=maxProc+100)

    uxz = np.ndarray((nz, nx),dtype=np.double)

    uxz_avg = np.ndarray((nz, nx),dtype=float)

    buf = np.ndarray((nx,),dtype=np.double)

    # 'Receiving the (non-averaged) reward'
    i_evolv = 1
    while i_evolv <= (ndrl // nst)-1:
        print(f'Receiving the (non-averaged) reward, i_evolv:{i_evolv}', flush=True)
        for i in range(1):
            for i_z in range(nz):
                sub_comm.Recv([buf, MPI.DOUBLE], 
                                source=0, 
                                tag=maxProc+10+i+i_z+1)
                #print(buf)
                uxz[i_z] = buf
        
        # Reward averaging (between two consecutive actions)
        if i_evolv == 1:
            uxz_avg = uxz.astype(float)
        else:
            uxz_avg += uxz.astype(float) #(uxz_avg*i_evolv + uxz.astype(float)/dy)/(i_evolv+1)
        
        i_evolv += 1

    uxz_avg /= ((i_evolv-1)*dy)

    print(f'Computing rewards')

    reward_history= []

    if rew_mode == 'Instantaneous':
        avg_array = uxz_avg
    elif rew_mode == 'MovingAverage':
        reward_history.extend(uxz_avg)
        avg_array = reward_history.average()
    else:
        raise ValueError('Unknown reward computing mode')

    # Distributing the field to compute the individual reward per each agent
    ws_stresses = distribute_field(np.expand_dims(avg_array,0),agents,nctrlx,nctrlz,partial_reward,reward=True)

    # Computing the reward
    rewards = {}
    for i_a, agent in enumerate(agents):
        if ((wbci == 6) or (wbci == 7)):
            reward = 1-np.mean(ws_stresses[agent])/baseline_dudy
        else:
            raise ValueError('Unknown boundary condition option')
        rewards[agent] = reward
        if nctrlz!=nz \
           and nctrlx!=nx:
            print(f"Reward {i_a}: {reward}",flush=True)
        elif i_a == 0:
            print(f"Reward {i_a}: {reward}",flush=True)

    ## STATE
    request = b'STATE'
            
    print("Python sending state message to Fortran")
    sub_comm.Send([request, MPI.CHARACTER], dest=0, tag=maxProc+100)
      
    buf = np.ndarray((nx,),dtype=np.double)
    print("Python receiving state from Fortran")
    for i in range(1,npl):
        for i_z in range(nz):
            sub_comm.Recv([buf, MPI.DOUBLE], source=0, tag=maxProc+10+i+i_z+1)

    # Receiving current time of the simulation
    current_time = np.array(0,dtype=np.double)
    sub_comm.Recv([current_time, MPI.DOUBLE], source=0, tag=maxProc+960)
    print(f"Simulation time: {current_time}")

    ## TERMINATE
    request = b'TERMN'
    print("Python sending terminate message to Fortran")
    sub_comm.Send([request, MPI.CHARACTER], dest=0, tag=maxProc+100)
            
    sub_comm.Disconnect()
