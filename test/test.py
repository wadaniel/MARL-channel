import numpy as np
from mpi4py import MPI
from helpers import distribute_field

## START
mpi_info = MPI.Info.Create()
mpi_info.Set('wdir','../bin/')
mpi_info.Set('bind_to','none')
launchCommand = './bla_16x65x16_1'
maxProc = 1
sub_comm = MPI.COMM_SELF.Spawn(launchCommand,maxprocs=maxProc,info=mpi_info)

## STATE
request = b'STATE'
        
print("Python sending state message to Fortran")
sub_comm.Send([request, MPI.CHARACTER], dest=0, tag=maxProc+100)
  
nx = 16
ny = 65
nz = 16
npl = 2
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

wbci = 7
nctrlx = 16
nctrlz = 16
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
i_evolv = 1
dy = 1.-np.cos(np.pi*1/(ny-1)) 
ndrl = 80
nst = 4

print("Python sending evolve message to Fortran")
sub_comm.Send([request, MPI.CHARACTER], dest=0,
                tag=maxProc+100)

uxz = np.ndarray((nz, nx),dtype=np.double)

uxz_avg = np.ndarray((nz, nx),dtype=float)

buf = np.ndarray((nx,),dtype=np.double)

# 'Receiving the (non-averaged) reward'
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

uxz_avg /= (i_evolv*dy-1)

print(f'Computing rewards')
partial_reward = False
rew_mode = 'Instantaneous'
#rew_mode = 'MovingAverage'
agents = [f"jet_z{r}_x{s}" for s in range(nctrlx)
            for r in range(nctrlz)]


retau = 180
nxs_ = 1
nzs_ = 1
xchange = 1
zchange = 1
reward_history= []

if rew_mode == 'Instantaneous':
    avg_array = uxz_avg
elif rew_mode == 'MovingAverage':
    reward_history.extend(uxz_avg)
    avg_array = reward_history.average()
else:
    raise ValueError('Unknown reward computing mode')

# Distributing the field to compute the individual reward per each agent
ws_stresses = distribute_field(np.expand_dims(avg_array,0),
        nx,nz,nxs_,nzs_,xchange,zchange,agents,nctrlx,nctrlz,partial_reward,reward=True)

baseline_dudy = -1.
if ((wbci == 6) or (wbci == 7)):
    # Load the reference average du/dy at the wall to use as baseline
    baseline_dudy_dict = {"180_16x65x16"   : 3.7398798426242075,
                          "180_32x33x32"   : 3.909412638928125,
                          "180_32x65x32"   : 3.7350180468974763,# 
                          "180_64x65x64"   : 3.82829465265046,
                          "180_128x65x128" : 3.82829465265046}
    baseline_dudy = baseline_dudy_dict[f"{int(retau)}_{nx}x{ny}x{nz}"]

# Computing the reward
rewards = {}
for i_a, agent in enumerate(agents):
    if ((wbci == 6) or (wbci == 7)):
        reward = 1-np.mean(ws_stresses[agent])/baseline_dudy
    else:
        raise ValueError('Unknown boundary condition option')
    # TODO add a baseline value
    rewards[agent] = reward
    if nctrlz!=nz \
       and nctrlx!=nx:
        print(f"Reward {i_a}: {reward}",flush=True)
    elif i_a == 0:
        print(f"Reward {i_a}: {reward}",flush=True)

#print(rewards)
#print("",flush=True)

## STATE
request = b'STATE'
        
print("Python sending state message to Fortran")
sub_comm.Send([request, MPI.CHARACTER], dest=0, tag=maxProc+100)
  
buf = np.ndarray((nx,),dtype=np.double)
print("Python receiving state from Fortran")
for i in range(1,npl):
    for i_z in range(nz):
        sub_comm.Recv([buf, MPI.DOUBLE], source=0, tag=maxProc+10+i+i_z+1)
        #print(buf)

# Receiving current time of the simulation
current_time = np.array(0,dtype=np.double)
sub_comm.Recv([current_time, MPI.DOUBLE], source=0,
                tag=maxProc+960)
print(f"Simulation time: {current_time}")


## TERMINATE
request = b'TERMN'
print("Python sending terminate message to Fortran")
sub_comm.Send([request, MPI.CHARACTER], dest=0, tag=maxProc+100)
        
sub_comm.Disconnect()
