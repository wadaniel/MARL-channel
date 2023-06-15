import numpy as np


def fieldToState(field, compression=1):
    if compression == 1:
        return field.flatten().tolist()

    ns, nz, nx, = field.shape
    cnz = nz//compression
    cnx = nx//compression
    state = np.zeros((ns, cnz, cnx))
    for s in range(ns):
        for zidx in range(cnz):
            for xidx in range(cnx):
                state[s,zidx,xidx] = np.mean(field[s,zidx*compression:(zidx+1)*compression, xidx*compression:(xidx+1)*compression])

    return state.flatten().tolist()

def distribute_field(field,agents,nctrlx,nctrlz,partial_reward,reward=True):
    distributed_fields = {}
    for i_a,agent in enumerate(agents):
        if partial_reward or reward!=True:
            i_z=i_a%nctrlz
            i_x=i_a//nctrlz
            distributed_fields[agent] = np.reshape(field[:,i_z,i_x],[-1,1,1])
        else:
            distributed_fields[agent] = field
    
    return distributed_fields
