import sys
import numpy as np


def fieldToState(field):
    return field.flatten().tolist()

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


def calcControl(nctrlz, nctrlx, step, maxv, version):
    control = np.zeros((nctrlz, nctrlx))

    if version == 0:
        pass

    # Stripes in z
    elif version == 1:
        for idx in range(nctrlz):
            if (step + idx) % 2 == 0:
                control[idx,:] = maxv
            else:
                control[idx,:] = -maxv

    # Stripes in x
    elif version == 2:
        for idx in range(nctrlx):
            if (step + idx) % 2 == 0:
                control[:,idx] = maxv
            else:
                control[:,idx] = -maxv

    # Checkerboard like
    elif version == 3:
        for zidx in range(nctrlz):
            for xidx in range(nctrlx):
                m = zidx + xidx + step
                if m % 2 == 0:
                    control[zidx,xidx] = maxv
                else:
                    control[zidx,xidx] = -maxv

    # Random normal
    elif version == 4:
        control = np.random.normal(loc=0., scale=maxv, size=(nctrlz, nctrlx))

    # Random uniform
    elif version == 5:
        control = np.random.uniform(low=-maxv, high=maxv, size=(nctrlz, nctrlx))

    # Random up/down
    elif version == 6:
        control = np.random.uniform(low=-1, high=1., size=(nctrlz, nctrlx))
        control[control>0] = maxv
        control[control<=0] = -maxv

    else:
        print("[helpers] control version not recognized")
        sys.exit()

    return control
