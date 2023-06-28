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


import os
import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import math
import time


def getHeights(lower, upper):
    #dy = abs(y[:-1] - y[1:])

    ## Minimal channel
    # lx = 2.67#*math.pi
    # lz = 0.8#math.pi
    Re_tau = 180
    nx = 192
    nz = 192
    # nx = 16
    # nz = 16
    ny = 65

    # Full channel
    # lx = 2*math.pi
    # lz = math.pi
    # Re_tau = 180
    # # nx = 192
    # # nz = 192
    # nx = 64
    # nz = 64
    # ny = 65

    lx = 4*math.pi
    lz = 2*math.pi
    # Re_tau = 550
    # nx = 512#*3/2
    # nz = 512#*3/2
    # ny = 193

    # lx = 2*math.pi
    # lz = 1*math.pi
    # Re_tau = 360
    # nx = 128#256#*3/2
    # nz = 128#256#*3/2
    # ny = 65


    # Re_tau = 180
    # Re_tau = 550

    Re_cl_dict = {'180':2100, '360':9290/2,'550':15037.50/2}
    Re_cl = Re_cl_dict[str(Re_tau)]
    nu = 1/Re_cl # According to the code

    # U bulk
    #u_b = 0.5*np.sum(dy*u_avg[1:])

    #U tau
    #u_tau = np.sqrt(u_avg[1]/dy[0]*nu)

    #Re_tau = u_tau*2/nu
    u_tau = (Re_tau*nu)/2

    lstar=nu/u_tau;

    dx = lx/nx/lstar
    dz = lz/nz/lstar
    #print(f'dx: {dx:.3f}, dz: {dz:.3f}')
    # tstar = nu/u_tau**2
    #u_p = u_avg/u_tau
    #y_p = (y+1)*u_tau/nu

    yF=np.cos(math.pi*np.arange(0,1+1/(ny-1),1/(ny-1)))
    #yF=yF+yF[0]
    dy = abs(yF[:-1] - yF[1:])

    y_p = (yF+1)*u_tau/nu
    dy_p = dy*u_tau/nu

    yy = np.concatenate((yF.reshape((-1,1)), y_p.reshape((-1,1))), axis=1)
    indeces = (yy[:,1] >= lower) * (yy[:,1] <= upper)
    #print(yy)
    return yy[indeces,:]
