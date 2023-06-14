import numpy as np

def distribute_field(field,nx,nz,nxs_,nzs_,xchange,zchange,agents,nctrlx,nctrlz,partial_reward,reward=True):

    # TODO modify to accept also nx==nctrlx or nz==nctrlz
    # TODO add the possibility to include the neighbouring jets if
    #      nx==nctrlx and/or nz==nctrlz

    # agent_field = np.ndarray((npls,
    #                           nzs,
    #                           nxs),
    #                           dtype=np.double)
    
    distributed_fields = {}

    field = np.roll(field,nx//2,axis=2)

    if nctrlz!=nz \
       and nctrlx!=nx:
        zcenters = np.arange(
            nz//nctrlz//2,
            nz,
            step=nz//nctrlz)

        xcenters = np.arange(
            nx//nctrlx//2,
            nx,
            step=nx//nctrlx)
        
        for i_a,agent in enumerate(agents):
            i_z=i_a%nctrlz
            i_x=i_a//nctrlz
            i_zc = i_z#nctrlz-i_z-1
            i_xc = i_x

            if zcenters[i_zc]<nzs_//2+int(zchange):
                zroll = nzs_//2+int(zchange)
            elif zcenters[i_zc]+nzs_//2+int(zchange)>nz:
                zroll = -(nzs_//2+int(zchange))
            else:
                zroll=0

            if xcenters[i_xc]<nxs_//2+int(xchange):
                xroll = nxs_//2+int(xchange)
            elif xcenters[i_xc]+nxs_//2+int(xchange)>nx:
                xroll = -(nxs_//2+int(xchange))
            else:
                xroll=0

            agent_field = np.roll(field,
                (zroll,xroll),axis=(1,2))[:,zcenters[i_zc]+
                        zroll-nzs_//2:zcenters[i_zc]+
                        zroll+nzs_//2+
                        int(zchange)+1,xcenters[i_xc]+
                        xroll-nxs_//2:xcenters[i_xc]+
                        xroll+nxs_//2+
                        int(xchange)+1]
            
            distributed_fields[agent] = agent_field

    elif nctrlz!=nz \
       or nctrlx!=nx:
        raise NotImplementedError('This feature is not currently available')
    else:
        for i_a,agent in enumerate(agents):
            if partial_reward or reward!=True:
                i_z=i_a%nctrlz
                i_x=i_a//nctrlz
                distributed_fields[agent] = np.reshape(field[:,i_z,i_x],[-1,1,1])
            else:
                distributed_fields[agent] = field
    # breakpoint()
    return distributed_fields
