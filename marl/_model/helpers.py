import numpy as np

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
