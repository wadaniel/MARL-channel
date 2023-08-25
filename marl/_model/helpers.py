import numpy as np

def field_to_state(field, nagx, nagz, compression=1):
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

    na = nagx*nagz
    if nagx*nagz == 1:
        return state.flatten().tolist()
    else:
        idx = 0
        state = []*na
        canz=cnz//nagz
        canx=cnx//nagx
        for zidx in range(nagz):
            for xidx in range(nagx):
                state[idx] = state[:,zidx*canz:(zidx+1)*canz, xidx*canx:(xidx+1)*canx].flatten().tolist()
                idx+=1

        return state


def action_to_control(action, nagx, nagz, nctrlx, nctrlz):
    na = nagx*nagz
    if na == 1:
        control = np.reshape(action,(nctrlz, nctrlx))
        return control
    else:
        idx = 0
        canz=nctrlz//nagz
        canx=nctrlx//nagx
        control = np.zeros((nctrlz, nctrlx))
        for zidx in range(nagz):
            for xidx in range(nagx):
                control[zidx*canz:(zidx+1)*canz,  xidx*canx:(xidx+1)*canx] = np.reshape(action[idx],(canz,canx))
                idx+=1

        return control
            
        
def field_to_reward(field,nagx,nagz,baseline_dudy):
    na = nagx*nagz
    if na == 1:
        return 1.-np.mean(field)/baseline_dudy
    else:
        idx = 0
    
        nz, nx = field.shape
        canz= nz//nagz
        canx= nx//nagx

        rewards = [0]*na
        for zidx in range(nagz):
            for xidx in range(nagx):
                rewards[idx] = 1.-np.mean(field[zidx*nz:(zidx+1)*nz, xidx*nx:(xidx+1)*nx])/baseline_dudy
                idx+=1

        return rewards
