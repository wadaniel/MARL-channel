import numpy as np

def field_to_state(field, nagx, nagz, compression=1):
    na = nagx*nagz
    ns, nz, nx, = field.shape
    cnz = nz//compression
    cnx = nx//compression

    if compression == 1:
        state = field
    else:
        state = np.zeros((ns, cnz, cnx))
        for s in range(ns):
            for zidx in range(cnz):
                for xidx in range(cnx):
                    state[s,zidx,xidx] = np.mean(field[s,zidx*compression:(zidx+1)*compression, xidx*compression:(xidx+1)*compression])

    if na == 1:
        return state.flatten().tolist()
    else:
        idx = 0
        canz=cnz//nagz
        canx=cnx//nagx
        stateList = []
        for zidx in range(nagz):
            for xidx in range(nagx):
                stateList.append(state[:,zidx*canz:(zidx+1)*canz, xidx*canx:(xidx+1)*canx].flatten().tolist())
                idx+=1

        return stateList


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

        for idx in range(len(action)):
            for zidx in range(nagz):
                for xidx in range(nagx):
                    control[zidx*canz:(zidx+1)*canz,xidx*canx:(xidx+1)*canx] = np.reshape(action[idx],(canz,canx))

        return control
            
        
def field_to_reward(field,nagx,nagz,nz,nx,baseline_dudy):
    na = nagx*nagz
    if na == 1:
        return 1.-np.mean(field)/baseline_dudy
    else:
        idx = 0
        canz= nz//nagz
        canx= nx//nagx

        rewards = [0]*na
        for idx in range(na):
            for zidx in range(nagz):
                for xidx in range(nagx):
                    rewards[idx] = 1.-np.mean(field[zidx*canz:(zidx+1)*canz, xidx*canx:(xidx+1)*canx])/baseline_dudy

        return rewards
