#import tpsa
import numpy as np
from itertools import product

# --- permutation of power index
def powerIndex(nv=4,norder=5):
    '''
    define power index array with nv variables up to their norder order
    nv: number of variable, 4 by default
    norder: order of power 5 by default
    return: power index array
    '''
    B = [range(norder+1)[::-1]]*nv
    a = []
    for i in xrange(norder+1):
        a += [list(ib) for ib in product(*B) if sum(ib)==i]
    return np.array(a)


# transport of a mvp through a linear matrix
def lmPass(lm,x):
    '''
    transport a tpsa list through a linear matrix 
    lm: linear transport square matrix
    x: tpsa list
    '''
    lm = np.array(lm,dtype=float)
    temp = []
    for mi in lm:
        t = 0
        for j,mj in enumerate(mi):
            t += mj*x[j]
        temp.append(t)
    return temp

def thinOctPass(K3L,x):
    '''
    thinlens oct with given K3L
    '''
    x[1] += -K3L/6*(x[0]*x[0]*x[0]-3*x[0]*x[2]*x[2])
    x[3] += -K3L/6*(x[2]*x[2]*x[2]-3*x[0]*x[0]*x[2])
    return x


# mvp for thin-lens sextupole
def thinSextPass(K2L,x):
    '''
    thinlens sext with given K2L
    '''
    x[1] += -K2L/2*(x[0]*x[0]-x[2]*x[2])
    x[3] += K2L*x[0]*x[2]
    return x

def thickSextPass(L,K2,nsl,x):
    '''
    combine thin-lens sext and drift to thick sextupole
    nsl: number of slicing
    '''
    Ld = float(L)/nsl
    Ld2 = float(L)/nsl/2
    dtm,dtm2 = np.eye(4),np.eye(4)
    dtm[0,1],dtm[2,3] = Ld,Ld
    dtm2[0,1],dtm2[2,3] = Ld2,Ld2
    dk2l = L*K2/nsl
    temp = lmPass(dtm2,x)
    temp = thinSextPass(dk2l,temp)
    for i in xrange(nsl-1):
        temp = lmPass(dtm,temp)
        temp = thinSextPass(dk2l,temp)
    temp = lmPass(dtm2,temp)
    return  temp
