from mvp import mvp
import numpy as np
from itertools import product
import scipy.sparse as sps
import copy

# ---  unit mvp for nv variable
def const(nv=4):
    '''
    define unit mvp constant with nv variables
    nv: number of variable, 4 by default
    return: unit mvp list, zero and one
    '''
    unit = []
    for i in xrange(nv):
        ind = np.zeros((1,nv))
        ind[0,i] = 1
        unit.append(mvp(ind,[1]))
    zero = mvp.const(unit[0],0)
    one = mvp.const(unit[0],1)
    return unit, zero, one

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
def lmPass(lm,mvplist):
    '''
    transport a mvp list through a linear matrix 
    lm: linear transport square matrix
    mvplist: mvp list
    '''
    lm = np.array(lm,dtype=float)
    #nlm = lm.shape[0]
    #if nlm != len(mvplist):
    #    raise RuntimeError('unmatched linear matrix and mvp list dimensions')
    mt = []
    for r in lm:
        temp = mvp.zero_like(mvplist[0])
        for j,rj in enumerate(r):
            temp += mvplist[j]*mvplist[j].const(rj)
        temp.simplify()
        mt.append(temp)
    return mt

def addDispersive(mvplist,delta,disp):
    '''
    add dispersive orbit: disp = [etax,etaxp,etay,etayp]
    delta: dp/p
    mvplist: [x,px,y,py]
    '''
    for i in xrange(len(mvplist)):
        mvplist[i] += mvplist[i].const(disp[i]*delta)
    return mvplist

# mvp for thin-lens sextupole
def thinSextPass(K2L,mvplist,truncate=7):
    '''
    thinlens sext with given K2L
    '''
    temp = copy.deepcopy(mvplist)
    m1 = mvplist[0].square(truncate=truncate)-mvplist[2].square(truncate=truncate)
    m1.value *= (-K2L/2)
    temp[1] = mvplist[1]+m1
    m2 = mvplist[0]*mvplist[2]
    m2.truncate(order=truncate)
    m2.value *= K2L
    temp[3] = mvplist[3]+m2
    temp[1].simplify()
    temp[3].simplify()
    return temp

def thickSextPass(L,K2,nsl,mvplist,truncate=7):
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
    temp = lmPass(dtm2,mvplist)
    temp = thinSextPass(dk2l,temp,truncate=truncate)
    for i in xrange(nsl-1):
        temp = lmPass(dtm,temp)
        temp = thinSextPass(dk2l,temp,truncate=truncate)
    temp = lmPass(dtm2,temp)
    return  temp


def sqmvp(mt,a,one=None):
    '''
    expand a mvp list to a square mvp list
    '''
    if not one:
        one = mvp.const(mt[0],1)
    t = []
    for ai in a:
        temp = one
        for i,n in enumerate(ai):
            temp *= mt[i]**n
        t.append(temp)
    return t

def aline(m,a):
    '''
    fill each term in m (a mvp) in to a (index list) corresponding column
    '''
    w = len(a)
    r = np.zeros((w,))
    for j,ii in enumerate(m.index):
        ci = np.nonzero([(ii==a[i]).all() for i in xrange(w)])[0]
        r[ci] = m.value[j]
    return r

def constructSM(t,a):
    '''
    from t (taylor series) to construct Square Matrix row by row
    '''
    d = len(a)
    m = np.zeros((d,d))
    for i,ti in enumerate(t):
        m[i] = aline(ti,a)
    return m

def sqmat_tm(tm,nv,a,zero=None,unit=None,one=None):
    '''
    square matrix for arbitary linear transfer
    '''
    if not zero:
        unit,zero,one = const(nv=nv)
    mt = tm2mvp(tm,nv,zero=zero,unit=unit)
    t = sqmvp(mt,a,one=one)
    m = constructSM(t,a)
    m = sps.csr_matrix(m)
    return m

def sqmat_thinsext(K2L,a,unit=None,one=None):
    '''
    square matrix for thin-lens sext given K2L
    '''
    if not one:
        unit,zero,one = const(len(a[0]))
    mt = thinSext(K2L,unit)
    t = sqmvp(mt,a,one=one)
    m = constructSM(t,a)
    m = sps.csr_matrix(m)
    return m

    
