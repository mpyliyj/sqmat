from mvp import mvp
import numpy as np
from itertools import product
import scipy.sparse as sps

# ---  unit mvp for nv variable
def const(nv=4):
    '''
    create unit constant for nv variable
    return unit list, constant zero and one
    '''
    unit = []
    for i in xrange(nv):
        ind = np.zeros((1,nv))
        ind[0,i] = 1
        unit.append(mvp(ind,[1]))
    zero = mvp.const(unit[0],0.)
    one = mvp.const(unit[0],1.)
    return unit, zero, one

# --- permutation of power index
def indexpower(nv=4,norder=5):
    '''
    create power index array for nv variables up to norder
    '''
    norder = norder
    B = [range(norder+1)[::-1]]*nv
    a = []
    for i in xrange(norder+1):
        a += [list(ib) for ib in product(*B) if sum(ib)==i]
    return np.array(a)

# mvp for arbitary linear nxn matrix
def tm2mvp(tm,nv,zero=None,unit=None):
    '''
    convert arbitary linear matrix (tm) to mvp
    return mvp list for each row of tm
    '''
    if not zero:
        unit,zero,one = const(nv=nv)
    tm = np.array(tm[:nv,:nv])
    mt = []
    for r in tm:
        temp = zero
        for j,rj in enumerate(r):
            temp += unit[j]*mvp.const(unit[j],rj)
        mt.append(temp)
    return mt

# mvp for thin-lens sextupole
def thinSext(K2L,unit):
    '''
    thinlens sext with given K2L
    here nv = 4 temporially
    '''
    return [unit[0],unit[1]+mvp([[2,0,0,0],[0,0,2,0]],[-K2L/2,K2L/2]),
            unit[2],unit[3]+mvp([[1,0,1,0]],[K2L])]

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

def thickSext(L,K2,a,nsl=4,nv=4,zero=None,unit=None,one=None):
    '''
    combine thin-lens sext and drift to thick sextupole
    '''
    if not zero:
        unit,zero,one = const(nv=nv)
    Ld = float(L)/2/nsl
    dtm = np.eye(nv)
    dtm[0,1],dtm[2,3] = Ld,Ld
    md = sqmat_tm(dtm,nv,a,zero=zero,unit=unit)
    ms = sqmat_thinsext(K2*L/nsl,a,unit=unit,one=one)
    m = sps.eye(md.shape[0])
    for i in xrange(nsl):
        m = md*ms*md*m
    return m
