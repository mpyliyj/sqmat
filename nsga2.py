import math
import random
import numpy as np

def mutate(x,range_x=[0.,1.],eta_m=20.):
    '''
    muataion a real with polynomial probability
    see Deb's book: p124

    input:
    x:        a real
    range_x:  the range of x
    eta_m:    probabilty coefficient of mutation

    return:
    xm:       mutation of x
    '''
    while 1:
        ui = random.random()
        if ui < 0.5:
            deltai = math.pow(2*ui,1./(eta_m+1.))-1.
        else:
            deltai = 1.-math.pow(2*(1.-ui),1./(eta_m+1.))
        xm = x+(range_x[1]-range_x[0])*deltai
        if xm >= range_x[0] and xm <= range_x[1]:
            return xm

def crossover(x1,x2,range_x=[0.,1.],eta_c=20.):
    '''
    crossover with polynomial probability
    see Deb's book: p114

    input:
    x1, x2:   two reals
    range_x:  the range of x1, x2
    eta_c:    probabilty coefficient of cross-over

    return:
    x1t, x2t: off-springs of x1 and x2 after cross-over
    '''
    while 1:
        ui = random.random()
        if ui <= 0.5:
            bqi = math.pow(2*ui,1./(eta_c+1.))
        else:
            bqi = math.pow(0.5/(1.-ui),1./(eta_c+1))
        x1t = 0.5*((1.+bqi)*x1+(1.-bqi)*x2)
        x2t = 0.5*((1.-bqi)*x1+(1.+bqi)*x2)
        if x1t >= range_x[0] and x1t <= range_x[1] \
                and x2t >= range_x[0] and x2t <= range_x[1]: 
            return x1t,x2t


def check_dominance(a,b):
    '''
    Routine for usual non-domination checking

    input:
    a, b:    two listlike arrays (same length)
             last element in a/b (ie. a[-1]/b[-1]) is constraint flag
             if a[-1] == 0, a is within constraints
                a[-1] <  0, a voilate constraints

    return:
     1  if a dominates b
    -1  if b dominates a
     0  if both a and b are non-dominated
    '''
    if (a[-1]<0 and b[-1]<0):
        if (a[-1] > b[-1]):
            return 1
        else:
            if (a[-1] < b[-1]):
                return -1
            else:
                return 0
    else:
        if (a[-1]<0 and b[-1]==0):
            return -1
        else:
            if (a[-1]==0 and b[-1]<0):
                return 1
            else:
                if all(a[:-1] < b[:-1]):
                    return 1
                else:
                    if all(a[:-1] > b[:-1]):
                        return -1
                    else:
                        return 0

def ndsort(x):
    ''' 
    non-dominated sort

    ref: Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan, 
    A Fast Elitist Multiobjective Genetic Algorithm: NSGA-II, 
    IEEE Transactions on Evolutionary Computation 6 (2002),
    No. 2, 182 ~ 197.

    inputs:
    x: a two-dimension array representing a set of fittness
       one row is a solution
       one column is one fittness for all solutions

    returns:
    rank: a list of indexes represent different rank, 
          lower rank dominate higher one
    '''
    n = x.shape[0]
    r0 = []
    entity = np.empty(n,dtype=object)
    for i in range(n):
        a = {'Sp':[],'np':0}
        for j in range(n):
            if i != j:
                dmt = check_dominance(x[i],x[j])
                if dmt == 1:
                    a['Sp'].append(j)
                elif dmt == -1:
                    a['np'] += 1
        if a['np'] == 0:
            r0.append(i)
        entity[i] = a
    rank = [r0]
    i = 0
    while rank[i]:
        Q = []
        for p in rank[i]:
            for q in entity[p]['Sp']:
                entity[q]['np'] -= 1
                if entity[q]['np'] == 0:
                    Q.append(q)
        Q = list(set(Q))
        i += 1
        rank.append(Q)
    #remove last empty rank
    while not rank[-1]:
        rank = rank[:-1]
    return rank


def crowd(x,rank):
    '''
    crowding distance

    inputs:
    x:     two-dimension array
    rank:  index rank from non-dominated sorting

    returns:
    ld:    crowding-distance of each pop
    '''
    m,n = x.shape #Number of fittness
    ld = np.zeros(m)
    for r in rank:
        if len(r) <= 2:
            ld[r] = float('inf')
            continue
        for ni in range(n):
            ind = np.argsort(x[r,ni])
            ld[r[ind[0]]] = float('inf')
            ld[r[ind[-1]]] = float('inf')
            mi = x[r[ind[0]],ni]
            ma = x[r[ind[-1]],ni]
            unitI = ma-mi
            if unitI != 0:
                for i1 in range(1,len(ind)-1):
                    ld[r[ind[i1]]] += (x[r[ind[i1+1]],ni]-x[r[ind[i1-1]],ni])/unitI
    return ld


def nsga2init(npop, ngen, nvar, nobj, vran):
    '''
    initialize NSGA-II
    '''
    pop = np.zeros((npop,nvar+nobj+1))
    for i in range(nvar):
        pop[:,i] = np.random.rand(npop)*(vran[i][1]-vran[i][0])+vran[i][0]
    return pop


def nsga2getChild(pop, nvar, nobj, vran, eta_c=20, eta_m=20):
    alist = range(len(pop))
    matetimes = len(pop)/2
    kd = np.zeros_like(pop)
    for i in range(matetimes):
        p1 = random.choice(alist)
        alist.remove(p1)
        p2 = random.choice(alist)
        alist.remove(p2)
        for ii in range(nvar):
            k1,k2 = crossover(pop[p1,ii],pop[p2,ii],vran[ii],eta_c=eta_c)
            k1 = mutate(k1,vran[ii],eta_m=eta_m)
            k2 = mutate(k2,vran[ii],eta_m=eta_m)
            kd[i*2,ii] = k1
            kd[i*2+1,ii] = k2
    return kd


def nsga2toursel(pop,kd,nvar):
    temp  = np.append(pop,kd,axis=0)
    rank = ndsort(temp[:,nvar:])
    npop = len(pop)
    new_rank = []
    rlen = 0
    for r in rank:
        rlen += len(r)
        if rlen > npop:
            ld = crowd(temp,[r])
            ind = np.argsort(ld)[::-1]
            new_rank += list(ind)
            new_rank = new_rank[0:npop]
            break
        elif rlen < npop:
            new_rank += r
            continue
        else:
            new_rank += r
            break
    return temp[new_rank,:]
