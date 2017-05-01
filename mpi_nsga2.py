import numpy as np
import string
import h5py
from mpi4py import MPI
import nsga2

#import c#a
import e
import nsls2sr_supercell_ch77_20150406 as nsls2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def mpi_controller(task_list):
    ''' 
    Controls the distribution of data-sets to the nodes
    '''
    result = np.zeros_like(task_list)
    process_list = range(1, size)

    njobs = len(task_list)
    nsend = 0
    nrecv = 0

    for i in process_list:
        if nsend < njobs:
            data = task_list[nsend]
            comm.send(data, tag=nsend, dest=i)
            nsend += 1
        else:
            comm.send(data, tag=99999, dest=i)

    while nsend < njobs:
        status = MPI.Status()
        d = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        result[nrecv] = d
        nrecv += 1
        data = task_list[nsend]
        comm.send(data, tag=nsend, dest=status.source)
        nsend += 1

    while nrecv < njobs:
        status = MPI.Status()
        d = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        result[nrecv] = d
        nrecv += 1
        data = 0
        comm.send(data, tag=99999, dest=status.source)
    return result

def mpi_worker(f,nvar,nobj):
    '''
    Worker process
    '''
    while True:
        status = MPI.Status()
        d = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.tag == 99999:
            break
        d = f(d,nvar,nobj)
        comm.send(d, dest=0, tag=status.tag)
        print('%4i'%status.tag)


def yu_2d(p,nvar,nobj):
    '''
    linear lattice finding
    '''

    nsls2.sh1.put('K2',p[0])
    nsls2.sh3.put('K2',p[1])
    nsls2.sh4.put('K2',p[2])
    nsls2.sl3.put('K2',p[3])
    nsls2.sl2.put('K2',p[4])
    nsls2.sl1.put('K2',p[5])
    
    nsls2.ring.geth1()
    nsls2.ring.geth2()
    
    p[nvar+0] = abs(nsls2.ring.h1['h30000'])
    p[nvar+1] = abs(nsls2.ring.h1['h21000'])
    p[nvar+2] = abs(nsls2.ring.h1['h10110'])
    p[nvar+3] = abs(nsls2.ring.h1['h10200'])
    p[nvar+4] = abs(nsls2.ring.h1['h10020'])
    p[nvar+5] = abs(nsls2.ring.h1['h20001'])
    p[nvar+6] = abs(nsls2.ring.h1['h10002'])
    p[nvar+7] = abs(nsls2.ring.h1['h00201'])
    
    p[nvar+8] = abs(nsls2.ring.h2['h00310'])
    p[nvar+9] = abs(nsls2.ring.h2['h11200'])
    p[nvar+10] = abs(nsls2.ring.h2['h10111'])
    p[nvar+11] = abs(nsls2.ring.h2['h00112'])
    p[nvar+12] = abs(nsls2.ring.h2['h30001'])
    p[nvar+13] = abs(nsls2.ring.h2['h11110'])
    p[nvar+14] = abs(nsls2.ring.h2['h22000'])
    p[nvar+15] = abs(nsls2.ring.h2['h00004'])
    p[nvar+16] = abs(nsls2.ring.h2['h00400'])
    p[nvar+17] = abs(nsls2.ring.h2['h10201'])
    p[nvar+18] = abs(nsls2.ring.h2['h20020'])
    p[nvar+19] = abs(nsls2.ring.h2['h10021'])
    p[nvar+20] = abs(nsls2.ring.h2['h10003'])
    p[nvar+21] = abs(nsls2.ring.h2['h21001'])
    p[nvar+22] = abs(nsls2.ring.h2['h31000'])
    p[nvar+23] = abs(nsls2.ring.h2['h40000'])
    p[nvar+24] = abs(nsls2.ring.h2['h20002'])
    p[nvar+25] = abs(nsls2.ring.h2['h00220'])
    p[nvar+26] = abs(nsls2.ring.h2['h20200'])
    p[nvar+27] = abs(nsls2.ring.h2['h20110'])
    p[nvar+28] = abs(nsls2.ring.h2['h11002'])
    p[nvar+29] = abs(nsls2.ring.h2['h00202'])

    cd = a.cirDist(p[0:6])
    p[nvar+30] = cd[0] 
    p[nvar+31] = cd[1]
    p[nvar+32] = cd[2] 
    p[nvar+33] = cd[3]

    cons = np.zeros(3)
    cons[0] = 500 - abs(nsls2.ring.h2['h22000'])
    cons[1] = 500 - abs(nsls2.ring.h2['h11110'])
    cons[2] = 500 - abs(nsls2.ring.h2['h00220'])
    p[-1] = sum(cons[np.nonzero(cons<0)])

    return p
'''
def yu_2d_only(p,nvar,nobj):
    # on momentum only
    cd = c.cirDist(p[0:6])
    try:
        p[nvar+0] = cd[0] 
        p[nvar+1] = cd[1]
        p[nvar+2] = cd[2]
        p[nvar+3] = cd[3]
        p[nvar+4] = cd[4]
        p[nvar+5] = cd[5]
        p[nvar+6] = cd[6]
        p[nvar+7] = cd[7]
        p[nvar+8] = cd[8]
        p[nvar+9] = cd[9]

        cons = np.zeros(2)
        cons[0] = 0.05 - abs(cd[6])
        cons[1] = 0.05 - abs(cd[7])
        p[-1] = sum(cons[np.nonzero(cons<0)])
    except:
        p[nvar+0] = 100. 
        p[nvar+1] = 100.
        p[nvar+2] = 100.
        p[nvar+3] = 100.
        p[nvar+4] = 100.
        p[nvar+5] = 100.
        p[nvar+6] = 100.
        p[nvar+7] = 100.
        p[nvar+8] = 100.
        p[nvar+9] = 100.

        p[-1] = 1.

    return p
'''
def yu_2d_only(p,nvar,nobj):
    #off-momentum
    cd = e.cirDist(p[0:6])
    for i in xrange(nobj):
        p[nvar+i] = cd[i]

    cons = np.zeros(2)
    cons[0] = 0.05 - abs(cd[6])
    cons[1] = 0.05 - abs(cd[7])
    p[-1] = sum(cons[np.nonzero(cons<0)])
    return p


def mpi_run():
    problem = {
        'yu_2d':
            {
            'npop': 4000,
            'ngen': 100,
            'nobj': 34,
            'nvar': 6,
            'vran': [[0.,35.],[-35.,0],[-35.,0],[-35.,0],[0.,35.],[-35.,0]],
            'f': yu_2d
            },

        'yu_2d_only':
            {
            'npop': 4000,
            'ngen': 100,
            'nobj': 8,
            'nvar': 6,
            'vran': [[0.,35.],[-35.,0],[-35.,0],[-35.,0],[0.,35.],[-35.,0]],
            'f': yu_2d_only
            },
        }

    def def_problem(v):
        p = problem[v]
        return p['f'], p['npop'], p['ngen'], p['nvar'], p['nobj'], p['vran']
    
    eta_c, eta_m = 20, 20
    f, npop, ngen, nvar, nobj, vran = def_problem('yu_2d_only')

    #== create random parents
    if rank == 0:
        pop = nsga2.nsga2init(npop, ngen, nvar, nobj, vran)
        #pop = np.loadtxt('./data/yu_2d_only_0037.txt')
        pop = mpi_controller(pop)
    else:
        mpi_worker(f,nvar,nobj)

    #== evolve
    for ng in range(ngen):
        if rank == 0:
            print('\n%4i-th generation ...'%(ng+1))
            kd = nsga2.nsga2getChild(pop, nvar, nobj, vran, eta_c, eta_m)
            kd = mpi_controller(kd)
            pop = nsga2.nsga2toursel(pop,kd,nvar)
            #--- save results
            '''
            fname = './data/yu_2d_only_new_'+string.zfill(ng+1,4)+'.txt'
            fid = open(fname,'w')
            m,n = pop.shape
            for i1 in range(m):
                fid.write((n*'%15.6e'+'\n')%tuple(pop[i1]))
            fid.close()
            '''
            fid = h5py.File('data/yu_2d_only_triangler.h5','a')
            fid['%04i'%(ng+1)] = pop
            fid.close()
        else:
            mpi_worker(f,nvar,nobj)

    MPI.Finalize()

if __name__ == "__main__":
    mpi_run()
