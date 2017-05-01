import numpy as np
import string,datetime,copy
import h5py
from mpi4py import MPI
import nsga2

import i_anhe as g

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
        #print('%4i'%status.tag)

def yu_2d_only(p,nvar,nobj):
    cd = g.cirDist(p[0:9])
    for i in xrange(nobj):
        p[nvar+i] = cd[i]

    cons = np.zeros(2)
    cons[0] = 0.05 - abs(cd[8])
    cons[1] = 0.05 - abs(cd[9])
    p[-1] = sum(cons[np.nonzero(cons<0)])
    return p

def mpi_run():
    problem = {
        'yu_2d_only':
            {
            'npop': 4000,
            'ngen': 100,
            'nobj': 10,
            'nvar': 9,
            #'vran': [[23.,27.],[-15.,-11.],[-16.,-12.],[-29.,-27.],[31.,35.],[-4.,0],[23.,27.],[-15.,-11.],[-16.,-12.]],
            'vran': [[0.,35.],[-35.,0.],[-35.,0.],[-35.,0.],[0.,35.],[-35.,0.],[-2.,2.],[-2.,2.],[-2.,2.]],
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
        # --- starting from previous result
        #fid = h5py.File('./data/anhe_20170203.h5','r')
        #pop = copy.deepcopy(np.array(fid['0030']))
        #fid.close()
        # --- 
        pop = mpi_controller(pop)
    else:
        mpi_worker(f,nvar,nobj)

    #== evolve
    for ng in xrange(0,ngen):
        if rank == 0:
            print('\n%4i-th generation ...'%ng)
            kd = nsga2.nsga2getChild(pop, nvar, nobj, vran, eta_c, eta_m)
            kd = mpi_controller(kd)
            pop = nsga2.nsga2toursel(pop,kd,nvar)
            #--- save results
            fid = h5py.File('data/anhe_20170204.h5','a')
            fid['%04i'%ng] = pop
            fid.close()
            print('finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        else:
            mpi_worker(f,nvar,nobj)

    MPI.Finalize()

if __name__ == "__main__":
    mpi_run()
            
