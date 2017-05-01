import numpy as np
import string,copy
import h5py
from mpi4py import MPI
import nsga2
import datetime

import j_sls
import sls_12 as sls


WORKTAG = 0
DIETAG = 1

class Work():
    def __init__(self, work_items):
        self.work_items = work_items.tolist()

    def get_next_item(self):
        if len(self.work_items) == 0:
            return None
        return self.work_items.pop()

def master(wi):
    all_data = []
    size = MPI.COMM_WORLD.Get_size()
    current_work = Work(wi) 
    comm = MPI.COMM_WORLD
    status = MPI.Status()

    nsend,nrece = 1,1

    for i in range(1, size): 
        anext = current_work.get_next_item()
        if not anext: break
        comm.send(obj=anext, dest=i, tag=WORKTAG)

        nsend += 1

    while 1:
        anext = current_work.get_next_item()
        if not anext: break
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

        nrece += 1

        all_data.append(data)
        comm.send(obj=anext, dest=status.Get_source(), tag=WORKTAG)
        nsend += 1
    print 'master sent out all data'

    # --- fuck this computer, last job can't recived
    # --- repeat last solution
    for i in range(1,size-1):
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

        nrece += 1

        all_data.append(data)

    print nsend
    print nrece
    #data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
    all_data.append(data)
    print 'master received all data'

    for i in range(1,size):
        comm.send(obj=None, dest=i, tag=DIETAG)

    return np.array(all_data)


def slave(f,nvar,nobj):
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    while 1:
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag(): break
        comm.send(obj=f(data,nvar,nobj), dest=0)
        #print 'slave done'


def yu_2d_only(p,nvar,nobj):
    '''
    cal distortion
    '''
    cd = j_sls.cirDist(p[:nvar])
    for i in xrange(nobj):
        p[nvar+i] = cd[i]

    cons = np.zeros(2)
    cons[0] = 1000-abs(cd[-1])
    cons[1] = 1000-abs(cd[-2])
    p[-1] = sum(cons[np.nonzero(cons<0)])
    return p


def mpi_run():

    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    size = MPI.COMM_WORLD.Get_size() 

    problem = {
        'yu_2d_only':
            {
            'npop': 500,
            'ngen': 100,
            'nobj': 10,
            'nvar': 5,
            'vran': [[0,1000],[-1000,0],[0,1000],[-1000,0],[0,1000]],
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
        #pop = nsga2.nsga2init(npop, ngen, nvar, nobj, vran)

        fid = h5py.File('./data/sls_20161128_8.h5','r')
        last = fid.keys()[-1]
        pop = copy.deepcopy(np.array(fid[last]))
        fid.close()

        pop = master(pop)
    else:
        slave(f,nvar,nobj)

    #== evolve
    for ng in range(ngen):
        if rank == 0:
            print('\n%4i-th generation'%(ng+1))
            print('started at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

            kd = nsga2.nsga2getChild(pop, nvar, nobj, vran, eta_c, eta_m)
            print 'got child' 

            kd = master(kd)
            print 'cal child' 

            pop = nsga2.nsga2toursel(pop,kd,nvar)
            print 'select child' 

            #--- save results
            fid = h5py.File('./data/sls_20161128_9.h5','a')
            fid['%04i'%(ng+1)] = pop
            fid.close()
            print('finished at {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        else:
            slave(f,nvar,nobj)

    MPI.Finalize()

if __name__ == "__main__":
    mpi_run()
