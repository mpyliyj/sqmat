import string
from mpi4py import MPI
import h5py,copy
import numpy as np

import apsu as apsu
#import nsls2sr_supercell_ch77_20150406 as nsls2
sexts = apsu.ring.getElements('sext')[::2]
apsu.ring.getChrm(sexts[-2:])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

exitTag = 9999

def mpi_controller(task_list):
    ''' 
    Controls the distribution of data-sets to the nodes
    '''
    result = []
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
            comm.send(data, tag=exitTag, dest=i)

    while nsend < njobs:
        status = MPI.Status()
        d = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        result.append(d)
        nrecv += 1
        data = task_list[nsend]
        comm.send(data, tag=nsend, dest=status.source)
        nsend += 1

    while nrecv < njobs:
        status = MPI.Status()
        d = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        result.append(d)
        nrecv += 1
        data = 0
        comm.send(data, tag=exitTag, dest=status.source)

    return result


def mpi_worker(f):
    '''
    Worker process
    '''
    while True:
        status = MPI.Status()
        fin = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.tag == exitTag:
            break
        fout = f(fin)
        print('tag: %10i'%status.tag)
        comm.send((fin,fout), dest=0, tag=status.tag)


def f(fin):
    sext = nsls2.ring.getElements('sext','sh1')
    sext[0].put('K2',fin[0])
    sext = nsls2.ring.getElements('sext','sh3')
    sext[0].put('K2',fin[1])
    sext = nsls2.ring.getElements('sext','sh4')
    sext[0].put('K2',fin[2])
    sext = nsls2.ring.getElements('sext','sl3')
    sext[0].put('K2',fin[3])
    sext = nsls2.ring.getElements('sext','sl2')
    sext[0].put('K2',fin[4])
    sext = nsls2.ring.getElements('sext','sl1')
    sext[0].put('K2',fin[5])
    nsls2.ring.finddyapsym4(xmin=-0.04,xmax=0.04,nx=161,ymin=0,ymax=0.010,ny=21,nturn=128,dfu=0)
    return nsls2.ring.dyap['dyap']

def f_aps(fin):
    for i,s in enumerate(sexts[:-2]):
        s.put('K2',fin[i])
    apsu.ring.chrom()
    apsu.ring.cchrom1(sexts[-2:],[0.25,0.25])
    #apsu.ring.chrom()
    apsu.ring.finddyapsym4(xmin=-6e-3,xmax=6e-3,nx=61,ymin=1e-8,ymax=5e-3,ny=11,nturn=256,dfu=0)
    return apsu.ring.dyap['dyap']

def mpi_run():
    '''
    
    '''
    if rank == 0:
        fid = h5py.File('./data/apsu_20160916.h5','r')
        pop = copy.deepcopy(np.array(fid['0009']))
        fid.close()

        # --- resort
        if 1:
            k2 = copy.deepcopy(pop[:,10:10+6]) 
            k2av = np.average(k2,axis=0)
            k2w = k2/k2av
            js = np.argsort(np.sum(k2w,axis=1))
            pop = pop[js]

        result = mpi_controller(pop[:1500])
    else:
        mpi_worker(f_aps)

    if rank == 0:
        fid = h5py.File('./data/apsu_20160916_dyap.h5','a')
        for i,d in enumerate(result):
            dsn = string.zfill(i,4)
            grp = fid.create_group(dsn)
            grp['K2'] = d[0]
            grp['DYAP'] = d[1]
        fid.close()

    MPI.Finalize()

if __name__ == "__main__":
    mpi_run()
