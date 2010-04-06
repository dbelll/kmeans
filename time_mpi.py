from ctypes import c_int, c_double, c_uint
from numpy.ctypeslib import ndpointer
import numpy as N
from numpy import empty,array,reshape,arange
import time

def kmeans(X, nclst, maxiter=0, numruns=1):
    """Wrapper for Peter Gehlers accelerated MPI-Kmeans routine."""
    
    mpikmeanslib = N.ctypeslib.load_library("libmpikmeans.so", ".")
    mpikmeanslib.kmeans.restype = c_double
    mpikmeanslib.kmeans.argtypes = [ndpointer(dtype=c_double, ndim=1, flags='C_CONTIGUOUS'), \
                                    ndpointer(dtype=c_double, ndim=1, flags='C_CONTIGUOUS'), \
                                    ndpointer(dtype=c_uint, ndim=1, flags='C_CONTIGUOUS'), \
                                    c_uint, c_uint, c_uint, c_uint, c_uint ]
    
    npts,dim = X.shape
    assignments=empty( (npts), c_uint )
    
    bestSSE=N.Inf
    bestassignments=empty( (npts), c_uint)
    Xvec = array( reshape( X, (-1,) ), c_double )
    permutation = N.random.permutation( range(npts) ) # randomize order of points
    CX = array(X[permutation[:nclst],:], c_double).flatten()
    SSE = mpikmeanslib.kmeans( CX, Xvec, assignments, dim, npts, min(nclst, npts), maxiter, numruns)
    return reshape(CX, (nclst,dim)), SSE, (assignments+1)


def quiet_runs(nPts_list, nDim_list, nClusters_list, nRep_list):
    # quiet_runs(nTest_list, nPts_list, nDim_list, nClusters_list [, print_it]):
    # when number of tests is -1, it will be calculated based on the size of the problem
    print "mpi_kmeans timing runs"
    for pts in nPts_list:
        for dim in nDim_list:
            if dim >= pts:
                continue
            for clst in nClusters_list:
                if clst >= pts:
                    continue
                for rep in nRep_list:
                    data = N.random.rand(pts, dim).astype(N.float32)
                    t1 = time.time()
                    kmeans(data, clst, rep, 0)
                    t2 = time.time()
                    print "[MPIKMEANS]({0:8},{1:5},{2:5},{3:5})...".format(pts, dim, clst, rep),
                    print 1000.*(t2-t1)
                    

quiet_runs([100, 1000, 10000, 100000], [4, 20, 100], [5, 15, 45], [4, 12, 36])
