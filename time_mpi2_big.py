import py_kmeans
import numpy as N
import time


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
                    data = N.random.rand(pts, dim)
                    t1 = time.time()
                    py_kmeans.kmeans(data, clst, rep, 0)
                    t2 = time.time()
                    print "[MPIKMEANS]({0:8},{1:5},{2:5},{3:5})...".format(pts, dim, clst, rep),
                    print 1000.*(t2-t1)
                    
quiet_runs([1000000], [4, 20, 100, 500], [5, 15, 45, 135], [4, 8, 16, 32])
