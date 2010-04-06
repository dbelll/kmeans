#   CS 292, Fall 2009
#   Final Project
#   Dwight Bell
#--------------------

"""
Verify that each k-means function produces the same results as
the scipy vq algorithm using a variety of problems.

To run from command line:
    python verify.py
    
To run more extensive tests, start python and then:
    import verify
    verify.run_tests()
"""

import numpy as np
import numpy.random as random

try:
    import py_kmeans
    print "py_kmeans found!"
    py_kmeansFlag = 1
except:
    py_kmeansFlag = 0
    print "py_kmeans not found"
    print "run the script ./make_py_kmeans to compile it on resonance"
    
try:
    from scipy.cluster.vq import kmeans, vq, kmeans2
    print "scipy.cluster.vq found!"
    scipyFlag = 1
except ImportError:
    scipyFlag = 0
    print "scipy.cluster.vq not available"

import time
import cuda_kmeans_tri
import cpu_kmeans

VERBOSE = 0
PRINT_TIMES = 0
SEED = 100
SKIP_CPU = 1

def mpi_labels(data, num_clusters, nReps, seed = SEED):
    # reset the random seed so the first cluster assignments will be the same
    # as calculated at the beginning of run_labels()
    random.seed(seed)
    clusters, dist, labels = py_kmeans.kmeans(data, num_clusters, nReps, 0)
    return labels-1, clusters

def scipy_labels(data, clusters, nReps):
    # run scipy.cluster.vq.kmeans on data using an initial clusters
    # number of iterations is one less than used for mpi, since the
    # starting clusters are the result after one mpi iteration
    codebook, dist = kmeans2(data, clusters, nReps, 1e-6)
    labels, dist = vq(data, codebook)
    return labels, codebook

def run_labels(data, nClusters, nReps, skip_cpu = SKIP_CPU, seed=SEED):
    random.seed(seed)
    # run py_kmeans.kmeans once to get a starting label assignment,
    # which will be used by the scipy routine and others
    clusters, dist, labels = py_kmeans.kmeans(data, nClusters, 1, 0)
    if VERBOSE:
        print "data"
        print data
        print "initial clusters:"
        print clusters
 
    (nPts, nDim) = data.shape
    nClusters = clusters.shape[0] 
    print "[nPts:{0:6}][nDim:{1:4}][nClusters:{2:4}][nReps:{3:3}]...".format(nPts, nDim, 
                                                                            nClusters, nReps),

    data2 = np.swapaxes(data, 0, 1).astype(np.float32).copy('C')
    clusters2 = np.swapaxes(clusters, 0, 1).astype(np.float32).copy('C')

    if VERBOSE:
        print "data2"
        print data2
        print "clusters2"
        print clusters2

    """
    t1 = time.time()
    (cuda_clusters, cuda_labels) = cuda_kmeans.kmeans_gpu(data2, clusters2, nReps+1)
    if VERBOSE:
        print "cuda_kmeans labels:"
        print cuda_labels
    t2 = time.time()
    if PRINT_TIMES:
        print "\ncuda ", t2-t1
    """
    
    t1 = time.time()
    (tri_clusters, tri_labels) = cuda_kmeans_tri.trikmeans_gpu(data2, clusters2, nReps+1)
    if VERBOSE:
        print "\ncuda_kmeans_tri labels:"
        print tri_labels
    t2 = time.time()
    if PRINT_TIMES:
        print "\ntri  ", t2-t1

    t1 = time.time()
    labels_mpi = mpi_labels(data, nClusters, nReps+1, seed)
    if VERBOSE:
        print "mpi labels:"
        print labels_mpi[0]
    t2 = time.time()
    if PRINT_TIMES:
        print "mpi  ", t2-t1

    if scipyFlag:
        t1 = time.time()
        labels_scipy = scipy_labels(data, clusters, nReps)
        if VERBOSE:
            print "scipy labels:"
            print labels_scipy[0]
        t2 = time.time()
        if PRINT_TIMES:
            print "scipy", t2-t1
    
    t1 = time.time()
    if not skip_cpu:
        (cpu_clusters, cpu_labels) = cpu_kmeans.kmeans_cpu(data2, clusters2, nReps+1)
        if VERBOSE:
            print "cpu_kmeans labels:"
            print cpu_labels
        t2 = time.time()
        if PRINT_TIMES:
            print "cpu  ", t2-t1

    error = 0
    
    if scipyFlag:
        try:
            np.testing.assert_array_equal(labels_mpi[0], labels_scipy[0])
        except AssertionError:
            print "mpi<>scipy",
            error = 1
    
    if not skip_cpu:
        try:
            np.testing.assert_array_equal(labels_mpi[0], cpu_labels)
        except AssertionError:
            print "mpi<>cpu",
            error = 1
    
        try:
            np.testing.assert_array_equal(tri_labels, cpu_labels)
        except AssertionError:
            print "tri<>cpu",
            error = 1

    """
    try:
        np.testing.assert_array_equal(cuda_labels, tri_labels)
    except AssertionError:
        print "cuda<>tri",
        error = 1
    """
    
    try:
        np.testing.assert_array_equal(labels_mpi[0], tri_labels)
    except AssertionError:
        print "tri<>mpi",
        error = 1


    if error == 0:
        print "Labels OK ..."
    else:
        print ""
    

def run_tests():
    t1 = time.time()
    print "Testing that all k-means algorithms produce same results..."
    for nReps in [1,10]:
        for nClusters in [3, 30, 120]:
            for nDim in [3, 30]:
                for nPts in [10, 100, 1000, 10000]:
                    if nClusters > nPts: 
                        continue
                    data = random.rand(nPts, nDim)
                    run_labels(data, nClusters, nReps)

    print "Testing complete in ", time.time()-t1, "seconds"

def run_quick(nReps = 4):
    if py_kmeansFlag:
        data = random.rand(1000, 60)
        run_labels(data, 20, nReps)
        data = random.rand(1000, 600)
        run_labels(data, 2, nReps)
        data = random.rand(1000, 6)
        run_labels(data, 200, nReps)
        data = random.rand(10000, 60)
        run_labels(data, 20, nReps)
        data = random.rand(10000, 600)
        run_labels(data, 2, nReps)
        data = random.rand(10000, 6)
        run_labels(data, 200, nReps)
        data = random.rand(30000, 6)
        run_labels(data, 20, nReps)
    
def run_full():
    if py_kmeansFlag:
        SKIP_CPU = 1
        for nPts in [1000, 10000]:
            for nDim in [6, 60, 600]:
                for nClusters in [2, 20, 600]:
                    for nReps in [1, 4, 8, 32]:
                        data = random.rand(nPts, nDim)
                        run_labels(data, nClusters, nReps, 1)

    
if __name__ == '__main__':
    run_quick()

