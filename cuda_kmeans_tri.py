#   CS 292, Fall 2009
#   Final Project
#   Dwight Bell
#--------------------
"""
PyCuda implementation of K-means using the triangle inequality
algorithm presented in the paper "Using the Triangle Inequality
to Accelerate k-Means" by Charles Elkan, Univ Calif, San Diego

Example of running within python:
    >>> import cuda_kmeans_tri as kmt
    >>> import numpy as np
    >>> data = np.random.rand(2,5)
    >>> data
    array([[ 0.89399496,  0.51213574,  0.66063651,  0.76437086,  0.96740785],
           [ 0.11343231,  0.27004973,  0.40700805,  0.955545  ,  0.19054395]])
    >>> clusters = np.random.rand(2,3)
    >>> clusters
    array([[ 0.58353937,  0.04198189,  0.40181198],
           [ 0.02162198,  0.86451144,  0.32205501]])
    >>> (new_clusters, labels) = kmt.trikmeans_gpu(data, clusters, 1)
    >>> labels
    array([0, 2, 2, 1, 0])
    >>> new_clusters
    array([[ 0.93070138,  0.76437086,  0.58638608],
           [ 0.15198813,  0.95554501,  0.33852887]], dtype=float32)
"""

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.reduction import ReductionKernel

from cpu_kmeans import kmeans_cpu
from cpu_kmeans import assign_cpu
from cpu_kmeans import calc_cpu

import numpy as np
import math
import time

import mods4 as kernels

VERBOSE = 0
PRINT_TIMES = 0
SEED = 100

# Set the CPU_SIZE_LIMIT to limit the size of problem that will be calculated on the CPU
# This can be set to a lower value to save time, or a maximum value based on the amount of
# CPU memory
CPU_SIZE_LIMIT = 250000000  #maximum
#CPU_SIZE_LIMIT =   1000000  #only check smaller problems

BOUNDS = np.float32(1.0e-6)

#------------------------------------------------------------------------------------
#                kmeans using triangle inequality algorithm on the gpu
#------------------------------------------------------------------------------------

def trikmeans_gpu(data, clusters, iterations, return_times = 0):
    """trikmeans_gpu(data, clusters, iterations) returns (clusters, labels)
    
    K-means using triangle inequality algorithm and PyCuda
    Input arguments are the data, intial cluster values, and number of iterations to repeat.
    The shape of data is (nDim, nPts) where nDim = # of dimensions in the data and
        nPts = number of data points.
    The shape of clusters is (nDim, nClusters) 
    
    The return values are the updated clusters and labels for the data
    """

    #---------------------------------------------------------------
    #                   get problem parameters
    #---------------------------------------------------------------
    (nDim, nPts) = data.shape
    nClusters = clusters.shape[1]


    #---------------------------------------------------------------
    #            set calculation control variables
    #---------------------------------------------------------------
    useTextureForData = 0
    
    usePageLockedMemory = 0

    if(nPts > 32768):
        useTextureForData = 0
    
    
    # block and grid sizes for the ccdist kernel (also for hdclosest)
    blocksize_ccdist = min(512, 16*(1+(nClusters-1)/16))
    gridsize_ccdist = 1 + (nClusters-1)/blocksize_ccdist
    
    #block and grid sizes for the init module
    threads_desired = 16*(1+(max(nPts, nDim*nClusters)-1)/16)
    #blocksize_init = min(512, threads_desired)
    blocksize_init = min(128, threads_desired)
    gridsize_init = 1 + (threads_desired - 1)/blocksize_init
    
    #block and grid sizes for the step3 module
    blocksize_step3 = blocksize_init
    if not useTextureForData:
        blocksize_step3 = min(256, blocksize_step3)
    gridsize_step3 = gridsize_init
    
    #block and grid sizes for the step4 module
    # Each block of threads will handle seqcount times the data
    # eg blocksize of 512 and seqcount of 4, each block reduces 4*512 = 2048 elements
    blocksize_step4 = 2
    while(blocksize_step4 < min(512, nPts)):
        blocksize_step4 *= 2
    maxblocks = 512
    seqcount_step4 = 1 + (nPts-1)/(blocksize_step4*maxblocks)
    gridsize_step4 = 1 + (nPts-1)/(seqcount_step4*blocksize_step4)
    
    blocksize_step4part2 = 1
    while(blocksize_step4part2 < gridsize_step4):
        blocksize_step4part2 *= 2
    """
    print "blocksize_step4 =", blocksize_step4
    print "gridsize_step4 =", gridsize_step4
    print "seqcount_step4 =", seqcount_step4
    """
    
    #block and grid sizes for the calc_movement module
    for blocksize_calcm in range(32, 512, 32):
        if blocksize_calcm >= nClusters:
            break;
    gridsize_calcm = 1 + (nClusters-1)/blocksize_calcm
    
    #block and grid sizes for the step56 module
    blocksize_step56 = blocksize_init
    gridsize_step56 = gridsize_init
    
    
    #---------------------------------------------------------------
    #                    prepare source modules
    #---------------------------------------------------------------
    t1 = time.time()
    
    mod_ccdist = kernels.get_big_module(nDim, nPts, nClusters,
                                        blocksize_step4, seqcount_step4, gridsize_step4, 
                                        blocksize_step4part2, useTextureForData, BOUNDS)

    ccdist = mod_ccdist.get_function("ccdist")
    calc_hdclosest = mod_ccdist.get_function("calc_hdclosest")
    init = mod_ccdist.get_function("init")
    step3 = mod_ccdist.get_function("step3")
    step4 = mod_ccdist.get_function("step4")
    step4part2 = mod_ccdist.get_function("step4part2")
    calc_movement = mod_ccdist.get_function("calc_movement")
    step56 = mod_ccdist.get_function("step56")
    pycuda.autoinit.context.synchronize()
    t2 = time.time()
    module_time = t2-t1


    #---------------------------------------------------------------
    #                    setup data on GPU
    #---------------------------------------------------------------
    t1 = time.time()

    data = np.array(data).astype(np.float32)
    clusters = np.array(clusters).astype(np.float32)
    
    if useTextureForData:
        # copy the data to the texture
        texrefData = mod_ccdist.get_texref("texData")
        cuda.matrix_to_texref(data, texrefData, order="F")
    else:
        if usePageLockedMemory:
            data_pl = cuda.pagelocked_empty_like(data)
            data_pl[:,:] = data;
            gpu_data = gpuarray.to_gpu(data_pl)
        else:
            gpu_data = gpuarray.to_gpu(data)

    if usePageLockedMemory:
        clusters_pl = cuda.pagelocked_empty_like(clusters)
        clusters_pl[:,:] = clusters
        gpu_clusters = gpuarray.to_gpu(clusters_pl)
    else:
        gpu_clusters = gpuarray.to_gpu(clusters)


    gpu_assignments = gpuarray.zeros((nPts,), np.int32)         # cluster assignment
    gpu_lower = gpuarray.zeros((nClusters, nPts), np.float32)   # lower bounds on distance between 
                                                                # point and each cluster
    gpu_upper = gpuarray.zeros((nPts,), np.float32)             # upper bounds on distance between
                                                                # point and any cluster
    gpu_ccdist = gpuarray.zeros((nClusters, nClusters), np.float32)    # cluster-cluster distances
    gpu_hdClosest = gpuarray.zeros((nClusters,), np.float32)    # half distance to closest
    gpu_hdClosest.fill(1.0e10)  # set to large value // **TODO**  get the acutal float max
    gpu_badUpper = gpuarray.zeros((nPts,), np.int32)   # flag to indicate upper bound needs recalc
    gpu_clusters2 = gpuarray.zeros((nDim, nClusters), np.float32);
    gpu_cluster_movement = gpuarray.zeros((nClusters,), np.float32);
    
    gpu_cluster_changed = gpuarray.zeros((nClusters,), np.int32)
    gpu_cluster_changed.fill(1)
    
    gpu_reduction_out = gpuarray.zeros((nDim, nClusters*gridsize_step4), np.float32)
    gpu_reduction_counts = gpuarray.zeros((nClusters*gridsize_step4,), np.int32)
    
    pycuda.autoinit.context.synchronize()
    t2 = time.time()
    data_time = t2-t1
    
    #---------------------------------------------------------------
    #                    do calculations
    #---------------------------------------------------------------
    ccdist_time = 0.
    hdclosest_time = 0.
    init_time = 0.
    step3_time = 0.
    step4_time = 0.
    step56_time = 0.

    t1 = time.time()
    ccdist(gpu_clusters, gpu_ccdist, gpu_hdClosest,
             block = (blocksize_ccdist, 1, 1),
             grid = (gridsize_ccdist, 1))
    pycuda.autoinit.context.synchronize()
    t2 = time.time()
    ccdist_time += t2-t1
    
    t1 = time.time()
    calc_hdclosest(gpu_ccdist, gpu_hdClosest,
            block = (blocksize_ccdist, 1, 1),
            grid = (gridsize_ccdist, 1))
    pycuda.autoinit.context.synchronize()
    t2 = time.time()
    hdclosest_time += t2-t1
    
    t1 = time.time()
    if useTextureForData:
        init(gpu_clusters, gpu_ccdist, gpu_hdClosest, gpu_assignments, 
                gpu_lower, gpu_upper,
                block = (blocksize_init, 1, 1),
                grid = (gridsize_init, 1),
                texrefs=[texrefData])
    else:
        init(gpu_data, gpu_clusters, gpu_ccdist, gpu_hdClosest, gpu_assignments, 
                gpu_lower, gpu_upper,
                block = (blocksize_init, 1, 1),
                grid = (gridsize_init, 1))
    pycuda.autoinit.context.synchronize()
    t2 = time.time()
    init_time += t2-t1

    for i in range(iterations):
    
        if i>0:
            t1 = time.time()
            ccdist(gpu_clusters, gpu_ccdist, gpu_hdClosest,
                     block = (blocksize_ccdist, 1, 1),
                     grid = (gridsize_ccdist, 1))
            pycuda.autoinit.context.synchronize()
            t2 = time.time()
            ccdist_time += t2-t1
            
            t1 = time.time()
            calc_hdclosest(gpu_ccdist, gpu_hdClosest,
                    block = (blocksize_ccdist, 1, 1),
                    grid = (gridsize_ccdist, 1))
            pycuda.autoinit.context.synchronize()
            t2 = time.time()
            hdclosest_time += t2-t1

            
        t1 = time.time()
        if i > 0:
            gpu_cluster_changed.fill(0)
        if useTextureForData:
            step3(gpu_clusters, gpu_ccdist, gpu_hdClosest, gpu_assignments,
                    gpu_lower, gpu_upper, gpu_badUpper, gpu_cluster_changed,
                    block = (blocksize_step3, 1, 1),
                    grid = (gridsize_step3, 1),
                    texrefs=[texrefData])
        else:
            step3(gpu_data, gpu_clusters, gpu_ccdist, gpu_hdClosest, gpu_assignments,
                    gpu_lower, gpu_upper, gpu_badUpper,  gpu_cluster_changed,
                    block = (blocksize_step3, 1, 1),
                    grid = (gridsize_step3, 1))
        
        pycuda.autoinit.context.synchronize()
        t2 = time.time()
        step3_time += t2-t1
        
        
        t1 = time.time()
        
        if useTextureForData:
            step4(gpu_cluster_changed, gpu_reduction_out,
                gpu_reduction_counts, gpu_assignments,
                block = (blocksize_step4, 1, 1),
                grid = (gridsize_step4, nDim),
                texrefs=[texrefData])
        else:
            step4(gpu_data, gpu_cluster_changed, gpu_reduction_out,
                gpu_reduction_counts, gpu_assignments,
                block = (blocksize_step4, 1, 1),
                grid = (gridsize_step4, nDim))
        
        step4part2(gpu_cluster_changed, gpu_reduction_out, gpu_reduction_counts, 
                gpu_clusters2, gpu_clusters,
                block = (blocksize_step4part2, 1, 1),
                grid = (1, nDim))
        
        calc_movement(gpu_clusters, gpu_clusters2, gpu_cluster_movement, gpu_cluster_changed,
                block = (blocksize_calcm, 1, 1),
                grid = (gridsize_calcm, 1))
        
        pycuda.autoinit.context.synchronize()
        t2 = time.time()
        step4_time += t2-t1
    
        t1 = time.time()
        if useTextureForData:
            step56(gpu_assignments, gpu_lower, gpu_upper, 
                    gpu_cluster_movement, gpu_badUpper,
                    block = (blocksize_step56, 1, 1),
                    grid = (gridsize_step56, 1),
                    texrefs=[texrefData])
        else:
            step56(gpu_assignments, gpu_lower, gpu_upper, 
                    gpu_cluster_movement, gpu_badUpper,
                    block = (blocksize_step56, 1, 1),
                    grid = (gridsize_step56, 1))
        pycuda.autoinit.context.synchronize()
        t2 = time.time()
        step56_time += t2-t1
        
        # prepare for next iteration
        temp = gpu_clusters
        gpu_clusters = gpu_clusters2
        gpu_clusters2 = temp
        
    if return_times:
        return gpu_ccdist, gpu_hdClosest, gpu_assignments, gpu_lower, gpu_upper, \
                gpu_clusters.get(), gpu_cluster_movement, \
                data_time, module_time, init_time, \
                ccdist_time/iterations, hdclosest_time/iterations, \
                step3_time/iterations, step4_time/iterations, step56_time/iterations
    else:
        return gpu_clusters.get(), gpu_assignments.get()





#--------------------------------------------------------------------------------------------
#                           testing functions
#--------------------------------------------------------------------------------------------
    
def run_tests1(nTests, nPts, nDim, nClusters, nReps=1, verbose = VERBOSE, 
                print_times = PRINT_TIMES):
    # run_tests(nTests, nPts, nDim, nClusters, nReps [, verbose [, print_times]]
    # Runs one repition and checks various intermdiate values against a cpu calculation
    
    if nReps > 1:
        print "This method only runs test for nReps == 1"
        return 1
        
    # Generate nPts random data elements with nDim dimensions and nCluster random clusters,
    # then run kmeans for nReps and compare gpu and cpu results.  This is repeated nTests times
    cpu_time = 0.
    gpu_time = 0.
    
    gpu_data_time = 0.
    gpu_module_time = 0.
    gpu_ccdist_time = 0.
    gpu_hdclosest_time = 0.
    gpu_init_time = 0.
    gpu_step3_time = 0.
    gpu_step4_time = 0.
    gpu_step56_time = 0.

    np.random.seed(SEED)
    data = np.random.rand(nDim, nPts).astype(np.float32)
    clusters = np.random.rand(nDim, nClusters).astype(np.float32)

    if verbose:
        print "data"
        print data
        print "\nclusters"
        print clusters

    nErrors = 0

    # repeat this test nTests times
    for iTest in range(nTests):
    
        #run the gpu algorithm
        t1 = time.time()
        (gpu_ccdist, gpu_hdClosest, gpu_assignments, gpu_lower, gpu_upper, \
            gpu_clusters2, gpu_cluster_movement, \
            data_time, module_time, init_time, ccdist_time, hdclosest_time, \
            step3_time, step4_time, step56_time) = \
            trikmeans_gpu(data, clusters, nReps, 1)
        t2 = time.time()        
        gpu_time += t2-t1
        gpu_data_time += data_time
        gpu_module_time += module_time
        gpu_ccdist_time += ccdist_time
        gpu_hdclosest_time += hdclosest_time
        gpu_init_time += init_time
        gpu_step3_time += step3_time
        gpu_step4_time += step4_time
        gpu_step56_time += step56_time
        
        if verbose:
            print "------------------------ gpu results ------------------------"
            print "cluster-cluster distances"
            print gpu_ccdist
            print "half distance to closest"
            print gpu_hdClosest
            print "gpu time = ", t2-t1
            print "gpu_assignments"
            print gpu_assignments
            print "gpu_lower"
            print gpu_lower
            print "gpu_upper"
            print gpu_upper
            print "gpu_clusters2"
            print gpu_clusters2
            print "-------------------------------------------------------------"
            

        # check ccdist and hdClosest
        ccdist = np.array(gpu_ccdist.get())
        hdClosest = np.array(gpu_hdClosest.get())
        
        t1 = time.time()
        cpu_ccdist = 0.5 * np.sqrt(((clusters[:,:,np.newaxis]-clusters[:,np.newaxis,:])**2).sum(0))
        t2 = time.time()
        cpu_ccdist_time = t2-t1
        
        if verbose:
            print "cpu_ccdist"
            print cpu_ccdist
        
        error = np.abs(cpu_ccdist - ccdist)
        if np.max(error) > 1e-7 * nDim * 2:
            print "iteration", iTest,
            print "***ERROR*** max ccdist error =", np.max(error)
            nErrors += 1
        if verbose:
            print "average ccdist error =", np.mean(error)
            print "max ccdist error     =", np.max(error)
        
        t1 = time.time()
        cpu_ccdist[cpu_ccdist == 0.] = 1e10
        good_hdClosest = np.min(cpu_ccdist, 0)
        t2 = time.time()
        cpu_hdclosest_time = t2-t1
        
        if verbose:
            print "good_hdClosest"
            print good_hdClosest
        err = np.abs(good_hdClosest - hdClosest)
        if np.max(err) > 1e-7 * nDim:
            print "***ERROR*** max hdClosest error =", np.max(err)
            nErrors += 1
        if verbose:
            print "errors on hdClosest"
            print err
            print "max error on hdClosest =", np.max(err)
    
    
        # calculate cpu initial assignments
        t1 = time.time()
        cpu_assign = assign_cpu(data, clusters)
        t2 = time.time()
        cpu_assign_time = t2-t1
        
        if verbose:
            print "assignments shape =", cpu_assign.shape
            print "data shape =", data.shape
            print "cpu assignments"
            print cpu_assign
            print "gpu assignments"
            print gpu_assignments
            print "gpu new clusters"
            print gpu_clusters2
            
        differences = sum(gpu_assignments.get() - cpu_assign)
        if(differences > 0):
            nErrors += 1
            print differences, "errors in initial assignment"
        else:
            if verbose:
                print "initial cluster assignments match"
    
        # calculate the number of data points in each cluster
        c = np.arange(nClusters)
        c_counts = np.sum(cpu_assign.reshape(nPts,1) == c, axis=0)

        # calculate cpu new cluster values:
        t1 = time.time()
        cpu_new_clusters = calc_cpu(data, cpu_assign, clusters)
        t2 = time.time()
        cpu_calc_time = t2-t1
        
        if verbose:
            print "cpu new clusters"
            print cpu_new_clusters
        
        diff = np.max(np.abs(gpu_clusters2 - cpu_new_clusters))
        if diff > 1e-7 * max(c_counts) or math.isnan(diff):
            iDiff = np.arange(nClusters)[((gpu_clusters2 - cpu_new_clusters)**2).sum(0) > 1e-7]
            print "clusters that differ:"
            print iDiff
            nErrors += 1
            if verbose:
                print "Test",iTest,"*** ERROR *** max diff was", diff
                for x in iDiff:
                    print "\ndata for cluster ",x
                    print "gpu:"
                    print gpu_clusters2[:,x]
                    print "cpu:"
                    print cpu_new_clusters[:,x]
                    print "points assigned:"
                    for ii in range(nPts):
                        if cpu_assign[ii] == x:
                            print "data point #",ii
                            print data[:,ii]
        else:
            if verbose:
                print "Test", iTest, "OK"
        
        #check if the cluster movement values are correct
        cpu_cluster_movement = np.sqrt(((clusters - cpu_new_clusters)**2).sum(0))
        diff = np.max(np.abs(cpu_cluster_movement - gpu_cluster_movement.get()))
        if diff > 1e-6 * nDim:
            print "*** ERROR *** max cluster movement error =", diff
            nErrors += 1
        if verbose:
            print "cpu cluster movements"
            print cpu_cluster_movement
            print "gpu cluster movements"
            print gpu_cluster_movement
            print "max diff in cluster movements is", diff
        
        cpu_time = cpu_assign_time + cpu_calc_time
    

    if print_times:
        print "\n---------------------------------------------"
        print "nPts      =", nPts
        print "nDim      =", nDim
        print "nClusters =", nClusters
        print "nReps     =", nReps
        print "average cpu time (ms) =", cpu_time/nTests*1000.
        print "     assign time (ms) =", cpu_assign_time/nTests*1000.
        if nReps == 1:
            print "       calc time (ms) =", cpu_calc_time/nTests*1000.
            print "average gpu time (ms) =", gpu_time/nTests*1000.
        else:
            print "       calc time (ms) ="
            print "average gpu time (ms) ="
        print "       data time (ms) =", gpu_data_time/nTests*1000.
        print "     module time (ms) =", gpu_module_time/nTests*1000.
        print "       init time (ms) =", gpu_init_time/nTests*1000.        
        print "     ccdist time (ms) =", gpu_ccdist_time/nTests*1000.        
        print "  hdclosest time (ms) =", gpu_hdclosest_time/nTests*1000.        
        print "      step3 time (ms) =", gpu_step3_time/nTests*1000.        
        print "      step4 time (ms) =", gpu_step4_time/nTests*1000.        
        print "     step56 time (ms) =", gpu_step56_time/nTests*1000.        
        print "---------------------------------------------"

    return nErrors



def verify_assignments(gpu_assign, cpu_assign, data, gpu_clusters, cpu_clusters, verbose = 0, 
                        iTest = -1): 
    # check that assignments are equal

    """
    print "verify_assignments"
    print "gpu_assign", gpu_assign, "is type", type(gpu_assign)
    print "gpu_assign", cpu_assign, "is type", type(cpu_assign)
    """
    differences = sum(gpu_assign != cpu_assign)
    # print "differences =", differences
    error = 0
    if(differences > 0):
        error = 1
        if verbose:
            if iTest >= 0:
                print "Test", iTest,
            print "*** ERROR ***", differences, "differences"
            iDiff = np.arange(gpu_assign.shape[0])[gpu_assign != cpu_assign]
            print "iDiff", iDiff
            for ii in iDiff:
                print "data point is", data[:,ii]
                print "cpu assigned to", cpu_assign[ii]
                print "   with center at (cpu)", cpu_clusters[:,cpu_assign[ii]]
                print "   with center at (gpu)", gpu_clusters[:,cpu_assign[ii]]
                print "gpu assigned to", gpu_assign[ii]
                print "   with center at (cpu)", cpu_clusters[:,gpu_assign[ii]]
                print "   with center at (gpu)", gpu_clusters[:, gpu_assign[ii]]
                print ""
                print "cpu calculated distances:"
                print "   from point", ii, "to:"
                print "      cluster", cpu_assign[ii], "is", np.sqrt(np.sum((data[:,ii]-
                                                            cpu_clusters[:,cpu_assign[ii]])**2))
                print "      cluster", gpu_assign[ii], "is", np.sqrt(np.sum((data[:,ii]-
                                                            cpu_clusters[:,gpu_assign[ii]])**2))
                print "gpu calculated distances:"
                print "   from point", ii, "to:"
                print "      cluster", cpu_assign[ii], "is", np.sqrt(np.sum((data[:,ii]-
                                                            gpu_clusters[:,cpu_assign[ii]])**2))
                print "      cluster", gpu_assign[ii], "is", np.sqrt(np.sum((data[:,ii]-
                                                            gpu_clusters[:,gpu_assign[ii]])**2))
    else:
        if verbose:
            if iTest >= 0:
                print "Test", iTest,
            print "Cluster assignment is OK"
    return error


def verify_clusters(gpu_clusters, cpu_clusters, cpu_assign, verbose = 0, iTest = -1):
    # check that clusters are equal
    error = 0
    
    # calculate the number of data points in each cluster
    nPts = cpu_assign.shape[0]
    nClusters = cpu_clusters.shape[1]
    c = np.arange(nClusters)
    c_counts = np.sum(cpu_assign.reshape(nPts,1) == c, axis=0)
    
    err = np.abs(gpu_clusters - cpu_clusters)
    diff = np.max(err)
    
    if verbose:
        print "max error in cluster centers is", diff
        print "avg error in cluster centers is", np.mean(err)
    
    allowable_diff = max(c_counts) * 1e-7
    if diff > allowable_diff or math.isnan(diff):
        error = 1
        iDiff = np.arange(nClusters)[((gpu_clusters - cpu_clusters)**2).sum(0) > allowable_diff]
        if verbose:
            print "clusters that differ:"
            print iDiff
            if iTest >= 0:
                print "Test",iTest,
            print "*** ERROR *** max diff was", diff
            for cc in iDiff:
                print "cluster", cc
                print "gpu"
                print gpu_clusters[:,cc]
                print "cpu"
                print cpu_clusters[:,cc]
    else:
        if verbose:
            if iTest >= 0:
                print "Test", iTest,
            print "Clusters are OK"
        
    return error


def run_tests(nTests, nPts, nDim, nClusters, nReps=1, verbose = VERBOSE, print_times = PRINT_TIMES,
                 verify = 1):
    # run_tests(nTests, nPts, nDim, nClusters, nReps [, verbose [, print_times]]
    
    # Generate nPts random data elements with nDim dimensions and nCluster random clusters,
    # then run kmeans for nReps and compare gpu and cpu results.  This is repeated nTests times
    
    if(nPts * nDim *nClusters > CPU_SIZE_LIMIT):
        #print "Too big to verify wiht cpu calculation"
        verify = 0  # too big to run on cpu
        
    cpu_time = 0.
    gpu_time = 0.
    
    gpu_data_time = 0.
    gpu_module_time = 0.
    gpu_ccdist_time = 0.
    gpu_hdclosest_time = 0.
    gpu_init_time = 0.
    gpu_step3_time = 0.
    gpu_step4_time = 0.
    gpu_step56_time = 0.


    nErrors = 0

    # repeat this test nTests times
    for iTest in range(nTests):
    
        np.random.seed(SEED+iTest)
        data = np.random.rand(nDim, nPts).astype(np.float32)
        clusters = np.random.rand(nDim, nClusters).astype(np.float32)

        if verbose:
            print "data"
            print data
            print "\nclusters"
            print clusters

        if verify:
            #run the cpu algorithm
            t1 = time.time()
            (cpu_clusters, cpu_assign) = kmeans_cpu(data, clusters, nReps)
            cpu_assign.shape = (nPts,)
            t2 = time.time()
            cpu_time += t2-t1
            
            if verbose:
                print "------------------------ cpu results ------------------------"
                print "cpu_assignments"
                print cpu_assign
                print "cpu_clusters"
                print cpu_clusters
                print "-------------------------------------------------------------"
        
        #run the gpu algorithm
        t1 = time.time()
        (gpu_ccdist, gpu_hdClosest, gpu_assign, gpu_lower, gpu_upper, \
            gpu_clusters, gpu_cluster_movement, \
            data_time, module_time, init_time, ccdist_time, hdclosest_time, \
            step3_time, step4_time, step56_time) = \
            trikmeans_gpu(data, clusters, nReps, 1)
        t2 = time.time()        
        gpu_time += t2-t1
        gpu_data_time += data_time
        gpu_module_time += module_time
        gpu_ccdist_time += ccdist_time
        gpu_hdclosest_time += hdclosest_time
        gpu_init_time += init_time
        gpu_step3_time += step3_time
        gpu_step4_time += step4_time
        gpu_step56_time += step56_time
        
        if verbose:
            print "------------------------ gpu results ------------------------"
            print "gpu_assignments"
            print gpu_assign
            print "gpu_clusters"
            print gpu_clusters
            print "-------------------------------------------------------------"
            

        if verify:
            # calculate the number of data points in each cluster
            c = np.arange(nClusters)
            c_counts = np.sum(cpu_assign.reshape(nPts,1) == c, axis=0)

            # verify the results...
            err = verify_assignments(gpu_assign.get(), cpu_assign, data, gpu_clusters, 
                                            cpu_clusters, verbose, iTest)
            err += verify_clusters(gpu_clusters, cpu_clusters, cpu_assign, verbose, iTest)
            if err:
                nErrors += 1

    if print_times:
        print "\n---------------------------------------------"
        print "nPts      =", nPts
        print "nDim      =", nDim
        print "nClusters =", nClusters
        print "nReps     =", nReps
        if verify:
            print "average cpu time (ms) =", cpu_time/nTests*1000.
        else:
            print "average cpu time (ms) = N/A"
        print "average gpu time (ms) =", gpu_time/nTests*1000.
        print "       data time (ms) =", gpu_data_time/nTests*1000.
        print "     module time (ms) =", gpu_module_time/nTests*1000.
        print "       init time (ms) =", gpu_init_time/nTests*1000.        
        print "     ccdist time (ms) =", gpu_ccdist_time/nTests*1000.        
        print "  hdclosest time (ms) =", gpu_hdclosest_time/nTests*1000.        
        print "      step3 time (ms) =", gpu_step3_time/nTests*1000.        
        print "      step4 time (ms) =", gpu_step4_time/nTests*1000.        
        print "     step56 time (ms) =", gpu_step56_time/nTests*1000.        
        print "---------------------------------------------"

    if verify:
        return nErrors
    else:
        return -1


#----------------------------------------------------------------------------------------
#                           multi-tests
#----------------------------------------------------------------------------------------

def quiet_run(nTests, nPts, nDim, nClusters, nReps, ptimes = PRINT_TIMES, verify = 1):
    # quiet_run(nTests, nPts, nDim, nClusters, nReps [, ptimes]):
    print "[TEST]({0:3},{1:8},{2:5},{3:5}, {4:5})...".format(nTests, nPts, nDim, nClusters, nReps),
    try:
        result =  run_tests(nTests, nPts, nDim, nClusters, nReps, 0, ptimes, verify)
        if result == 0:
            if verify:
                print "OK"
            else:
                print ""
        else:
            if result < 0:
                print "(not checked)"
            else:
                print "*** ERROR *** ({0} of {1})".format(result, nTests)
    except cuda.LaunchError:
        print "launch error"
    
def quiet_runs(nTest_list, nPts_list, nDim_list, nClusters_list, nRep_list, print_it = PRINT_TIMES, 
                verify = 1):
    # quiet_runs(nTest_list, nPts_list, nDim_list, nClusters_list [, print_it]):
    # when number of tests is -1, it will be calculated based on the size of the problem
    for t in nTest_list:
        for pts in nPts_list:
            for dim in nDim_list:
                if dim >= pts:              # skip if dimensions is greater than number of points
                    continue
                for clst in nClusters_list:
                    if clst >= pts:     # skip if clusters are more than half the number of points
                        continue
                    for rep in nRep_list:
                        if t < 0:
                            tt = max(1, min(10, 10000000/(pts*dim*clst)))
                        else:
                            tt = t
                        quiet_run(tt, pts, dim, clst, rep, print_it, verify);

def run_all(pFlag = 1):
    quiet_run(1, 100, 3, 3, 2, ptimes = pFlag)
    quiet_run(1, 100, 6, 3, 2, ptimes = pFlag)
    quiet_run(1, 100, 12, 3, 2, ptimes = pFlag)
    quiet_run(1, 100, 3, 3, 2, ptimes = pFlag)
    quiet_run(1, 100, 6, 3, 2, ptimes = pFlag)
    quiet_run(1, 100, 12, 3, 2, ptimes = pFlag)
    quiet_run(1, 100, 3, 3, 2, ptimes = pFlag)
    quiet_run(1, 100, 6, 3, 2, ptimes = pFlag)
    quiet_run(1, 100, 12, 3, 2, ptimes = pFlag)
    quiet_run(1, 10000, 60, 20, 1, ptimes = pFlag)
    quiet_run(1, 10000, 600, 5, 1, ptimes = pFlag)
    quiet_run(1, 10000, 5, 600, 1, ptimes = pFlag)
    quiet_run(1, 1000, 600, 50, 1, ptimes = pFlag)      # clusters too big for shared memory
    quiet_run(1, 1000, 50, 600, 1, ptimes = pFlag)      # clusters too big for shared memory
    quiet_run(1, 100000, 60, 20, 1, ptimes = pFlag)
    quiet_run(1, 10000, 60, 20, 2, ptimes = pFlag)
    quiet_run(1, 10000, 600, 5, 2, ptimes = pFlag)
    quiet_run(1, 10000, 5, 600, 2, ptimes = pFlag)
    quiet_run(1, 100000, 60, 20, 2, ptimes = pFlag)

def run_reps(pFlag = 1):
    quiet_run(1, 10, 4, 3, 5, ptimes = pFlag)
    quiet_run(1, 1000, 60, 20, 5, ptimes = pFlag)
    quiet_run(1, 50000, 60, 20, 5, ptimes = pFlag)
    quiet_run(1, 10000, 600, 5, 5, ptimes = pFlag)
    quiet_run(1, 10000, 5, 600, 5, ptimes = pFlag)
    
def timings(t = 1, v = 0):
    # run a bunch of tests with optional timing
    quiet_runs([1], [100, 1000, 10000, 100000], [4, 20, 100, 500], [5, 15, 45, 135], [4, 8, 16, 32],
                         t, v)
    
def prime(t = 0, v = 0):
    # run each test once to get the module compiled and on the gpu
    quiet_runs([1], [100, 1000, 10000, 100000], [4, 20, 100, 500], [5, 15, 45, 135], [1], t, v)

def big_timings(t = 1, v = 0):
    # run a bunch of tests with optional timing
    quiet_runs([1], [1000000], [4, 20, 100, 500], [5, 15, 45, 135], [4, 8, 16, 32], t, v)
    
def big_prime(t = 0, v = 0):
    # run each test once to get the module compiled and on the gpu
    quiet_runs([1], [1000000], [4, 20, 100, 500], [5, 15, 45, 135], [1], t, v)


def quickTimes(nReps = 5):
    # quick check of timing values
    if quickRun() > 0:
        print "***ERROR***"
    else:
        quiet_run(3, 1000, 60, 20, nReps, 1)
        quiet_run(3, 1000, 600, 2, nReps, 1)
        quiet_run(3, 1000, 60, 200, nReps, 1)
        quiet_run(3, 10000, 60, 20, nReps, 1)
        quiet_run(3, 10000, 600, 2, nReps, 1)
        quiet_run(3, 10000, 6, 200, nReps, 1)
        quiet_run(3, 10000, 1000, 100, nReps, 1)
        quiet_run(3, 30000, 6, 20, nReps, 1)

def quickRun():
    # run to make sure answers have not changed
    nErrors = run_tests1(1, 1000, 6, 2, 1)
    print nErrors
    nErrors += run_tests1(1, 1000, 600, 2, 1)
    print nErrors
    nErrors += run_tests1(1, 1000, 6, 200, 1)
    print nErrors
    nErrors += run_tests1(1, 10000, 60, 20, 1)
    return nErrors
    
if __name__ == '__main__':
    print quickRun()

