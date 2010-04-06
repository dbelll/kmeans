#   CS 292, Fall 2009
#   Final Project
#   Dwight Bell
#--------------------

"""
Runs standard k-means clustering using the cpu.

kmeans_cpu(data, clusters, iterations)

return (new_clusters, labels)

data shape is (nDim, nPts) where nDim is # dimensions,
    nPts is # of points
clusters shape is (nDim, nClusters) where nClusters is
    number of clusters
new_clusters will have same shame as clusters
labels will be (nPts), a lable for each point
"""
import numpy as np

#------------------------------------------------------------------------------------
#                               kmeans on the cpu
#------------------------------------------------------------------------------------

BOUNDS = 1.0e-6

def kmeans_cpu(data, clusters, iterations):
    # kmeans_cpu(data, clusters, iterations) returns (clusters, labels)
    
    for i in range(iterations):
        assign = assign_cpu(data, clusters)
        clusters = calc_cpu(data, assign, clusters)
    assign = np.array(assign).reshape(data.shape[1],)
    clusters = np.array(clusters).astype(np.float32)
    return (clusters, assign)
    
def bounded_kmeans_cpu(data, clusters, iterations):
    # kmeans_cpu(data, clusters, iterations) returns (clusters, labels)
    
    for i in range(iterations):
        #assign = bounded_assign_cpu(data, clusters, assign)
        assign = assign_cpu(data,clusters)
        clusters = calc_cpu(data, assign, clusters)
    assign = np.array(assign).reshape(data.shape[1],)
    clusters = np.array(clusters).astype(np.float32)
    return (clusters, assign)
    
def assign_cpu(data, clusters):
    # assign data to the nearest cluster, using cpu
    
    cpu_dist = np.sqrt(((data[:,:,np.newaxis]-clusters[:,np.newaxis,:])**2).sum(0))
    return np.argmin(cpu_dist, 1)

def calc_cpu(data, assign, clusters):
    # calculate new clusters for the data based on assignments

    # calc_cpu(data, assign, clusters)
    # clusters argument is the current clusters
    # returns the recalculated clusters
    
    (nDim, nPts) = data.shape
    nClusters = clusters.shape[1]
    
    c = np.arange(nClusters)
    assign.shape = (nPts, 1)
    c_counts = np.sum(assign == c, axis=0)
    cpu_new_clusters = np.sum(data[:,:,np.newaxis] * (assign==c)[np.newaxis,:,:], axis=1) \
                        / (c_counts + (c_counts == 0))
    cpu_new_clusters = cpu_new_clusters + clusters * (c_counts == 0)[np.newaxis:,]
    return cpu_new_clusters

def bounded_assign_cpu(data, clusters, old_assign):
    # assign data to the nearest cluster if it is more than BOUNDS closer
    nPts = data.shape[1]
    cpu_dist = np.sqrt(((data[:,:,np.newaxis]-clusters[:,np.newaxis,:])**2).sum(0))
    cpu_dist[range(nPts), old_assign] -= BOUNDS
    return np.argmin(cpu_dist, 1)

