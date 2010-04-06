#   CS 292, Fall 2009
#   Final Project
#   Dwight Bell
#--------------------

"""
Source modules for PyCuda implementation of
k-means using triangle inequality algorithm.
"""

from pycuda.compiler import SourceModule

import meta_utils as meta


#------------------------------------------------------------------------------------
#                                   source modules
#------------------------------------------------------------------------------------

def get_cuda_module(nDim, nPts, nClusters,
                    blocksize_calc, seqcount_calc, gridsize_calc, 
                    blocksize_calc_part2, useTextureForData, bounds):
    
    modString = """

#define NCLUSTERS         """ + str(nClusters)                    + """
#define NDIM              """ + str(nDim)                         + """
#define NPTS              """ + str(nPts)                         + """

#define THREADS4          """ + str(blocksize_calc)             + """
#define BLOCKS4           """ + str(gridsize_calc)              + """
#define SEQ_COUNT4        """ + str(seqcount_calc)              + """
#define RED_OUT_WIDTH     """ + str(gridsize_calc*nClusters)    + """
#define THREADS4PART2     """ + str(blocksize_calc_part2)        + """

#define BOUNDS     (float)""" + str(bounds)                      + """

texture<float, 2, cudaReadModeElementType>texData;


//-----------------------------------------------------------------------
//                          misc functions
//-----------------------------------------------------------------------


// calculate the distance squared from a data point to a cluster
__device__ float dc_dist(float *data, float *cluster)
{
    float dist = (data[0]-cluster[0]) * (data[0]-cluster[0]);

//------------------------------------------------------------------------
""" + meta.loop(1, nDim, 16, """ 
        dist += (data[{0}*NPTS] - cluster[{0}*NCLUSTERS])
                *(data[{0}*NPTS] - cluster[{0}*NCLUSTERS]);
"""        ) + """
//------------------------------------------------------------------------

    return dist;
}

// calculate the distance squared from a data point to a cluster
__device__ float dc_dist2(float *data, float *cluster)
{
    float dist = (data[0]-cluster[0]) * (data[0]-cluster[0]);
    float *pData = data;
    for(float *pCluster = cluster + NCLUSTERS; 
            pCluster < cluster + NCLUSTERS * NDIM; pCluster += NCLUSTERS){
        pData += NPTS;
        dist +=((*pData) - (*pCluster)) * ((*pData) - (*pCluster));
    }

    return dist;
}


// calculate the distance squared from a data point in texture to a cluster
__device__ float dc_dist_tex(int pt, float *cluster)
{
    float dist = (tex2D(texData, 0, pt)-cluster[0]) * (tex2D(texData, 0, pt)-cluster[0]);
    for(int i=1; i<NDIM; i++){
        float diff = tex2D(texData, i, pt) - cluster[i*NCLUSTERS];
        dist += diff * diff;
    }
    return dist;
}


// calculate the distance squared from a data point in texture to a cluster
__device__ float dc_dist_tex2(int pt, float *cluster)
{
    float dist = (tex2D(texData, 0, pt)-cluster[0]) * (tex2D(texData, 0, pt)-cluster[0]);
    int i = 0;
    for(float *pCluster = cluster + NCLUSTERS;
            pCluster < cluster + NCLUSTERS * NDIM; pCluster += NCLUSTERS){
        i += 1;
        float diff = tex2D(texData, i, pt) - *pCluster;
        dist += diff * diff;
    }
    return dist;
}


//-----------------------------------------------------------------------
//                              assign
//-----------------------------------------------------------------------

// Assign data points to the nearest cluster

"""
    if useTextureForData:
        modString += "__global__ void assign(float *clusters,\n"
    else:
        modString += "__global__ void assign(float *data, float *clusters,\n"
    modString += """
                                int *assignments)
{
""" + meta.copy_to_shared("float", "clusters", "s_clusters", nClusters*nDim) + """

    // calculate distance to each cluster
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= NPTS) return;
    
    // start with cluster 0 as the closest
"""
    if useTextureForData:
        modString += "float min_dist = dc_dist_tex(idx, s_clusters);\n"
    else:
        modString += "float min_dist = dc_dist(data+idx, s_clusters);\n"
    modString += """
    int closest = 0;
    
    for(int c=1; c<NCLUSTERS; c++){
"""
    if useTextureForData:
        modString += "float d = dc_dist_tex2(idx, s_clusters + c);\n"
    else:
        modString += "float d = dc_dist(data + idx, s_clusters + c);\n"
    modString += """
        if(d < min_dist){
            min_dist = d;
            closest = c;
        }
    }
    assignments[idx] = closest;
}



//-----------------------------------------------------------------------
//                                calc
//-----------------------------------------------------------------------

// Calculate the new cluster centers
"""
    if useTextureForData:
        modString += "__global__ void calc(\n"
    else:
        modString += "__global__ void calc(float *data,\n"
    modString += """
                        float *reduction_out, int *reduction_counts, 
                        int *assignments)
{
    __shared__ float s_data[THREADS4];
    __shared__ int s_count[THREADS4];

    int idx = threadIdx.x;
//    int iData = blockIdx.x * THREADS4 * SEQ_COUNT4 + idx;
    
    int dim = blockIdx.y;
    
    for(int c=0; c<NCLUSTERS; c++){
        float tot = 0.0f;
        int count = 0;
        for(int s=0; s<SEQ_COUNT4; s++){
            int iData = blockIdx.x * THREADS4 * SEQ_COUNT4 + s * blockDim.x + idx;
            if(iData >= NPTS) break;
            if(assignments[iData] == c){
                count += 1;
"""
    if useTextureForData:
        modString += "tot += tex2D(texData, dim, iData);\n"
    else:
        modString += "tot += data[dim*NPTS + iData];\n"
    modString += """
            }
        }
        s_data[idx] = tot;
        s_count[idx] = count;
"""

    modString += meta.reduction2("s_data", "s_count", blocksize_calc) + """

        if(idx == 0){
            reduction_out[dim * RED_OUT_WIDTH + blockIdx.x * NCLUSTERS + c] = s_data[0];
            reduction_counts[blockIdx.x * NCLUSTERS + c] = s_count[0];
        }
    }
}


//-----------------------------------------------------------------------
//                           calc_part2
//-----------------------------------------------------------------------

// Calculate new cluster centers using reduction, part 2

__global__ void calc_part2(float *reduction_out, int *reduction_counts,
                            float *new_clusters, float *clusters)
{
    __shared__ float s_data[THREADS4PART2];
    __shared__ int s_count[THREADS4PART2];
    
    int idx = threadIdx.x;
    
    int dim = blockIdx.y;

    for(int c=0; c<NCLUSTERS; c++){
        s_data[idx] = 0.0f;
        s_count[idx] = 0;
        if(idx < BLOCKS4){
            // straight copy of data into shared memory
            s_data[idx] = reduction_out[dim*RED_OUT_WIDTH + idx*NCLUSTERS + c];
            s_count[idx] = reduction_counts[idx*NCLUSTERS + c];
        }
"""

    modString += meta.reduction2("s_data", "s_count", blocksize_calc_part2) + """

        // calculate the new cluster, or copy the old one if has no values or didn't change
        if(idx == 0){
            if(s_count[0] == 0){
                new_clusters[dim * NCLUSTERS + c] = clusters[dim*NCLUSTERS + c];
            }else{
                new_clusters[dim * NCLUSTERS + c] = s_data[0] / s_count[0];
            }
        }
    }
}
    

"""
    #print modString
    return SourceModule(modString)
