import cuda_kmeans_tri as kmt
for nClusters in [2, 4, 8]:
    for nPts in [2000, 4000, 8000]:
        for nDim in [8, 100, 500]:
            for nReps in [8, 16, 32]:
                for nTests in [1, 10]:
                    kmt.quiet_run(nTests, nPts, nDim, nClusters, nReps)

