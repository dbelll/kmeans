import cuda_kmeans_tri as kmt
for nClusters in [512, 513]:
    for nPts in [2, 12, 15, 16, 17, 512, 513, 1023, 1024]:
        for nReps in [2]:
            kmt.quiet_run(10, nPts, 2, nClusters, nReps)

