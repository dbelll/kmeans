import cuda_kmeans as ckm
for nClusters in [512, 513]:
    for nPts in [2, 12, 15, 16, 17, 512, 513, 1023, 1024]:
        for nReps in [2]:
            ckm.quiet_run(10, nPts, 2, nClusters, nReps)

