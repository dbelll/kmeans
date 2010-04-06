import cuda_kmeans as ckm
for nClusters in [2, 4, 8]:
    for nPts in [2000, 4000, 8000]:
        for nDim in [8, 100, 500]:
            for nReps in [8, 16, 32]:
                for nTests in [1, 10]:
                    ckm.quiet_run(nTests, nPts, nDim, nClusters, nReps)
ckm.quiet_run(1, 262144, 2, 100, 1)
ckm.quiet_run(1, 262145, 2, 100, 1)
