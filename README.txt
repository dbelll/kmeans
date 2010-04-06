kmeans using PyCUDA
-----------------

cuda_kmeans.py		kmeans on GPU
cuda_kmeans_tri.py	kmeans on GPU, with triangle inequality
cpu_kmeans.py		kmeans on CPU

mods_cuda.py		source modules for cuda_kmeans.py
mods4.py		source modules for cuda_kmeans_tri.py
meta_utils.py		code-writing utilities for writing source modules




Testing scripts
---------------
The quiet_run() function in cuda_kmeans.py and cuda_kmeans_tri.py is used to run tests on a
specific problem size.  quiet_runs() can run many tests.

There are a number of scripts to run tests or to time the calculations:

cudarun_*.py	various test/timing runs using cuda_kmeans.py
run_*.py	various test/timing runs using cuda_kmeans_tri.py

There are some scripts to 'prime' the GPU with the kernels.  This will compile the modules on
the GPU so the timing of subsequent runs using those modules will not include the compilation
time.

cudarun_primer.py	primer routine for cuda_kmeans.py
run_primer.py		primer routine for cuda_kmeans_tri.py

