# cuda_ANN

After clone this repository to your computer, run this command:

```bash
make all
```

Then the project will generate a brute_nn file which is executable, and execute this file.
And you will get a output from your shell like this:

```bash
cd ./gpu_ann_ivf && mkdir -p build
cd ./gpu_ann_ivf/build && cmake ../
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /home/xein/CODE/CUDACODE/cuda_ANN/gpu_ann_ivf/build
cd ./gpu_ann_ivf/build && make
make[1]: 进入目录“/home/xein/CODE/CUDACODE/cuda_ANN/gpu_ann_ivf/build”
[100%] Built target brute_nn
make[1]: 离开目录“/home/xein/CODE/CUDACODE/cuda_ANN/gpu_ann_ivf/build”
cd ./gpu_ann_ivf && ./build/brute_nn
N = 100000, dim = 128
running CPU brute force NN (ground truth)...
CPU nn index: 0, time = 39.450 ms
Training IVF centroids (nlist=256, iters=5)...
k-means iter 1 / 5 done.
k-means iter 2 / 5 done.
k-means iter 3 / 5 done.
k-means iter 4 / 5 done.
k-means iter 5 / 5 done.
IVF lists built: total=100000, avg per list=390.62
Selecting candidates with IVF (nprobe=16)...
Candidates: 6300 (6.30% of DB)
Running GPU IVF-Flat nn kernel...
GPU IVF nn index: 0, kernel time = 0.149 ms
✅ IVF + GPU result matches CPU ground truth (with this setting).
```
