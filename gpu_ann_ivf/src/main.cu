#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cfloat>
#include <chrono>

using namespace std;

#define CUDA_CHECK(expr)\
    do{\
        cudaError_t err = (expr); \
        if (err != cudaSuccess){\
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", \
                #expr, __FILE__, __LINE__, cudaGetErrorString(err));\
            std::exit(EXIT_FAILURE); \
        }\
    }while(0)

void fill_random(vector<float>& vec){
    mt19937 rng(42);
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    for(auto &x: vec) x = dist(rng);
}

int cpu_bruteforce_nn(const vector<float>& db, const vector<float>& query, int N, int dim){
    float best_dist = numeric_limits<float>:: max();
    int best_idx = -1;

    for(int i = 0; i<N; i++){
        float dist = 0.0f;
        const float* x = &db[i*dim];
        for(int d = 0; d<dim; d++){
            float diff = x[d] - query[d];
            dist += diff * diff;
        }
        if (dist<best_dist){
            best_dist = dist;
            best_idx = i;
        }
    }

    return best_idx;
}

// =================== IVF 结构定义 ===================
// inverted file index
// 最简单的 IVF-Flat：
// - centroids: [nlist, dim]
// - list_offsets: size nlist+1, 每个簇的起始/结束位置
// - list_indices: size N, 存向量在原DB中的索引
struct IVFIndex{
    int nlist;
    int dim;
    vector<float> centroids;  // [nlist * dim]
    vector<int> list_offsets; // [nlist +1]
    vector<int> list_indices; // [N]
};

// 把这个方法改成能在gpu上运行的方法，比如__device__
int nearset_centroid(const float* x, const vector<float>& centroids, int nlist, int dim){
    // 寻找与x最近的centroid的id,算法上和brute nn一样
    int best_c=0;
    float best_dist = numeric_limits<float>:: max();

    for (int c=0; c<nlist; c++){
        const float* cent = &centroids[c*dim];
        float dist = 0.0f;
        for(int d=0; d<dim; d++){
            float diff = x[d] - cent[d];
            dist += diff*diff;
        }
        if(dist< best_dist){
            best_dist = dist;
            best_c=c;
        }
    }
    return best_c;
}

// k-means training, cpu
void train_ivf_centroids(const vector<float>& db, int N, int dim, int nlist, int niter, vector<float>& centroids){
    centroids.resize(nlist*dim);
    // init
    for(int c=0; c<nlist;c++){
        const float* src = &db[(N>c? c:0)*dim];
        float* dest = &centroids[c*dim];
        for (int d=0; d<dim; d++){
            dest[d]=src[d];
        }
    }

    //temporal buffer
    vector<float> new_centroids(nlist*dim, 0.0f);
    vector<int> counts(nlist, 0);

    for(int it=0; it<niter; it++){
        std::fill(new_centroids.begin(), new_centroids.end(), 0);
        std::fill(counts.begin(), counts.end(), 0);

        // assign
        for (int i=0; i<N; i++){
            const float* x = &db[i*dim];
            int cid = nearset_centroid(x, centroids, nlist, dim);
            float* acc = &new_centroids[cid*dim];
            for (int d=0; d<dim; d++){
                acc[d]+=x[d];
            }
            counts[cid] +=1;
        }

        // update
        for (int c=0; c<nlist; c++){
            float* cent = &centroids[c*dim];
            if(counts[c]>0){
                float inv = 1.0/counts[c];
                float* acc = &new_centroids[c*dim];
                for (int d=0; d<dim; d++){
                    cent[d] = acc[d]*inv;
                }
            }
            else{
                //对于没有分配的簇，随机初始化
                static random_device rd;
                static mt19937 rng(rd());
                uniform_real_distribution<float> dist(0.0f, 1.0f);
                for(int d=0; d<dim; d++){
                    cent[d] = dist(rng);
                }
            }
        }
        printf("k-means iter %d / %d done.\n", it + 1, niter);
    }
}

// 根据centroids构建倒排表
void build_ivf_lists(const vector<float>& db, int N, int dim, int nlist, const vector<float>& centroids, IVFIndex& index){
    index.nlist = nlist;
    index.dim = dim;
    index.centroids = centroids;
    
    vector<int> counts(nlist, 0);
    vector<int>assign_ids(N,0);

    //寻找每个向量所属的簇
    for (int i=0;i<N; i++){
        const float* x = &db[i*dim];
        int cid = nearset_centroid(x, index.centroids, nlist, dim);
        assign_ids[i] = cid;
        counts[cid] += 1;
    }

    // 前缀和计算offset
    index.list_offsets.assign(nlist+1, 0);
    for(int c=0; c<nlist; c++){
        index.list_offsets[c+1] = index.list_offsets[c] + counts[c];
    }
    //构建倒排表(建立下标的映射，使cu内的向量聚在一起)
    int total = index.list_offsets[nlist];
    vector<int> cur_pos = index.list_offsets;
    index.list_indices.assign(N, -1);
    for (int i=0;i<N;i++){
        int cid = assign_ids[i];
        int pos = cur_pos[cid];
        cur_pos[cid]++;
        index.list_indices[pos] = i;
    }
    printf("IVF lists built: total=%d, avg per list=%.2f\n",
           total, total * 1.0f / nlist);
}

vector<int> ivf_select_candidates(const IVFIndex& index, const vector<float>& query, int nprobe){
    int nlist = index.nlist;
    int dim = index.dim;
    vector<pair<float, int>> dist2cid;
    dist2cid.reserve(nlist);

    const float* q = query.data();
    for (int c=0; c<nlist; c++){
        const float* cent = &index.centroids[c*dim];
        float dist = 0.0f;
        for (int d=0; d<dim; d++){
            float diff = q[d]-cent[d];
            dist+=diff*diff;
        }
        dist2cid.emplace_back(dist, c);
    }

    //选出距离最小的nprobe个簇
    if(nprobe>nlist) nprobe = nlist;
    std::partial_sort(dist2cid.begin(), dist2cid.begin()+nprobe, dist2cid.end(), [](const auto& a, const auto& b){return a.first < b.first;});

    //选出这些簇中的ids
    vector<int>candidates;
    for (int i=0;i<nprobe;i++){
        int cid = dist2cid[i].second;
        int st = index.list_offsets[cid];
        int ed = index.list_offsets[cid+1];
        for(int p=st;p<ed; p++){
            candidates.push_back(index.list_indices[p]);
        }
    }
    return candidates;
}

// kernel
__global__ void l2_nn_subset_kernel(const float* __restrict__ db, const float* __restrict__ query, const int* __restrict__ candidates, int num_candidates, int dim, int* out_idx){
    extern __shared__ float sdata[];
    float* sdist = sdata;
    int* sidx = (int*)&sdist[blockDim.x];

    int tid = threadIdx.x;
    float best_dist = FLT_MAX;
    int best_idx = -1;

    for (int i= tid; i<num_candidates; i+=blockDim.x){
        int idx = candidates[i];
        const float* x = db + idx*dim;
        float dist = 0.0f;

        for(int d=0;d<dim;d++){
            float diff = query[d] - x[d];
            dist += diff*diff;
        }
        if(dist<best_dist){
            best_dist=dist;
            best_idx=idx;
        }
    }
    sdist[tid] = best_dist;
    sidx[tid]  = best_idx;
    __syncthreads();

    //reduction
    for (int offset = blockDim.x/2; offset>0; offset>>=1){
        if(tid<offset){
            if(sdist[tid+offset]<sdist[tid]){
                sdist[tid] = sdist[tid+offset];
                sidx[tid] = sidx[tid+offset];
            }
        }
        __syncthreads();
    }

    if(tid==0){
        *out_idx = sidx[0];
    }
}

int main(){
    const int N = 100000;
    const int dim = 128;

    printf("N = %d, dim = %d\n", N, dim);

    // ============= random number on cpu ===================
    vector<float> h_db(N*dim);
    vector<float> h_query(dim);
    fill_random(h_db);
    fill_random(h_query);

    // ============ calculate groundtruth on cpu ===================
    printf("running CPU brute force NN (ground truth)...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int cpu_idx = cpu_bruteforce_nn(h_db, h_query, N, dim);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, milli>(cpu_end-cpu_start).count();
    printf("CPU nn index: %d, time = %.3f ms\n", cpu_idx, cpu_ms);

    // ============ build IVF index on CPU =======================
    int nlist = 256; // list number
    int kmeans_iters = 5;
    int nprobe = 16; // list number to probe

    // 下面能不能把训练挪到device上进行
    printf("Training IVF centroids (nlist=%d, iters=%d)...\n", nlist, kmeans_iters);
    vector<float> centroids;
    train_ivf_centroids(h_db, N, dim, nlist, kmeans_iters, centroids);

    IVFIndex ivf;
    build_ivf_lists(h_db, N, dim, nlist, centroids, ivf);

    // ============ select candidates using IVF ===================
    printf("Selecting candidates with IVF (nprobe=%d)...\n", nprobe);
    vector<int> candidates = ivf_select_candidates(ivf, h_query, nprobe);

    printf("Candidates: %zu (%.2f%% of DB)\n",
           candidates.size(), candidates.size() * 100.0 / N);
    if (candidates.empty()) {
        printf("No candidates selected. Something is wrong.\n");
        return 0;
    }

    // ============ allocate GPU mem and copy data =================
    float* d_db = nullptr;
    float* d_query = nullptr;
    int* d_out_index = nullptr;
    int* d_candidates = nullptr;

    CUDA_CHECK(cudaMalloc(&d_db, N*dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query, dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_index, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidates, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_db, h_db.data(), N*dim*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), dim*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidates, candidates.data(), sizeof(int), cudaMemcpyHostToDevice));

    // ================ launch GPU kernel ==========================
    int threads = 256;
    int blocks = 1; // single query
    size_t shared_bytes = threads * (sizeof(float) + sizeof(int));

    printf("Running GPU IVF-Flat nn kernel...\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    l2_nn_subset_kernel<<<blocks, threads, shared_bytes>>>(
        d_db, d_query, d_candidates,
        (int)candidates.size(), dim, d_out_index
    );
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ============== copy back & compare ==========================
    int gpu_idx = -1;
    CUDA_CHECK(cudaMemcpy(&gpu_idx, d_out_index, sizeof(int), cudaMemcpyDeviceToHost));
    printf("GPU IVF nn index: %d, kernel time = %.3f ms\n", gpu_idx, gpu_ms);

    if (gpu_idx == cpu_idx) {
        printf("✅ IVF + GPU result matches CPU ground truth (with this setting).\n");
    } else {
        printf("ℹ️  IVF is approximate: CPU=%d, GPU IVF=%d\n", cpu_idx, gpu_idx);
    }

    // ============ release resources =============================
    CUDA_CHECK(cudaFree(d_db));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_out_index));
    CUDA_CHECK(cudaFree(d_candidates));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}