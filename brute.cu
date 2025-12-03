#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>
#include <random>
#include <cuda_runtime.h>
#include <cfloat>
using namespace std;

//  ================= 工具函数 =================
// 为什么用宏而不是函数？？
#define CUDA_CHECK(expr)\
    do{\
        cudaError_t err = (expr); \
        if (err != cudaSuccess){\
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", \
                #expr, __FILE__, __LINE__, cudaGetErrorString(err));\
            std::exit(EXIT_FAILURE); \
        }\
    }while(0)

// [0,1) random float gen， 这里怎么生成，更多是语法问题，先放过去
void fill_random(vector<float>& vec){
    mt19937 rng(42);
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &x: vec) x = dist(rng);
}

// cpu brute force ann
// 为什么这里的距离可以按位计算？？
int cpu_bruteforce_nn(const vector<float>& db,
                      const vector<float>& query,
                      int N, int dim){
    float best_dist = numeric_limits<float>::max();
    int best_idx = -1;

    for(int i=0; i<N; i++){
        float dist = 0.0f;
        const float* x = &db[i * dim];
        for(int d=0; d<dim; d++){
            float diff = x[d] - query[d];
            dist += diff * diff;
        }
        if(dist<best_dist){
            best_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

// ================= GPU kernel =================
// 每个block处理一个query
// 每个线程遍历若干向量，计算L2距离

// a kernel
__global__ void l2_nn_kernel(
    const float* __restrict__ db, //[N,dim]
    const float* __restrict__ query, //[dim] one query
    int N,
    int dim,
    int* out_index //nn index
){
    extern __shared__ float sdata[]; 
    float* sdist = sdata;
    int* sidx = (int*)&sdata[blockDim.x];

    int tid = threadIdx.x;
    float best_dist = FLT_MAX;
    int best_idx = -1;

    // calc dists
    for(int idx = tid; idx<N; idx +=blockDim.x){
        const float* x = db + idx * dim;
        float dist = 0.0f;

        for(int d = 0; d<dim; d++){
            float diff = x[d] - query[d];
            dist += diff*diff;
        }
        if (dist<best_dist){
            best_dist = dist;
            best_idx = idx;
        }
    }

    // write in shared mem
    sdist[tid] = best_dist;
    sidx[tid] = best_idx;
    __syncthreads(); // ??

    //规约找最小距离
    for (int offset =blockDim.x /2; offset>0; offset >>=1){
        if (tid < offset){
            if(sdist[tid+offset] < sdist[tid]){
                sdist[tid] = sdist[tid + offset];
                sidx[tid] = sidx[tid + offset];
            }
        }
        __syncthreads();
    }

    // 结果放在tid==0的线程
    if (tid ==0){
        *out_index = sidx[0];
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
    // 数据和query都是向量，这里的实现，向量的每一位是一个0-1的小数？？？

    // ============ calculate groundtruth on cpu ===================
    printf("running CPU brute force NN...\n");
    int cpu_idx = cpu_bruteforce_nn(h_db, h_query, N, dim);
    printf("CPU nn index: %d\n", cpu_idx);

    // ================= allocate GPU mem and cpy data =======================
    float *d_db = nullptr;
    float *d_query = nullptr;
    int *d_out_index = nullptr;

    CUDA_CHECK(cudaMalloc(&d_db, N*dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query, dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_index, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_db, h_db.data(), N*dim*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), dim*sizeof(float), cudaMemcpyHostToDevice));

    // ================ activate GPU kernel =====================
    int threads = 256;
    int  block = 1;
    size_t shared_bytes = threads * (sizeof(float) + sizeof(int));

    printf("Running GPU nn kernel...\n");
    l2_nn_kernel<<<block, threads, shared_bytes>>>(
        d_db, d_query, N, dim, d_out_index
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // ============== cpy back & compare ================
    int gpu_idx = -1;
    CUDA_CHECK(cudaMemcpy(&gpu_idx, d_out_index, sizeof(int), cudaMemcpyDeviceToHost));
    printf("GPU nn index: %d\n", gpu_idx);

    if (gpu_idx == cpu_idx){
        printf("✅ CPU and GPU results MATCH.\n");
    } else {
        printf("❌ MISMATCH! (cpu=%d, gpu=%d)\n", cpu_idx, gpu_idx);
    }
    
    // ============ release resourse =====================
    CUDA_CHECK(cudaFree(d_db));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_out_index));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}