#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>
#include <random>
#include <algorithm>   // for std::sort, std::partial_sort
#include <cuda_runtime.h>
#include <cfloat>

using namespace std;

//  ================= 工具函数 =================

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t err = (expr);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                \
                    #expr, __FILE__, __LINE__, cudaGetErrorString(err));     \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// [0,1) random float gen
void fill_random(vector<float>& vec) {
    mt19937 rng(42);
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : vec) x = dist(rng);
}

// cpu brute force nn over FULL DB
int cpu_bruteforce_nn(const vector<float>& db,
                      const vector<float>& query,
                      int N, int dim) {
    float best_dist = numeric_limits<float>::max();
    int best_idx = -1;

    for (int i = 0; i < N; i++) {
        float dist = 0.0f;
        const float* x = &db[i * dim];
        for (int d = 0; d < dim; d++) {
            float diff = x[d] - query[d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

// =================== IVF 结构定义 ===================

// 最简单的 IVF-Flat：
// - centroids: [nlist, dim]
// - list_offsets: size nlist+1, 每个簇的起始/结束位置
// - list_indices: size N, 存向量在原DB中的索引
struct IVFIndex {
    int nlist;
    int dim;
    vector<float> centroids;   // [nlist * dim]
    vector<int>   list_offsets;// [nlist + 1]
    vector<int>   list_indices;// [N]
};

// 计算 x 与所有 centroid 的 L2 距离，返回最近的 centroid id
// k 敏感，初值敏感
int nearest_centroid(const float* x,
                     const vector<float>& centroids,
                     int nlist, int dim) {
    int best_c = 0;
    float best_dist = numeric_limits<float>::max();

    for (int c = 0; c < nlist; ++c) {
        const float* cent = &centroids[c * dim];
        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = x[d] - cent[d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_c = c;
        }
    }
    return best_c;
}

// 用一个非常朴素的 k-means 在 CPU 上训练 IVF 的 centroids
void train_ivf_centroids(const vector<float>& db,
                         int N, int dim,
                         int nlist, int niter,
                         vector<float>& centroids) {
    centroids.resize(nlist * dim);

    // 1) 初始化：简单起见，直接取前 nlist 个向量作为初始中心
    for (int c = 0; c < nlist; ++c) {
        const float* src = &db[(N > c ? c : 0) * dim];
        float* dst = &centroids[c * dim];
        for (int d = 0; d < dim; ++d) {
            dst[d] = src[d];
        }
    }

    // 临时缓冲
    vector<float> new_centroids(nlist * dim, 0.0f);
    vector<int>   counts(nlist, 0);

    // 2) 简单几轮 k-means
    for (int it = 0; it < niter; ++it) {
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        // 2.1 assign
        for (int i = 0; i < N; ++i) {
            const float* x = &db[i * dim];
            int cid = nearest_centroid(x, centroids, nlist, dim);

            float* acc = &new_centroids[cid * dim];
            for (int d = 0; d < dim; ++d) {
                acc[d] += x[d];
            }
            counts[cid] += 1;
        }

        // 2.2 update
        for (int c = 0; c < nlist; ++c) {
            float* cent = &centroids[c * dim];
            if (counts[c] > 0) {
                float inv = 1.0f / counts[c];
                float* acc = &new_centroids[c * dim];
                for (int d = 0; d < dim; ++d) {
                    cent[d] = acc[d] * inv;
                }
            }
            // 如果某个簇没人分配，随机重置
        }

        printf("k-means iter %d / %d done.\n", it + 1, niter);
    }
}

// 根据训练好的 centroids 构建倒排表
void build_ivf_lists(const vector<float>& db,
                     int N, int dim,
                     int nlist,
                     const vector<float>& centroids,
                     IVFIndex& index) {
    index.nlist = nlist;
    index.dim = dim;
    index.centroids = centroids;

    vector<int> counts(nlist, 0);
    vector<int> assign_ids(N, 0);

    // 1) 为每个向量分配一个最近的簇，并统计每个簇的大小
    for (int i = 0; i < N; ++i) {
        const float* x = &db[i * dim];
        int cid = nearest_centroid(x, index.centroids, nlist, dim);
        assign_ids[i] = cid;
        counts[cid] += 1;
    }

    // 2) 前缀和得到 offsets
    index.list_offsets.assign(nlist + 1, 0);
    for (int c = 0; c < nlist; ++c) {
        index.list_offsets[c + 1] = index.list_offsets[c] + counts[c];
    }
    int total = index.list_offsets[nlist];
    index.list_indices.assign(total, -1);

    // 3) 再来一遍，把向量 id 填入倒排表
    vector<int> cur_pos = index.list_offsets;
    for (int i = 0; i < N; ++i) {
        int cid = assign_ids[i];
        int pos = cur_pos[cid]++;
        index.list_indices[pos] = i;
    }

    printf("IVF lists built: total=%d, avg per list=%.2f\n",
           total, total * 1.0f / nlist);
}

// 给定 query，用 IVF 选择候选向量（只返回 ids，后面交给 GPU 计算精确距离）
// 返回的 candidates 是「原 DB 索引」的列表。
vector<int> ivf_select_candidates(const IVFIndex& index,
                                  const vector<float>& query,
                                  int nprobe) {
    int nlist = index.nlist;
    int dim = index.dim;

    // 1) 计算 query 到所有 centroids 的距离
    vector<pair<float, int>> dist2cid;
    dist2cid.reserve(nlist);

    const float* q = query.data();
    for (int c = 0; c < nlist; ++c) {
        const float* cent = &index.centroids[c * dim];
        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = q[d] - cent[d];
            dist += diff * diff;
        }
        dist2cid.emplace_back(dist, c);
    }

    // 2) 选出距离最小的 nprobe 个簇
    if (nprobe > nlist) nprobe = nlist;
    std::partial_sort(dist2cid.begin(),
                      dist2cid.begin() + nprobe,
                      dist2cid.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });

    // 3) 拼出候选向量 id 列表
    vector<int> candidates;
    for (int i = 0; i < nprobe; ++i) {
        int cid = dist2cid[i].second;
        int start = index.list_offsets[cid];
        int end   = index.list_offsets[cid + 1];
        for (int p = start; p < end; ++p) {
            candidates.push_back(index.list_indices[p]);
        }
    }

    return candidates;
}

// ================= GPU kernel =================
// 在「候选集合」上做 L2 最近邻搜索
// 每个 block 处理一个 query（这里还是单 query）
// candidates: 一维数组，存的是「原 DB 索引」

__global__ void l2_nn_subset_kernel(
    const float* __restrict__ db,       // [N, dim]
    const float* __restrict__ query,    // [dim]
    const int*   __restrict__ candidates, // [num_candidates]
    int num_candidates,
    int dim,
    int* out_index      // 输出：原 DB 中的索引
) {
    extern __shared__ float sdata[];
    float* sdist = sdata;
    int*   sidx  = (int*)&sdist[blockDim.x];

    int tid = threadIdx.x;
    float best_dist = FLT_MAX;
    int   best_idx  = -1;

    // 每个线程处理若干候选向量
    for (int i = tid; i < num_candidates; i += blockDim.x) {
        int idx = candidates[i];   // 原 DB 索引
        const float* x = db + idx * dim;

        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = x[d] - query[d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = idx;
        }
    }

    // 写入共享内存
    sdist[tid] = best_dist;
    sidx[tid]  = best_idx;
    __syncthreads();

    // 规约求最小距离
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (sdist[tid + offset] < sdist[tid]) {
                sdist[tid] = sdist[tid + offset];
                sidx[tid]  = sidx[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_index = sidx[0];
    }
}

int main() {
    const int N = 100000;
    const int dim = 128;

    printf("N = %d, dim = %d\n", N, dim);

    // ============= random number on cpu ===================
    vector<float> h_db(N * dim);
    vector<float> h_query(dim);

    fill_random(h_db);
    fill_random(h_query);

    // ============ calculate groundtruth on cpu ===================
    printf("running CPU brute force NN (ground truth)...\n");
    int cpu_idx = cpu_bruteforce_nn(h_db, h_query, N, dim);
    printf("CPU nn index: %d\n", cpu_idx);

    // ============ build IVF index on CPU =======================
    int nlist = 256;      // 簇的数量（粗量化桶数）
    int kmeans_iters = 5; // k-means 迭代次数
    int nprobe = 16;      // 搜索时探测的簇数

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
    int*   d_out_index = nullptr;
    int*   d_candidates = nullptr;

    CUDA_CHECK(cudaMalloc(&d_db, N * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_index, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidates, candidates.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_db, h_db.data(),
                          N * dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(),
                          dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidates, candidates.data(),
                          candidates.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    // ================ launch GPU kernel ==========================
    int threads = 256;
    int blocks  = 1;  // 单 query
    size_t shared_bytes = threads * (sizeof(float) + sizeof(int));

    printf("Running GPU IVF-Flat nn kernel...\n");
    l2_nn_subset_kernel<<<blocks, threads, shared_bytes>>>(
        d_db, d_query, d_candidates,
        (int)candidates.size(), dim, d_out_index
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // ============== copy back & compare ==========================
    int gpu_idx = -1;
    CUDA_CHECK(cudaMemcpy(&gpu_idx, d_out_index,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
    printf("GPU IVF nn index: %d\n", gpu_idx);

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
