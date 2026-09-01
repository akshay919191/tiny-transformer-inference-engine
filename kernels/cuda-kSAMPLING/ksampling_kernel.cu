#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


constexpr int MAX_K     = 32;
constexpr int THREADS   = 256;
constexpr int NUM_WARPS = THREADS / 32;

template<int CAP>
__forceinline__ __device__ void insert_topk(
    const float value,
    const int val_idx,
    float* top,
    int* idx,
    float& min_value,
    int& min_position
) {
    if (value > min_value) {
        top[min_position] = value;
        idx[min_position] = val_idx;

        min_value = top[0];
        min_position = 0;

        #pragma unroll
        for (int j = 1; j < CAP; ++j) {
            if (top[j] < min_value) {
                min_value = top[j];
                min_position = j;
            }
        }
    }
}

template<int CAP>
__forceinline__ __device__ void warp_reduce_topk(
    float* top,
    int* idx,
    float& min_value,
    int& min_position
) {
    constexpr unsigned FULL_MASK = 0xffffffffu;
    const int lane = threadIdx.x & 31;

    for (int delta = 16; delta > 0; delta >>= 1) {
        #pragma unroll
        for (int i = 0; i < CAP; ++i) {
            const float recv_val = __shfl_down_sync(FULL_MASK, top[i], delta);
            const int   recv_idx = __shfl_down_sync(FULL_MASK, idx[i], delta);
            if (lane < delta) {
                insert_topk<CAP>(recv_val, recv_idx, top, idx, min_value, min_position);
            }
        }
    }
}

template<int CAP>
__global__ void __launch_bounds__(THREADS)
ksmapling(
    const float* __restrict__ vocab,
    const int V,
    const int k,                   
    float* __restrict__ out_vals,
    int* __restrict__ out_idxs
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    float top[CAP];
    int   idx[CAP];

    float min_value = -INFINITY;
    int   min_position = 0;

    #pragma unroll
    for (int i = 0; i < CAP; ++i) {
        top[i] = -INFINITY;
        idx[i] = -1;
    }

    // 64-bit row offset: safe for large batch * vocab
    const float* __restrict__ row_ptr =
        vocab + static_cast<size_t>(row) * static_cast<size_t>(V);

    const int n4 = V >> 2;            // number of full float4 groups
    const int tail_start = n4 << 2;

    if ((reinterpret_cast<uintptr_t>(row_ptr) & 15u) == 0u) {
        const float4* __restrict__ row4 = reinterpret_cast<const float4*>(row_ptr);

        for (int v4 = tid; v4 < n4; v4 += THREADS) {
            const float4 pkt = __ldcs(row4 + v4);
            const int base = v4 << 2;

            insert_topk<CAP>(pkt.x, base + 0, top, idx, min_value, min_position);
            insert_topk<CAP>(pkt.y, base + 1, top, idx, min_value, min_position);
            insert_topk<CAP>(pkt.z, base + 2, top, idx, min_value, min_position);
            insert_topk<CAP>(pkt.w, base + 3, top, idx, min_value, min_position);
        }
    } else {
        for (int v = tid; v < tail_start; v += THREADS) {
            insert_topk<CAP>(__ldcs(row_ptr + v), v, top, idx, min_value, min_position);
        }
    }

    for (int v = tail_start + tid; v < V; v += THREADS) {
        insert_topk<CAP>(__ldcs(row_ptr + v), v, top, idx, min_value, min_position);
    }

    warp_reduce_topk<CAP>(top, idx, min_value, min_position);

    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);                     // NUM_WARPS * CAP
    int*   s_idxs = reinterpret_cast<int*>(s_vals + NUM_WARPS * CAP);   // NUM_WARPS * CAP
    float* f_vals = reinterpret_cast<float*>(s_idxs + NUM_WARPS * CAP); // CAP, final stage
    int*   f_idxs = reinterpret_cast<int*>(f_vals + CAP);               // CAP, final stage

    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < CAP; ++i) {
            s_vals[warp_id * CAP + i] = top[i];
            s_idxs[warp_id * CAP + i] = idx[i];
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        min_value = -INFINITY;
        min_position = 0;

        #pragma unroll
        for (int i = 0; i < CAP; ++i) {
            top[i] = -INFINITY;
            idx[i] = -1;
        }

        constexpr int TOTAL = NUM_WARPS * CAP;
        for (int i = lane; i < TOTAL; i += 32) {
            insert_topk<CAP>(s_vals[i], s_idxs[i], top, idx, min_value, min_position);
        }

        warp_reduce_topk<CAP>(top, idx, min_value, min_position);

        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < CAP; ++i) {
                f_vals[i] = top[i];
                f_idxs[i] = idx[i];
            }

            float* out_v = out_vals + static_cast<size_t>(row) * k;
            int*   out_i = out_idxs + static_cast<size_t>(row) * k;

            for (int i = 0; i < k; ++i) {
                int best = 0;
                float best_val = f_vals[0];

                #pragma unroll
                for (int j = 1; j < CAP; ++j) {
                    const float v = f_vals[j];
                    if (v > best_val) {
                        best_val = v;
                        best = j;
                    }
                }

                out_v[i] = best_val;
                out_i[i] = f_idxs[best];
                f_vals[best] = -INFINITY;   
            }
        }
    }
}

void ksmapling_cuda(
    torch::Tensor vocab,
    int64_t K,
    torch::Tensor out_vals,
    torch::Tensor out_idxs
) {
    TORCH_CHECK(vocab.is_cuda() && out_vals.is_cuda() && out_idxs.is_cuda(),
                "All tensors must be CUDA tensors");
    TORCH_CHECK(vocab.dim() == 2, "vocab must be 2-D [batch, vocab]");
    TORCH_CHECK(vocab.scalar_type() == torch::kFloat32, "vocab must be float32");
    TORCH_CHECK(vocab.is_contiguous(), "vocab must be contiguous");
    TORCH_CHECK(K >= 1 && K <= MAX_K,
                "Unsupported K: ", K, " (supported range: 1..", MAX_K, ")");

    const int batch = static_cast<int>(vocab.size(0));
    const int V     = static_cast<int>(vocab.size(1));
    const int k     = static_cast<int>(K);

    TORCH_CHECK(out_vals.scalar_type() == torch::kFloat32, "out_vals must be float32");
    TORCH_CHECK(out_idxs.scalar_type() == torch::kInt32, "out_idxs must be int32");
    TORCH_CHECK(out_vals.is_contiguous() && out_idxs.is_contiguous(),
                "output tensors must be contiguous");
    TORCH_CHECK(out_vals.size(0) == batch && out_vals.size(1) == K,
                "out_vals must be [batch, K]");
    TORCH_CHECK(out_idxs.size(0) == batch && out_idxs.size(1) == K,
                "out_idxs must be [batch, K]");

    if (batch == 0) return;

    int cap = 1;
    while (cap < k) cap <<= 1;

    const at::cuda::OptionalCUDAGuard guard(vocab.device());

    dim3 grid(batch);
    dim3 block(THREADS);

    const size_t smem =
        static_cast<size_t>(2 * NUM_WARPS * cap + 2 * cap) * sizeof(float);

    const float* vocab_ptr    = vocab.data_ptr<float>();
    float*       out_vals_ptr = out_vals.data_ptr<float>();
    int*         out_idxs_ptr = out_idxs.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (cap) {
        case 1:
            ksmapling<1><<<grid, block, smem, stream>>>(
                vocab_ptr, V, k, out_vals_ptr, out_idxs_ptr);
            break;
        case 2:
            ksmapling<2><<<grid, block, smem, stream>>>(
                vocab_ptr, V, k, out_vals_ptr, out_idxs_ptr);
            break;
        case 4:
            ksmapling<4><<<grid, block, smem, stream>>>(
                vocab_ptr, V, k, out_vals_ptr, out_idxs_ptr);
            break;
        case 8:
            ksmapling<8><<<grid, block, smem, stream>>>(
                vocab_ptr, V, k, out_vals_ptr, out_idxs_ptr);
            break;
        case 16:
            ksmapling<16><<<grid, block, smem, stream>>>(
                vocab_ptr, V, k, out_vals_ptr, out_idxs_ptr);
            break;
        case 32:
            ksmapling<32><<<grid, block, smem, stream>>>(
                vocab_ptr, V, k, out_vals_ptr, out_idxs_ptr);
            break;
        default:
            TORCH_CHECK(false, "unreachable capacity");
    }

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}