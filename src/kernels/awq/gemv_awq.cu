/*
 * Modified from NVIDIA
 * [TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/d37b507f41a87457fe9f10f7459d08f5db235745/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv)
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}
*/

#include "gemv_awq.h"
#include "../dispatch_utils.h"

#include "../utils.cuh"
#include "../device_compat.h"
#include <stdio.h>
#include "dequantize.cuh"

#if defined(__HIP_PLATFORM_AMD__)

#include "device_compat.h"

namespace {

constexpr int kInterleave = 4;

__device__ __forceinline__ int compute_ic_div64(int in_features) {
    int value = in_features / 64;
    return value > 0 ? value : 1;
}

__global__ void dequantize_awq_kernel(const int32_t *packed,
                                      const half *scales,
                                      const half *zeros,
                                      half *dequantized,
                                      int rows,
                                      int cols,
                                      int in_features,
                                      int group_size,
                                      int padded_groups,
                                      int scale_stride) {
    const int out_features = rows * kInterleave;
    const int total        = out_features * in_features;
    const int idx          = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int oc_idx = idx / in_features;
    const int ic_idx = idx % in_features;

    const int block     = idx / 32;
    const int offset    = idx % 32;
    const int nibble    = offset / 8;
    const int j         = offset % 8;
    const int index1    = block * 8 + j;
    const int d         = index1 % 16;
    const int tmp       = index1 / 16;
    const int ic_div64  = compute_ic_div64(in_features);
    const int c         = tmp % ic_div64;
    const int row4b     = tmp / ic_div64;
    const int row       = row4b / 4;
    const int b         = row4b % 4;
    const int col_index = ((c * 4 + b) * 16) + d;

    if (row >= rows || col_index >= cols) {
        dequantized[idx] = __float2half(0.0f);
        return;
    }

    const uint16_t packed_val = static_cast<uint16_t>(packed[row * cols + col_index] & 0xFFFF);
    const int nibble_val      = (packed_val >> (4 * nibble)) & 0xF;

    if (padded_groups <= 0) {
        dequantized[idx] = __float2half(0.0f);
        return;
    }

    const int effective_group = group_size > 0 ? group_size : in_features;
    int group_idx              = ic_idx / effective_group;
    if (group_idx < 0) {
        group_idx = 0;
    }
    if (group_idx >= padded_groups) {
        group_idx = padded_groups - 1;
    }

    const float scale = __half2float(scales[group_idx * scale_stride + oc_idx]);
    const float zero  = __half2float(zeros[group_idx * scale_stride + oc_idx]);

    dequantized[idx] = __float2half(nibble_val * scale + zero);
}

__global__ void gemm_fp16_kernel(const half *input,
                                 const half *weight,
                                 half *output,
                                 int M,
                                 int N,
                                 int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    for (int k_idx = 0; k_idx < K; ++k_idx) {
        const float a_val = __half2float(input[row * K + k_idx]);
        const float b_val = __half2float(weight[col * K + k_idx]);
        acc += a_val * b_val;
    }

    output[row * N + col] = __float2half(acc);
}

inline void launch_dequantize(const int32_t *packed,
                              const half *scales,
                              const half *zeros,
                              half *dequantized,
                              int rows,
                              int cols,
                              int in_features,
                              int group_size,
                              int padded_groups,
                              int scale_stride) {
    const int out_features = rows * kInterleave;
    const int total        = out_features * in_features;
    constexpr int threads  = 256;
    dim3 block(threads);
    dim3 grid((total + threads - 1) / threads);
    hipLaunchKernelGGL(dequantize_awq_kernel,
                       grid,
                       block,
                       0,
                       getCurrentGpuStream(),
                       packed,
                       scales,
                       zeros,
                       dequantized,
                       rows,
                       cols,
                       in_features,
                       group_size,
                       padded_groups,
                       scale_stride);
    checkCUDA(gpu_runtime::getLastError());
}

inline void launch_gemm(const half *input, const half *weight, half *output, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    hipLaunchKernelGGL(gemm_fp16_kernel,
                       grid,
                       block,
                       0,
                       getCurrentGpuStream(),
                       input,
                       weight,
                       output,
                       M,
                       N,
                       K);
    checkCUDA(gpu_runtime::getLastError());
}

} // namespace

Tensor gemv_awq(Tensor _in_feats,
                Tensor _kernel,
                Tensor _scaling_factors,
                Tensor _zeros,
                int m,
                int n,
                int k,
                int group_size) {
    const int rows         = _kernel.size(0);
    const int cols         = _kernel.size(1);
    const int in_features  = k;
    const int out_features = rows * kInterleave;
    const int padded_groups = _scaling_factors.size(0);
    const int scale_stride  = _scaling_factors.size(1);

    assert(out_features == n);

    auto output_shape   = _in_feats.shape.dataExtent;
    output_shape.back() = out_features;

    Tensor dequantized = Tensor::empty({out_features, in_features}, Tensor::FP16, _kernel.device());

    launch_dequantize(_kernel.data_ptr<int32_t>(),
                      _scaling_factors.data_ptr<half>(),
                      _zeros.data_ptr<half>(),
                      dequantized.data_ptr<half>(),
                      rows,
                      cols,
                      in_features,
                      group_size,
                      padded_groups,
                      scale_stride);

    const int computed_m = _in_feats.numel() / in_features;
    assert(computed_m == m);

    Tensor out = Tensor::empty(output_shape, _in_feats.scalarType(), _in_feats.device());
    launch_gemm(_in_feats.data_ptr<half>(),
                dequantized.data_ptr<half>(),
                out.data_ptr<half>(),
                computed_m,
                out_features,
                in_features);

    return out;
}

#else

#define PACK_FACTOR 8
#define WARP_SIZE 32
#define MEM_ACCESS_SIZE 128

// Reduce sum within the warp using the tree reduction algorithm.
template<typename float_t, int Num, int WarpSize>
__device__ __forceinline__ static void warp_reduce(float_t *psum, float (*out_smem)[Num * 4]) {
    // kInterleave = 4
    float fpsum[Num];
#pragma unroll
    for (int i = 0; i < Num; ++i) {
        fpsum[i] = static_cast<float>(psum[i]);
    }

#pragma unroll
    for (int i = 0; i < Num; ++i) {
        // T0 + T1 + T8 + T9 + T16 + T17 + T24 + T25 (kInterleave = 4)
        fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 16);
        fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 8);
        fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 1);
    }
    __syncthreads();
    int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
    if (lane == 0 || lane == 2 || lane == 4 || lane == 6) {
#pragma unroll
        for (int i = 0; i < Num; ++i) {
            out_smem[warp][i * 4 + lane / 2] = fpsum[i];
        }
    }
    __syncthreads();
};

__device__ __forceinline__ int make_divisible(int c, int divisor) {
    return (c + divisor - 1) / divisor;
}

template<typename half_t>
__device__ __forceinline__ packed_as<half_t, 2>::type half2half2(half_t x);

template<>
__device__ __forceinline__ packed_as<half, 2>::type half2half2<half>(half x) {
    return __half2half2(x);
}

template<>
__device__ __forceinline__ packed_as<__nv_bfloat16, 2>::type half2half2<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162bfloat162(x);
}

template<typename T>
__device__ __forceinline__ float2 half22float2(T val);

template<>
__device__ __forceinline__ float2 half22float2<half2>(half2 val) {
    return __half22float2(val);
}

template<>
__device__ __forceinline__ float2 half22float2<__nv_bfloat162>(__nv_bfloat162 val) {
    return __bfloat1622float2(val);
}

template<typename half_t, int NPerBlock, int Batch, int BlockSize, int GroupSize>
__global__ void gemv_kernel(const half_t *inputs,
                            const uint32_t *weight,
                            const half_t *scales,
                            const half_t *zeros,
                            half_t *outputs,
                            const int IC,
                            const int OC) {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    if constexpr (std::is_same_v<half_t, __nv_bfloat16>) {
        trap_unsupported_arch();
        return;
    }
#endif
    using half2_t  = typename packed_as<half_t, 2>::type;
    using accum_t  = float;
    using accum2_t = typename packed_as<accum_t, 2>::type;

    const int kStride            = 64;
    const int kElemsPerThread    = MEM_ACCESS_SIZE / 4;
    const int kThreadsNumPerTile = kStride / kElemsPerThread;
    // assert(MEM_ACCESS_SIZE == 128);

    // static constexpr int kShuffleSize = 32;
    static constexpr int kShuffleBasicTile = 2;
    static constexpr int kShuffleContinous = 4;
    static constexpr int kShuffleStrided   = 4;

    constexpr int Num         = NPerBlock * Batch;
    constexpr int kInterleave = 4;

    alignas(16) half_t local_inputs[kElemsPerThread];
    alignas(16) uint32_t local_qweights[MEM_ACCESS_SIZE / 32];
    alignas(16) half_t half_weight_buffer[kElemsPerThread];
    alignas(16) half_t dequantized_weight[kElemsPerThread * NPerBlock];
    alignas(16) half_t local_scale[NPerBlock];
    alignas(16) half_t local_scaled_zeros[NPerBlock];

    accum_t psum[Num];
    for (int i = 0; i < Num; ++i)
        psum[i] = static_cast<accum_t>(0.f);

    // extern __shared__ uint8_t shmem[];
    // float(*out_smem)[Num * kInterleave] = reinterpret_cast<float(*)[Num * kInterleave]>(shmem);

    __shared__ float out_smem[BlockSize / WARP_SIZE * 2][Num * kInterleave];

    const int blk_row_offset = blockIdx.x * NPerBlock * kInterleave;
    const int thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kInterleave;
    const int act_k_offset   = threadIdx.x / (kThreadsNumPerTile * kInterleave) * kStride +
                             (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;
    const int group_offset = act_k_offset / GroupSize;
    // TODO: use make_divisible
    const uint32_t *blk_weight_ptr = weight + blk_row_offset * IC / PACK_FACTOR;
    const half_t *scale_ptr        = scales + blk_row_offset + thd_row_offset + group_offset * OC;
    const half_t *zeros_ptr        = zeros + blk_row_offset + thd_row_offset + group_offset * OC;
    const half_t *inputs_ptr       = inputs + act_k_offset;

    const int act_forward_step   = BlockSize * kElemsPerThread / kInterleave;
    const int scale_forward_step = act_forward_step / GroupSize * OC;

    // Main loop iteration, each block completes the outputs for several OCs
    for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread) {
// Load qweight, scales and scaled_zeros
#pragma unroll
        for (int idx = 0; idx < NPerBlock; ++idx) {
            // use float4 to load weights, each thread load 32 int4 numbers (1 x float4, 128 bit)
            *((float4 *)(local_qweights)) = *((float4 *)(blk_weight_ptr + (idx * kInterleave * IC + kk) / PACK_FACTOR));
            local_scale[idx]              = *(scale_ptr + idx * kInterleave);
            local_scaled_zeros[idx]       = *(zeros_ptr + idx * kInterleave);

// Map int4 qweight to fp format
#pragma unroll
            for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i) {
                // Converts 32 bits (8 x int4) to 8 fp16
                dequantize_s4_to_fp16x2(*reinterpret_cast<half2_t *>(local_qweights + i),
                                        reinterpret_cast<uint4 *>(half_weight_buffer + i * PACK_FACTOR));
            }

// Dequantize (apply s/z) and shuffle elements to match the weight packing format
#pragma unroll
            for (int i = 0; i < kShuffleContinous; ++i) {
#pragma unroll
                for (int j = 0; j < kShuffleStrided; ++j) {
                    half2_t w = *reinterpret_cast<half2_t *>(half_weight_buffer +
                                                             (i + j * kShuffleContinous) * kShuffleBasicTile);
                    w         = __hfma2(w, half2half2(local_scale[idx]), half2half2(local_scaled_zeros[idx]));
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 0) * NPerBlock + idx] = w.x;
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 1) * NPerBlock + idx] = w.y;
                }
            }
        }
#pragma unroll
        for (int batch_idx = 0; batch_idx < Batch; ++batch_idx) {
            const half_t *local_inputs_ptr = inputs_ptr + batch_idx * IC;
#pragma unroll
            for (int idx = 0; idx < kElemsPerThread / 8; ++idx) {
                // load activation, 8 halves (128 bits) / step.
                *((float4 *)(local_inputs + idx * 8)) = *((float4 *)(local_inputs_ptr + idx * 8));
            }
// Perform the MACs
#pragma unroll
            for (int x = 0; x < NPerBlock / 2; ++x) {
#pragma unroll
                for (int y = 0; y < kElemsPerThread; ++y) {
                    accum2_t prod = cuda_cast<accum2_t>(
                        __hmul2(*reinterpret_cast<half2_t *>(dequantized_weight + y * NPerBlock + x * 2),
                                half2half2(local_inputs[y])));
                    *reinterpret_cast<accum2_t *>(psum + batch_idx * NPerBlock + x * 2) =
                        prod + *reinterpret_cast<accum2_t *>(psum + batch_idx * NPerBlock + x * 2);
                    // *reinterpret_cast<half2_t*>(psum + batch_idx * NPerBlock + x * 2)
                    //     = __hfma2(*reinterpret_cast<half2_t*>(dequantized_weight + y * NPerBlock + x * 2),
                    //         half2half2(local_inputs[y]),
                    //         *reinterpret_cast<half2_t*>(psum + batch_idx * NPerBlock + x * 2));
                }
            }
        }
        inputs_ptr += act_forward_step;
        scale_ptr += scale_forward_step;
        zeros_ptr += scale_forward_step;
    }

    warp_reduce<accum_t, Num, WARP_SIZE>(psum, out_smem);

    // Num * Interleave = batch * NPerBlock * Interleave -> 1 thread_block write back num
    for (int i = threadIdx.x; i < Num * kInterleave; i += BlockSize) {
        int batch_idx = i / (NPerBlock * kInterleave);
        int oc_idx    = i % (NPerBlock * kInterleave);
        float acc     = 0.f;
        for (int j = 0; j < BlockSize / WARP_SIZE; ++j) {
            acc += out_smem[j][i];
        }
        outputs[batch_idx * OC + blk_row_offset + oc_idx] = static_cast<half_t>(acc);
    }
}

/*
Computes GEMV (PyTorch interface).

Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [OC, IC // 8];
  _zeros: int tensor of shape [OC, IC // G // 8];
  _scaling_factors: tensor of shape [OC, IC // G];
  blockDim_x: size of thread block, dimension x, where blockDim_x * workload_per_thread = IC;
  blockDim_y: size of thread block, dimension y, where blockDim_y * gridDim_y = OC;

Returns:
  out_feats: tensor of shape [B, OC];
*/
Tensor gemv_awq(
    Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, Tensor _zeros, int m, int n, int k, int group_size) {
    return dispatchFloat16(_scaling_factors.scalar_type(), [&]<typename half_t>() {
        assert(isTypeMatch<half_t>(_in_feats.dtype()));

        auto output_shape   = _in_feats.shape.dataExtent;
        output_shape.back() = n;

        auto in_feats        = reinterpret_cast<half_t *>(_in_feats.data_ptr<half_t>());
        auto kernel          = reinterpret_cast<uint32_t *>(_kernel.data_ptr());
        auto zeros           = reinterpret_cast<half_t *>(_zeros.data_ptr<half_t>());
        auto scaling_factors = reinterpret_cast<half_t *>(_scaling_factors.data_ptr<half_t>());

        Tensor _out_feats = Tensor::allocate(output_shape, _in_feats.dtype(), _in_feats.device());
        half_t *out_feats = reinterpret_cast<half_t *>(_out_feats.data_ptr());

        static constexpr int N_PER_BLOCK  = 2;
        static constexpr int K_INTERLEAVE = 4;
        static constexpr int BLOCK_SIZE   = 256;

        dim3 num_blocks(n / N_PER_BLOCK / K_INTERLEAVE);
        dim3 num_threads(BLOCK_SIZE);

        constexpr int GROUP_SIZE = 64;

        assert(m > 0 && m <= 8);
        assert(group_size == GROUP_SIZE);

        dispatchVal(m, std::make_integer_sequence<int, 9>(), [&]<int M>() {
            if constexpr (M == 0) {
                assert(false);
                return;
            }
            if constexpr (M > 0) {
                gemv_kernel<half_t, N_PER_BLOCK, M, BLOCK_SIZE, GROUP_SIZE>
                    <<<num_blocks, num_threads, 0, getCurrentGpuStream()>>>(
                        in_feats, kernel, scaling_factors, zeros, out_feats, k, n);
                checkCUDA(gpu_runtime::getLastError());
            }
        });

        return _out_feats;
    });
}

#endif // defined(__HIP_PLATFORM_AMD__)
