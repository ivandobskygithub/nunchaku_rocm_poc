#include "gemm_batched.h"

#if defined(NUNCHAKU_USE_HIP)

#include <hip/hip_fp16.h>
#if defined(ENABLE_BF16)
#include <hip/hip_bfloat16.h>
#endif

namespace {

template<typename Scalar>
__device__ inline float to_float(Scalar value);

template<>
__device__ inline float to_float<half>(half value) {
    return __half2float(value);
}

#if defined(ENABLE_BF16)
template<>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
#endif

template<typename Scalar>
__global__ void gemm_batched_kernel(const Scalar *a,
                                    size_t lda,
                                    size_t stride_a,
                                    const Scalar *b,
                                    size_t ldb,
                                    size_t stride_b,
                                    float *out,
                                    size_t ldc,
                                    size_t stride_out,
                                    int batch,
                                    int M,
                                    int N,
                                    int K) {
    const int total = batch * M * N;
    const int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear_idx >= total) {
        return;
    }

    const int batch_idx = linear_idx / (M * N);
    const int inner     = linear_idx - batch_idx * (M * N);
    const int row       = inner / N;
    const int col       = inner - row * N;

    const Scalar *batch_a = a + batch_idx * stride_a + row * lda;
    const Scalar *batch_b = b + batch_idx * stride_b + col * ldb;
    float *batch_out      = out + batch_idx * stride_out + row * ldc;

    float accum = 0.0f;
    for (int k = 0; k < K; ++k) {
        accum += to_float(batch_a[k]) * to_float(batch_b[k]);
    }

    batch_out[col] = accum;
}

} // namespace

#else

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/numeric_types.h>

#endif

using spdlog::fmt_lib::format;

Tensor gemm_batched_fp16(Tensor a,  // FP16 row-major [(... batch ...), M, K]
                         Tensor b,  // FP16 col-major [(... batch ...), N, K]
                         Tensor out // FP32 row-major [(... batch ...), M, N]
) {
    const int M     = a.shape[-2];
    const int K     = a.shape[-1];
    const int N     = b.shape[-2];
    const int batch = a.numel() / (M * K);

#if defined(NUNCHAKU_USE_HIP)
    if (!out.valid()) {
        auto outShape = TensorShape(a.shape.dataExtent);
        outShape[-1]  = N;
        out           = Tensor::empty(outShape, Tensor::FP32, a.device());
    }

    assert(K == b.shape[-1]);
    assert(M == out.shape[-2]);
    assert(N == out.shape[-1]);

    assert(a.dtype() == Tensor::FP16);
    assert(b.dtype() == Tensor::FP16);
    assert(out.dtype() == Tensor::FP32);

    const size_t lda = a.stride(-2);
    const size_t ldb = b.stride(-2);
    const size_t ldc = out.stride(-2);
    const size_t stride_a = a.ndims() >= 3 ? a.stride(-3) : size_t(M) * K;
    const size_t stride_b = b.ndims() >= 3 ? b.stride(-3) : size_t(N) * K;
    const size_t stride_o = out.ndims() >= 3 ? out.stride(-3) : size_t(M) * N;

    constexpr int Threads = 256;
    const int blocks      = (int)((batch * M * N + Threads - 1) / Threads);

    hipLaunchKernelGGL((gemm_batched_kernel<half>),
                       dim3(blocks),
                       dim3(Threads),
                       0,
                       getCurrentGpuStream(),
                       a.data_ptr<half>(),
                       lda,
                       stride_a,
                       b.data_ptr<half>(),
                       ldb,
                       stride_b,
                       out.data_ptr<float>(),
                       ldc,
                       stride_o,
                       batch,
                       M,
                       N,
                       K);

    gpu_runtime::check(gpu_runtime::getLastError());
#else
    using ElementInput  = cutlass::half_t;
    using ElementOutput = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutO = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::GemmBatched<
        ElementInput,
        LayoutA,
        ElementInput,
        LayoutB,
        ElementOutput,
        LayoutO,
        ElementOutput,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<32, 32, 64>,
        cutlass::gemm::GemmShape<32, 32, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                     128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                     ElementOutput,
                                                     ElementOutput>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        2>;

    auto sizeA = cutlass::MatrixCoord(M, K);
    auto sizeB = cutlass::MatrixCoord(K, N);
    auto sizeO = cutlass::MatrixCoord(M, N);

    if (!out.valid()) {
        auto outShape = TensorShape(a.shape.dataExtent);
        outShape[-1]  = N;
        out           = Tensor::empty(outShape, Tensor::FP32, a.device());
    }

    assert(K == b.shape[-1]);
    assert(M == out.shape[-2]);
    assert(N == out.shape[-1]);

    assert(a.dtype() == Tensor::FP16);
    assert(a.dtype() == b.dtype());
    assert(out.dtype() == Tensor::FP32);

    cutlass::gemm::GemmCoord problemSize(M, N, K);

    cutlass::TensorRef<ElementInput, LayoutA> refA(a.data_ptr<ElementInput>(), LayoutA(a.stride(-2)));
    cutlass::TensorRef<ElementInput, LayoutB> refB(b.data_ptr<ElementInput>(), LayoutB(b.stride(-2)));
    cutlass::TensorRef<ElementOutput, LayoutO> refO(out.data_ptr<ElementOutput>(), LayoutO(out.stride(-2)));

    typename Gemm::Arguments arguments{problemSize,
                                       refA,
                                       (int)a.stride(-3),
                                       refB,
                                       (int)b.stride(-3),
                                       refO,
                                       (int)out.stride(-3),
                                       refO,
                                       (int)out.stride(-3),
                                       {ElementOutput(1), ElementOutput(0)},
                                       batch};

    Gemm op;
    BufferGPU workspace(Gemm::get_workspace_size(arguments));

    cutlass::Status status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(format("cutlass cannot implement M={} N={} K={}", M, N, K));
    }

    status = op.initialize(arguments, workspace.getPtr());
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot initialize");
    }

    status = op();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot run");
    }
#endif

    return out;
}
