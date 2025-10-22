#include "gemm_f16.h"

#include "dispatch_cutlass.h"

#if defined(NUNCHAKU_USE_HIP)

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
__device__ inline Scalar from_float(float value);

template<>
__device__ inline half from_float<half>(float value) {
    return __float2half(value);
}

#if defined(ENABLE_BF16)
template<>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}
#endif

template<typename Scalar>
__global__ void gemm_f16_kernel(const Scalar *input,
                                size_t lda,
                                const Scalar *weight,
                                size_t ldb,
                                Scalar *out,
                                size_t ldc,
                                const Scalar *bias,
                                int M,
                                int N,
                                int K,
                                float alpha,
                                bool has_bias) {
    const int total = M * N;
    const int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear_idx >= total) {
        return;
    }

    const int row = linear_idx / N;
    const int col = linear_idx - row * N;

    const Scalar *row_input  = input + row * lda;
    const Scalar *row_weight = weight + col * ldb;
    float accum              = 0.0f;
    for (int k = 0; k < K; ++k) {
        accum += to_float(row_input[k]) * to_float(row_weight[k]);
    }

    float value = alpha * accum;
    if (has_bias) {
        value += to_float(bias[col]);
    }

    Scalar *row_out = out + row * ldc;
    row_out[col]    = from_float<Scalar>(value);
}

} // namespace

#else

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

#endif

using spdlog::fmt_lib::format;

Tensor gemm_f16(Tensor input,  // FP16
                Tensor weight, // FP16
                Tensor out,    // FP16
                Tensor bias,
                float alpha) {
    auto N = weight.size(0);
    auto K = input.size(-1);
    auto M = input.numel() / K;
    assert(weight.size(1) == K);

    spdlog::debug("gemm_f16: M={} K={} N={}", M, K, N);

    dispatchF16(weight.dtype(), [&]<typename scalar_t>() {
#if defined(NUNCHAKU_USE_HIP)
        using ElementInputA  = scalar_t;
        using ElementInputB  = scalar_t;
        using ElementOutput  = scalar_t;
        const size_t lda     = input.stride(-2);
        const size_t ldb     = weight.stride(-2);
        const size_t ldc     = out.stride(-2);
        const bool has_bias  = bias.valid();
        const ElementOutput *bias_ptr = has_bias ? bias.data_ptr<ElementOutput>() : nullptr;

        constexpr int Threads = 256;
        const int blocks      = (int)((M * N + Threads - 1) / Threads);

        hipLaunchKernelGGL((gemm_f16_kernel<ElementInputA>),
                           dim3(blocks),
                           dim3(Threads),
                           0,
                           getCurrentGpuStream(),
                           input.data_ptr<ElementInputA>(),
                           lda,
                           weight.data_ptr<ElementInputB>(),
                           ldb,
                           out.data_ptr<ElementOutput>(),
                           ldc,
                           bias_ptr,
                           M,
                           N,
                           K,
                           alpha,
                           has_bias);

        gpu_runtime::check(gpu_runtime::getLastError());
#else
        using ElementOutput          = half_t;
        using ElementAccumulator     = float;
        using ElementComputeEpilogue = half_t;
        using ElementInputA          = half_t; // <- data type of elements in input matrix A
        using ElementInputB          = half_t; // <- data type of elements in input matrix B

        using LayoutInputA = cutlass::layout::RowMajor;
        using LayoutInputB = cutlass::layout::ColumnMajor;
        using LayoutOutput = cutlass::layout::RowMajor;

        // #if CUDA_ARCH >= 800
        using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                                 cutlass::layout::RowMajor,
                                                 ElementInputB,
                                                 cutlass::layout::ColumnMajor,
                                                 ElementOutput,
                                                 cutlass::layout::RowMajor,
                                                 ElementAccumulator,
                                                 cutlass::arch::OpClassTensorOp,
                                                 cutlass::arch::Sm75>;
        // cutlass::gemm::GemmShape<128, 128, 64>,
        // cutlass::gemm::GemmShape<32, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
        // cutlass::epilogue::thread::LinearCombination<
        //     ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        //     ElementAccumulator, ElementComputeEpilogue>,
        // cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

        auto input_size  = cutlass::MatrixCoord(M, K);
        auto weight_size = cutlass::MatrixCoord(K, N);
        auto output_size = cutlass::MatrixCoord(M, N);

        auto device = input.device();
        // use the broadcasted bias as the output
        // auto out = bias.to(device).view({1, -1}).repeat({M, 1});

        if (!out.valid()) {
            auto out_shape = TensorShape(input.shape.dataExtent);
            out_shape[-1]  = N;
            out            = Tensor::empty(out_shape, input.scalar_type(), input.device());
        }

        // FIXME: check contiguous of input if dims >= 3
        assert(input.stride(-1) == 1);
        // assert(input.is_contiguous());
        assert(weight.is_contiguous());

        assert(out.dtype() == input.scalar_type());
        assert(out.shape[-1] == N);
        assert(out.numel() / out.shape[-1] == M);
        assert(out.stride(-1) == 1);
        // FIXME: check contiguous of output if dims >= 3

        assert(!bias.valid() || (bias.ndims() == 1 && bias.shape[0] == N));

        // constexpr int kSparse = Gemm::kSparse;
        // How many elements of A are covered per ElementE
        // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
        // The size of individual meta data
        // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(input.data_ptr<ElementInputA>(),
                                                                  LayoutInputA(input.stride(-2)));
        cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(weight.data_ptr<ElementInputB>(),
                                                                   LayoutInputB::packed(weight_size));
        cutlass::TensorRef<ElementOutput, LayoutOutput> bias_ref(
            bias.valid() ? bias.data_ptr<ElementOutput>() : out.data_ptr<ElementOutput>(), LayoutOutput(0));
        cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(out.data_ptr<ElementOutput>(),
                                                                LayoutOutput(out.stride(-2)));

        typename Gemm::Arguments arguments{problem_size, // <- problem size of matrix multiplication
                                           input_ref,    // <- reference to matrix A on device
                                           weight_ref,   // <- reference to matrix B on device
                                           bias_ref,     // <- reference to matrix C on device
                                           out_ref,      // <- reference to matrix D on device
                                           {ElementOutput(alpha), ElementOutput(bias.valid() ? 1.0f : 0.0f)},
                                           1};
        Gemm gemm_op;

        // Using the arguments, query for extra workspace required for matrix
        // multiplication computation
        size_t workspace_size = Gemm::get_workspace_size(arguments);

        // Allocate workspace memory
        // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        BufferGPU workspace(workspace_size);

        // Check the problem size is supported or not
        cutlass::Status status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error(format("cutlass cannot implement M={} N={} K={}", M, N, K));
        }

        // Initialize CUTLASS kernel with arguments and workspace pointer
        status = gemm_op.initialize(arguments, workspace.getPtr());
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("cutlass cannot initialize");
        }

        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("cutlass cannot run");
        }
#endif
    });

    return out;
}
