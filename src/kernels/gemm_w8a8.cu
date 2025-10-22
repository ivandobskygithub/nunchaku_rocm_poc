#include "common.h"
#include "Tensor.h"

#if defined(NUNCHAKU_USE_HIP)

#include "interop/torch.h"

#include <ATen/ATen.h>

namespace {

inline float half_to_float(half value) {
#if defined(__HIP_PLATFORM_AMD__)
    return __half2float(value);
#else
    return __half2float(value);
#endif
}

} // namespace

Tensor gemm_w8a8_fp16(Tensor input,
                      Tensor weight,
                      Tensor out,
                      half alpha,
                      half beta) {
    TorchOpContext ctx;

    auto input_t  = to_torch(input).contiguous();
    auto weight_t = to_torch(weight).contiguous();

    auto input_dtype = input_t.scalar_type();
    auto device      = input_t.device();

    auto input_f  = input_t.to(torch::kFloat32);
    auto weight_f = weight_t.to(torch::kFloat32);

    auto product = torch::matmul(input_f, weight_f.transpose(-1, -2));

    const float alpha_f = half_to_float(alpha);
    const float beta_f  = half_to_float(beta);

    product.mul_(alpha_f);

    at::Tensor result_f = product;

    if (out.valid() && beta_f != 0.0f) {
        auto out_t = to_torch(out).to(torch::kFloat32).contiguous();
        result_f.add_(out_t, beta_f);
    }

    auto result_t = result_f.to(input_dtype).to(device);

    if (out.valid()) {
        out.copy_(from_torch(result_t));
        return out;
    }

    return from_torch(result_t);
}

#else

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

using spdlog::fmt_lib::format;

Tensor gemm_w8a8_fp16(Tensor input,  // INT8
                      Tensor weight, // INT8
                      Tensor out,    // FP16
                      half alpha,
                      half beta // FP16
) {
    auto N = weight.size(0);
    auto K = input.size(-1);
    auto M = input.numel() / K;
    assert(weight.size(1) == K);

    spdlog::debug("gemm_w8a8: M={} K={} N={}", M, K, N);

    using ElementOutput          = cutlass::half_t;
    using ElementAccumulator     = int32_t;
    using ElementComputeEpilogue = cutlass::half_t;
    using ElementInputA          = int8_t; // <- data type of elements in input matrix A
    using ElementInputB          = int8_t; // <- data type of elements in input matrix B

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // #if CUDA_ARCH >= 800
    using Gemm = cutlass::gemm::device::Gemm<
        int8_t,
        cutlass::layout::RowMajor,
        int8_t,
        cutlass::layout::ColumnMajor,
        ElementOutput,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<32, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                     128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                     ElementAccumulator,
                                                     ElementComputeEpilogue>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3>;
    // #elif CUDA_ARCH >= 750
    //     using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
    //         cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    //         ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
    //     using Gemm = cutlass::gemm::device::Gemm<
    //         int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
    //         ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
    //         cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    //         DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
    //         DefaultGemmCfg::InstructionShape,
    //         cutlass::epilogue::thread::LinearCombination<
    //             ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    //             ElementAccumulator, ElementComputeEpilogue>>;
    // #elif CUDA_ARCH >= 700
    //     using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
    //         cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
    //         ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
    //     using Gemm = cutlass::gemm::device::Gemm<
    //         int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
    //         ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
    //         cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
    //         DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
    //         DefaultGemmCfg::InstructionShape,
    //         cutlass::epilogue::thread::LinearCombination<
    //             ElementOutput, 1, ElementAccumulator, ElementComputeEpilogue>>;
    // #else
    // #error "Unsupported cuda arch"
    // #endif

    auto input_size  = cutlass::MatrixCoord(M, K);
    auto weight_size = cutlass::MatrixCoord(K, N);
    auto output_size = cutlass::MatrixCoord(M, N);

    auto device = input.device();
    // use the broadcasted bias as the output
    // auto out = bias.to(device).view({1, -1}).repeat({M, 1});

    if (!out.valid()) {
        auto out_shape = TensorShape(input.shape.dataExtent);
        out_shape[-1]  = N;
        out            = Tensor::empty(out_shape, Tensor::FP16, input.device());
    }

    // FIXME: check contiguous of input if dims >= 3
    assert(input.stride(-1) == 1);
    // assert(input.is_contiguous());
    assert(weight.is_contiguous());

    assert(out.dtype() == Tensor::FP16);
    assert(out.shape[-1] == N);
    assert(out.numel() / out.shape[-1] == M);
    assert(out.stride(-1) == 1);
    // FIXME: check contiguous of output if dims >= 3

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
    cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(out.data_ptr<ElementOutput>(),
                                                            LayoutOutput(out.stride(-2)));

    typename Gemm::Arguments arguments{problem_size, // <- problem size of matrix multiplication
                                       input_ref,    // <- reference to matrix A on device
                                       weight_ref,   // <- reference to matrix B on device
                                       out_ref,      // <- reference to matrix C on device
                                       out_ref,      // <- reference to matrix D on device
                                       {ElementOutput(alpha), ElementOutput(beta)},
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

    return out;
}

#endif
