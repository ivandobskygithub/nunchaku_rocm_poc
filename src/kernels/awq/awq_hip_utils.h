#pragma once

#if defined(__HIP_PLATFORM_AMD__)

#include "../../interop/torch.h"

#include <ATen/ATen.h>

namespace nunchaku::kernels::awq_hip {

inline at::Tensor build_awq_weight(at::Tensor packed,
                                   at::Tensor scales,
                                   at::Tensor zeros,
                                   int group_size) {
    packed = packed.to(torch::kCPU, /*non_blocking=*/false, /*copy=*/true).to(torch::kInt32).contiguous();
    scales = scales.to(torch::kCPU, /*non_blocking=*/false, /*copy=*/true).to(torch::kFloat32).contiguous();
    zeros  = zeros.to(torch::kCPU, /*non_blocking=*/false, /*copy=*/true).to(torch::kFloat32).contiguous();

    const int rows         = packed.size(0);
    const int cols         = packed.size(1);
    const int out_features = rows * 4;
    const int in_features  = cols;

    auto result = torch::empty({out_features, in_features}, torch::dtype(torch::kFloat32));

    auto packed_ptr = packed.data_ptr<int32_t>();
    auto scales_ptr = scales.data_ptr<float>();
    auto zeros_ptr  = zeros.data_ptr<float>();
    auto result_ptr = result.data_ptr<float>();

    const int padded_groups = scales.size(0);
    const int scale_stride  = scales.size(1);
    const int groups        = group_size > 0 ? group_size : in_features;
    const int ic_div64      = in_features / 64;

    if (padded_groups <= 0 || scale_stride != out_features) {
        throw std::runtime_error("Invalid AWQ scale tensor shape for ROCm fallback");
    }

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            uint16_t value = static_cast<uint16_t>(packed_ptr[row * cols + col] & 0xFFFF);

            const int tmp = col / 16;
            const int d   = col % 16;
            const int c   = tmp / 4;
            const int b   = tmp % 4;

            const int index1 = ((row * 4 + b) * ic_div64 + c) * 16 + d;
            const int block  = index1 / 8;
            const int j      = index1 % 8;

            for (int i = 0; i < 4; ++i) {
                const int nibble = (value >> (4 * i)) & 0xF;
                const int p      = block * 32 + i * 8 + j;
                const int oc_idx = p / in_features;
                const int ic_idx = p % in_features;
                int group_idx    = groups > 0 ? ic_idx / groups : 0;
                if (group_idx >= padded_groups) {
                    group_idx = padded_groups - 1;
                }
                const float scale_val = scales_ptr[group_idx * scale_stride + oc_idx];
                const float zero_val  = zeros_ptr[group_idx * scale_stride + oc_idx];
                result_ptr[oc_idx * in_features + ic_idx] = nibble * scale_val + zero_val;
            }
        }
    }

    return result;
}

} // namespace nunchaku::kernels::awq_hip

#endif // defined(__HIP_PLATFORM_AMD__)

