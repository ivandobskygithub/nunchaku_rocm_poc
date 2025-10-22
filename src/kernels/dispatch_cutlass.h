#pragma once

#include "common.h"
#include "Tensor.h"

#if defined(NUNCHAKU_USE_HIP)

#include <hip/hip_fp16.h>
#if defined(ENABLE_BF16)
#include <hip/hip_bfloat16.h>
#endif

template<typename F>
inline void dispatchF16(Tensor::ScalarType type, F &&func) {
    if (type == Tensor::FP16) {
        func.template operator()<half>();
#if defined(ENABLE_BF16)
    } else if (type == Tensor::BF16) {
        func.template operator()<__nv_bfloat16>();
#endif
    } else {
        assert(false);
    }
}

#else

#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>

template<typename F>
inline void dispatchF16(Tensor::ScalarType type, F &&func) {
    if (type == Tensor::FP16) {
        func.template operator()<cutlass::half_t>();
    } else if (type == Tensor::BF16) {
        func.template operator()<cutlass::bfloat16_t>();
    } else {
        assert(false);
    }
}

#endif
