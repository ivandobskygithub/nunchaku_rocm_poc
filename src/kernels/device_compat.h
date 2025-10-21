#pragma once

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifdef ENABLE_BF16
#include <hip/hip_bfloat16.h>
#endif // ENABLE_BF16

#if defined(__HIP_DEVICE_COMPILE__)
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ __HIP_ARCH__
#endif
#endif

#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif // ENABLE_BF16
#endif

#if defined(__HIP_PLATFORM_AMD__) && defined(ENABLE_BF16)
using __nv_bfloat16  = hip_bfloat16;
using __nv_bfloat162 = hip_bfloat162;
#endif
