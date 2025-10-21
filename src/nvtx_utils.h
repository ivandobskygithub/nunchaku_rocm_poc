#pragma once

#if !defined(NUNCHAKU_ENABLE_NVTX) && !defined(NUNCHAKU_USE_HIP)
#define NUNCHAKU_ENABLE_NVTX 1
#endif

#if defined(NUNCHAKU_ENABLE_NVTX)
#include <nvtx3/nvToolsExt.h>
#define NUNCHAKU_NVTX_PUSH_RANGE(name) nvtxRangePushA(name)
#define NUNCHAKU_NVTX_POP_RANGE() nvtxRangePop()
#else
#define NUNCHAKU_NVTX_PUSH_RANGE(name) ((void)0)
#define NUNCHAKU_NVTX_POP_RANGE() ((void)0)
#endif

