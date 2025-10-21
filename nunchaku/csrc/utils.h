#pragma once

#include "common.h"
#include "Tensor.h"
#include "kernels/zgemm/zgemm.h"

namespace nunchaku::utils {

void set_cuda_stack_limit(int64_t newval) {
    size_t val = 0;
    checkCUDA(gpu_runtime::deviceSetLimit(gpu_runtime::LimitStackSize, (size_t)newval));
    checkCUDA(gpu_runtime::deviceGetLimit(&val, gpu_runtime::LimitStackSize));
    spdlog::debug("Stack={}", val);
}

void disable_memory_auto_release() {
    int device;
    checkCUDA(gpu_runtime::getDevice(&device));
    gpu_runtime::MemPool mempool;
    checkCUDA(gpu_runtime::deviceGetDefaultMemPool(&mempool, device));
    uint64_t threshold = UINT64_MAX;
    checkCUDA(gpu_runtime::memPoolSetAttribute(
        mempool, gpu_runtime::MemPoolAttrReleaseThreshold, &threshold));
}

void trim_memory() {
    int device;
    checkCUDA(gpu_runtime::getDevice(&device));
    gpu_runtime::MemPool mempool;
    checkCUDA(gpu_runtime::deviceGetDefaultMemPool(&mempool, device));
    size_t bytesToKeep = 0;
    checkCUDA(gpu_runtime::memPoolTrimTo(mempool, bytesToKeep));
}

void set_faster_i2f_mode(std::string mode) {
    spdlog::info("Set fasteri2f mode to {}", mode);
    kernels::set_faster_i2f_mode(mode);
}

}; // namespace nunchaku::utils
