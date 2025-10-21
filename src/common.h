#pragma once

#include <cstddef>
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <source_location>
#include <vector>
#include <list>
#include <stack>
#include <map>
#include <unordered_map>
#include <set>
#include <any>
#include <variant>
#include <optional>
#include <chrono>
#include <functional>
#include <spdlog/spdlog.h>

#include "gpu_runtime.h"

inline thread_local std::stack<gpu_runtime::Stream> stackGpuStreams;

inline gpu_runtime::Stream getCurrentGpuStream() {
    if (stackGpuStreams.empty()) {
        return nullptr;
    }
    return stackGpuStreams.top();
}

struct GpuStreamContext {
    gpu_runtime::Stream stream;

    explicit GpuStreamContext(gpu_runtime::Stream stream) : stream(stream) {
        stackGpuStreams.push(stream);
    }
    GpuStreamContext(const GpuStreamContext &)            = delete;
    GpuStreamContext &operator=(const GpuStreamContext &) = delete;
    GpuStreamContext(GpuStreamContext &&)                 = delete;
    GpuStreamContext &operator=(GpuStreamContext &&)      = delete;

    ~GpuStreamContext() {
        assert(stackGpuStreams.top() == stream);
        stackGpuStreams.pop();
    }
};

struct GpuStreamWrapper {
    gpu_runtime::Stream stream;

    GpuStreamWrapper() {
        gpu_runtime::check(gpu_runtime::streamCreate(&stream));
    }
    GpuStreamWrapper(const GpuStreamWrapper &)            = delete;
    GpuStreamWrapper &operator=(const GpuStreamWrapper &) = delete;
    GpuStreamWrapper(GpuStreamWrapper &&)                 = delete;
    GpuStreamWrapper &operator=(GpuStreamWrapper &&)      = delete;

    ~GpuStreamWrapper() {
        gpu_runtime::check(gpu_runtime::streamDestroy(stream));
    }
};

struct GpuEventWrapper {
    gpu_runtime::Event event;

    explicit GpuEventWrapper(unsigned int flags = gpu_runtime::EventDefault) {
        gpu_runtime::check(gpu_runtime::eventCreateWithFlags(&event, flags));
    }
    GpuEventWrapper(const GpuEventWrapper &)            = delete;
    GpuEventWrapper &operator=(const GpuEventWrapper &) = delete;
    GpuEventWrapper(GpuEventWrapper &&)                 = delete;
    GpuEventWrapper &operator=(GpuEventWrapper &&)      = delete;

    ~GpuEventWrapper() {
        gpu_runtime::check(gpu_runtime::eventDestroy(event));
    }
};

/**
 * 1. hold one when entered from external code (set `device` to -1 to avoid device change)
 * 2. hold one when switching device
 * 3. hold one with `disableCache` when calling external code that may change the device
 */
class GpuDeviceContext {
public:
    GpuDeviceContext(int device = -1, bool disableCache = false) : disableCache(disableCache) {
        if (cacheDisabled()) {
            // no previous context => we might entered from external code, reset cache
            // previous context is reset on => external code may be executed, reset
            currentDeviceCache = -1;
        }

        ctxs.push(this);
        lastDevice = getDevice();
        if (device >= 0) {
            setDevice(device);
        }

        if (disableCache) {
            // we are about to call external code, reset cache
            currentDeviceCache = -1;
        }
    }
    GpuDeviceContext(const GpuDeviceContext &)            = delete;
    GpuDeviceContext &operator=(const GpuDeviceContext &) = delete;
    GpuDeviceContext(GpuDeviceContext &&)                 = delete;
    GpuDeviceContext &operator=(GpuDeviceContext &&)      = delete;

    ~GpuDeviceContext() {
        if (disableCache) {
            // retured from external code, cache is not reliable, reset
            currentDeviceCache = -1;
        }

        setDevice(lastDevice);
        assert(ctxs.top() == this);
        ctxs.pop();

        if (cacheDisabled()) {
            // ctxs.empty() => we are about to return to external code, reset cache
            // otherwise => we are a nested context in a previous context with reset on, we might continue to execute
            // external code, reset
            currentDeviceCache = -1;
        }
    }

    const bool disableCache;
    int lastDevice;

public:
    static int getDevice() {
        int idx = -1;
        if (cacheDisabled() || currentDeviceCache < 0) {
            gpu_runtime::check(gpu_runtime::getDevice(&idx));
        } else {
            idx = currentDeviceCache;
        }
        currentDeviceCache = cacheDisabled() ? -1 : idx;
        return idx;
    }

private:
    static void setDevice(int idx) {
        // TODO: deal with stream when switching device
        assert(idx >= 0);
        if (!cacheDisabled() && currentDeviceCache == idx) {
            return;
        }
        gpu_runtime::check(gpu_runtime::setDevice(idx));
        currentDeviceCache = cacheDisabled() ? -1 : idx;
    }

private:
    static inline thread_local std::stack<GpuDeviceContext *> ctxs;
    static inline thread_local int currentDeviceCache = -1;

    static bool cacheDisabled() {
        return ctxs.empty() || ctxs.top()->disableCache;
    }
};

inline gpu_runtime::DeviceProp *getCurrentDeviceProperties() {
    static thread_local std::map<int, gpu_runtime::DeviceProp> props;

    int deviceId = GpuDeviceContext::getDevice();
    if (!props.contains(deviceId)) {
        gpu_runtime::DeviceProp prop;
        gpu_runtime::check(gpu_runtime::getDeviceProperties(&prop, deviceId));
        props[deviceId] = prop;
    }
    return &props.at(deviceId);
}

template<typename T>
constexpr T ceilDiv(T a, T b) {
    return (a + b - 1) / b;
}

template<typename T>
constexpr int log2Up(T value) {
    if (value <= 0)
        return 0;
    if (value == 1)
        return 0;
    return log2Up((value + 1) / 2) + 1;
}

struct GpuBlasWrapper {
    gpu_blas::Handle handle = nullptr;

    GpuBlasWrapper() {
        gpu_blas::check(gpu_blas::create(&handle));
    }
    GpuBlasWrapper(GpuBlasWrapper &&)            = delete;
    GpuBlasWrapper &operator=(GpuBlasWrapper &&) = delete;
    GpuBlasWrapper(const GpuBlasWrapper &)       = delete;
    GpuBlasWrapper &operator=(const GpuBlasWrapper &) = delete;
    ~GpuBlasWrapper() {
        if (handle) {
            gpu_blas::check(gpu_blas::destroy(handle));
        }
    }
};

inline std::shared_ptr<GpuBlasWrapper> getGpuBlas() {
    static thread_local std::weak_ptr<GpuBlasWrapper> inst;
    std::shared_ptr<GpuBlasWrapper> result = inst.lock();
    if (result) {
        return result;
    }
    result = std::make_shared<GpuBlasWrapper>();
    inst   = result;
    return result;
}

inline gpu_runtime::Error
checkCUDA(gpu_runtime::Error value, std::source_location location = std::source_location::current()) {
    return gpu_runtime::check(value, location);
}

inline gpu_blas::Status
checkCUBLAS(gpu_blas::Status value, std::source_location location = std::source_location::current()) {
    return gpu_blas::check(value, location);
}
