#pragma once

#include <cstddef>
#include <source_location>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

#if defined(NUNCHAKU_USE_HIP)
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
namespace gpu_runtime {
using Error          = hipError_t;
using Stream         = hipStream_t;
using Event          = hipEvent_t;
using DeviceProp     = hipDeviceProp_t;
using MemcpyKind     = hipMemcpyKind;
inline constexpr Error success                  = hipSuccess;
inline constexpr unsigned int HostAllocPortable = hipHostMallocPortable;
inline constexpr unsigned int HostRegisterPortable = hipHostRegisterPortable;
inline constexpr unsigned int HostRegisterReadOnly = hipHostRegisterReadOnly;
inline constexpr unsigned int EventDefault         = hipEventDefault;
inline constexpr MemcpyKind MemcpyHostToDevice     = hipMemcpyHostToDevice;
inline constexpr MemcpyKind MemcpyDeviceToHost     = hipMemcpyDeviceToHost;
inline constexpr MemcpyKind MemcpyDeviceToDevice   = hipMemcpyDeviceToDevice;
inline constexpr MemcpyKind MemcpyHostToHost       = hipMemcpyHostToHost;
inline constexpr MemcpyKind MemcpyDefault          = hipMemcpyDefault;
inline const char *getErrorString(Error error) { return hipGetErrorString(error); }
inline Error getLastError() { return hipGetLastError(); }
inline Error hostAlloc(void **ptr, size_t size, unsigned int flags) { return hipHostMalloc(ptr, size, flags); }
inline Error freeHost(void *ptr) { return hipHostFree(ptr); }
inline Error hostRegister(void *ptr, size_t size, unsigned int flags) {
    return hipHostRegister(ptr, size, flags);
}
inline Error hostUnregister(void *ptr) { return hipHostUnregister(ptr); }
inline Error mallocAsync(void **ptr, size_t size, Stream stream) {
    return hipMallocAsync(ptr, size, stream);
}
inline Error freeAsync(void *ptr, Stream stream) { return hipFreeAsync(ptr, stream); }
inline Error mallocMemory(void **ptr, size_t size) { return hipMalloc(ptr, size); }
inline Error freeMemory(void *ptr) { return hipFree(ptr); }
inline Error memsetAsync(void *ptr, int value, size_t count, Stream stream) {
    return hipMemsetAsync(ptr, value, count, stream);
}
inline Error memcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream stream) {
    return hipMemcpyAsync(dst, src, count, kind, stream);
}
inline Error memcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                           MemcpyKind kind, Stream stream) {
    return hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
}
inline Error memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
    return hipMemcpy(dst, src, count, kind);
}
inline Error streamCreate(Stream *stream) { return hipStreamCreate(stream); }
inline Error streamDestroy(Stream stream) { return hipStreamDestroy(stream); }
inline Error streamQuery(Stream stream) { return hipStreamQuery(stream); }
inline Error eventCreateWithFlags(Event *event, unsigned int flags) {
    return hipEventCreateWithFlags(event, flags);
}
inline Error eventDestroy(Event event) { return hipEventDestroy(event); }
inline Error eventRecord(Event event, Stream stream) { return hipEventRecord(event, stream); }
inline Error streamWaitEvent(Stream stream, Event event) { return hipStreamWaitEvent(stream, event, 0); }
inline Error eventSynchronize(Event event) { return hipEventSynchronize(event); }
inline Error getDevice(int *device) { return hipGetDevice(device); }
inline Error setDevice(int device) { return hipSetDevice(device); }
inline Error deviceSynchronize() { return hipDeviceSynchronize(); }
inline Error streamSynchronize(Stream stream) { return hipStreamSynchronize(stream); }
inline Error getDeviceProperties(DeviceProp *prop, int device) {
    return hipGetDeviceProperties(prop, device);
}
} // namespace gpu_runtime

namespace gpu_blas {
using Handle      = hipblasHandle_t;
using Status      = hipblasStatus_t;
inline constexpr Status statusSuccess = HIPBLAS_STATUS_SUCCESS;
inline const char *getStatusString(Status status) {
    switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
        return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:
        return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:
        return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:
        return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_ARCH_MISMATCH:
        return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_MAPPING_ERROR:
        return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:
        return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:
        return "HIPBLAS_STATUS_INTERNAL_ERROR";
    default:
        return "HIPBLAS_STATUS_UNKNOWN";
    }
}
inline Status create(Handle *handle) { return hipblasCreate(handle); }
inline Status destroy(Handle handle) { return hipblasDestroy(handle); }
using Operation = hipblasOperation_t;
inline constexpr Operation OperationN = HIPBLAS_OP_N;
inline constexpr Operation OperationT = HIPBLAS_OP_T;
inline Status hgemm(Handle handle,
                    Operation transa,
                    Operation transb,
                    int m,
                    int n,
                    int k,
                    const void *alpha,
                    const void *A,
                    int lda,
                    const void *B,
                    int ldb,
                    const void *beta,
                    void *C,
                    int ldc) {
    return hipblasHgemm(handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        reinterpret_cast<const hipblasHalf *>(alpha),
                        reinterpret_cast<const hipblasHalf *>(A),
                        lda,
                        reinterpret_cast<const hipblasHalf *>(B),
                        ldb,
                        reinterpret_cast<const hipblasHalf *>(beta),
                        reinterpret_cast<hipblasHalf *>(C),
                        ldc);
}
} // namespace gpu_blas

#else
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
namespace gpu_runtime {
using Error          = cudaError_t;
using Stream         = cudaStream_t;
using Event          = cudaEvent_t;
using DeviceProp     = cudaDeviceProp;
using MemcpyKind     = cudaMemcpyKind;
inline constexpr Error success                  = cudaSuccess;
inline constexpr unsigned int HostAllocPortable = cudaHostAllocPortable;
inline constexpr unsigned int HostRegisterPortable = cudaHostRegisterPortable;
inline constexpr unsigned int HostRegisterReadOnly = cudaHostRegisterReadOnly;
inline constexpr unsigned int EventDefault         = cudaEventDefault;
inline constexpr MemcpyKind MemcpyHostToDevice     = cudaMemcpyHostToDevice;
inline constexpr MemcpyKind MemcpyDeviceToHost     = cudaMemcpyDeviceToHost;
inline constexpr MemcpyKind MemcpyDeviceToDevice   = cudaMemcpyDeviceToDevice;
inline constexpr MemcpyKind MemcpyHostToHost       = cudaMemcpyHostToHost;
inline constexpr MemcpyKind MemcpyDefault          = cudaMemcpyDefault;
inline const char *getErrorString(Error error) { return cudaGetErrorString(error); }
inline Error getLastError() { return cudaGetLastError(); }
inline Error hostAlloc(void **ptr, size_t size, unsigned int flags) { return cudaHostAlloc(ptr, size, flags); }
inline Error freeHost(void *ptr) { return cudaFreeHost(ptr); }
inline Error hostRegister(void *ptr, size_t size, unsigned int flags) {
    return cudaHostRegister(ptr, size, flags);
}
inline Error hostUnregister(void *ptr) { return cudaHostUnregister(ptr); }
inline Error mallocAsync(void **ptr, size_t size, Stream stream) {
    return cudaMallocAsync(ptr, size, stream);
}
inline Error freeAsync(void *ptr, Stream stream) { return cudaFreeAsync(ptr, stream); }
inline Error mallocMemory(void **ptr, size_t size) { return cudaMalloc(ptr, size); }
inline Error freeMemory(void *ptr) { return cudaFree(ptr); }
inline Error memsetAsync(void *ptr, int value, size_t count, Stream stream) {
    return cudaMemsetAsync(ptr, value, count, stream);
}
inline Error memcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream stream) {
    return cudaMemcpyAsync(dst, src, count, kind, stream);
}
inline Error memcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                           MemcpyKind kind, Stream stream) {
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
}
inline Error memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
    return cudaMemcpy(dst, src, count, kind);
}
inline Error streamCreate(Stream *stream) { return cudaStreamCreate(stream); }
inline Error streamDestroy(Stream stream) { return cudaStreamDestroy(stream); }
inline Error streamQuery(Stream stream) { return cudaStreamQuery(stream); }
inline Error eventCreateWithFlags(Event *event, unsigned int flags) {
    return cudaEventCreateWithFlags(event, flags);
}
inline Error eventDestroy(Event event) { return cudaEventDestroy(event); }
inline Error eventRecord(Event event, Stream stream) { return cudaEventRecord(event, stream); }
inline Error streamWaitEvent(Stream stream, Event event) { return cudaStreamWaitEvent(stream, event); }
inline Error eventSynchronize(Event event) { return cudaEventSynchronize(event); }
inline Error getDevice(int *device) { return cudaGetDevice(device); }
inline Error setDevice(int device) { return cudaSetDevice(device); }
inline Error deviceSynchronize() { return cudaDeviceSynchronize(); }
inline Error streamSynchronize(Stream stream) { return cudaStreamSynchronize(stream); }
inline Error getDeviceProperties(DeviceProp *prop, int device) {
    return cudaGetDeviceProperties(prop, device);
}
} // namespace gpu_runtime

namespace gpu_blas {
using Handle      = cublasHandle_t;
using Status      = cublasStatus_t;
inline constexpr Status statusSuccess = CUBLAS_STATUS_SUCCESS;
inline const char *getStatusString(Status status) { return cublasGetStatusString(status); }
inline Status create(Handle *handle) { return cublasCreate(handle); }
inline Status destroy(Handle handle) { return cublasDestroy(handle); }
using Operation = cublasOperation_t;
inline constexpr Operation OperationN = CUBLAS_OP_N;
inline constexpr Operation OperationT = CUBLAS_OP_T;
inline Status hgemm(Handle handle,
                    Operation transa,
                    Operation transb,
                    int m,
                    int n,
                    int k,
                    const void *alpha,
                    const void *A,
                    int lda,
                    const void *B,
                    int ldb,
                    const void *beta,
                    void *C,
                    int ldc) {
    return cublasHgemm(handle,
                       transa,
                       transb,
                       m,
                       n,
                       k,
                       reinterpret_cast<const __half *>(alpha),
                       reinterpret_cast<const __half *>(A),
                       lda,
                       reinterpret_cast<const __half *>(B),
                       ldb,
                       reinterpret_cast<const __half *>(beta),
                       reinterpret_cast<__half *>(C),
                       ldc);
}
} // namespace gpu_blas

#endif

namespace gpu_runtime {
class ErrorException : public std::runtime_error {
public:
    ErrorException(Error errorCode, std::source_location location)
        : std::runtime_error(format(errorCode, location)), errorCode(errorCode), location(location) {}

    Error errorCode;
    std::source_location location;

private:
    static std::string format(Error errorCode, std::source_location location) {
        return spdlog::fmt_lib::format(
            "GPU runtime error: {} (at {}:{})", getErrorString(errorCode), location.file_name(), location.line());
    }
};

inline Error check(Error result, std::source_location location = std::source_location::current()) {
    if (result != success) {
        (void)getLastError();
        throw ErrorException(result, location);
    }
    return result;
}
} // namespace gpu_runtime

namespace gpu_blas {
class ErrorException : public std::runtime_error {
public:
    ErrorException(Status status, std::source_location location)
        : std::runtime_error(format(status, location)), status(status), location(location) {}

    Status status;
    std::source_location location;

private:
    static std::string format(Status status, std::source_location location) {
        return spdlog::fmt_lib::format(
            "GPU BLAS error: {} (at {}:{})", getStatusString(status), location.file_name(), location.line());
    }
};

inline Status check(Status status, std::source_location location = std::source_location::current()) {
    if (status != statusSuccess) {
        throw ErrorException(status, location);
    }
    return status;
}
} // namespace gpu_blas

