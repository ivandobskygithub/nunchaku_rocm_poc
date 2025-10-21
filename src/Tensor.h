#pragma once

#include "common.h"

struct Device {
    enum Type { INVALID_DEVICE_TYPE = 0, CPU, CUDA, ROCM };

    Type type = INVALID_DEVICE_TYPE;
    int idx   = 0;

    static constexpr Device cpu(int idx = 0) {
        return Device{CPU, idx};
    }
    static constexpr Device cuda(int idx = 0) {
        return Device{CUDA, idx};
    }
    static constexpr Device rocm(int idx = 0) {
        return Device{ROCM, idx};
    }
    static constexpr Device gpu(int idx = 0) {
#if defined(NUNCHAKU_USE_HIP)
        return rocm(idx);
#else
        return cuda(idx);
#endif
    }

    bool is_gpu() const {
        return type == CUDA || type == ROCM;
    }
};

// template<bool readonly>
class Buffer : public std::enable_shared_from_this<Buffer> {
public:
    virtual ~Buffer() {}

    void *getPtr() {
        return ptr;
    }

    template<typename T>
    T *getPtr() {
        return reinterpret_cast<T *>(ptr);
    }

    size_t getSize() {
        return size;
    }
    Device getDevice() {
        return device;
    }

    virtual bool isAsyncBuffer() {
        return false;
    }

protected:
    template<typename Derived>
    std::shared_ptr<Derived> shared_from_base() {
        return std::static_pointer_cast<Derived>(shared_from_this());
    }

protected:
    // std::conditional_t<readonly, const void *, void *> ptr;
    void *ptr;
    size_t size;
    Device device;
};

// using Buffer = BufferTemplate<false>;
// using BufferReadonly = BufferTemplate<true>;

class BufferMalloc : public Buffer {
public:
    BufferMalloc(size_t size) {
        this->size        = size;
        this->device.type = Device::CPU;
        this->ptr         = malloc(size);
    }
    virtual ~BufferMalloc() {
        free(this->ptr);
    }
};

class BufferHost : public Buffer {
public:
    BufferHost(size_t size) {
        this->size        = size;
        this->device.type = Device::CPU;
        gpu_runtime::check(gpu_runtime::hostAlloc(&this->ptr, size, gpu_runtime::HostAllocPortable));
    }
    virtual ~BufferHost() {
        gpu_runtime::check(gpu_runtime::freeHost(this->ptr));
    }
};

class BufferGPU : public Buffer {
public:
    BufferGPU(size_t size) {
        this->size        = size;
        this->device      = Device::gpu();
        // this->device.idx = GpuDeviceContext::getDevice();
        this->device.idx = GpuDeviceContext::getDevice();
        if (size == 0) {
            this->ptr = nullptr;
        }
        // TODO: buffer used in multiple streams?
        gpu_runtime::check(gpu_runtime::mallocAsync(&this->ptr, size, getCurrentGpuStream()));
    }
    virtual ~BufferGPU() {
        if (this->size == 0) {
            assert(!this->ptr);
            return;
        }
        gpu_runtime::check(gpu_runtime::freeAsync(this->ptr, getCurrentGpuStream()));
    }
    virtual bool isAsyncBuffer() override {
        return true;
    }
};

class BufferGPUSync : public Buffer {
public:
    BufferGPUSync(size_t size) {
        this->size        = size;
        this->device      = Device::gpu();
        gpu_runtime::check(gpu_runtime::getDevice(&this->device.idx));
        gpu_runtime::check(gpu_runtime::mallocMemory(&this->ptr, size));
    }
    virtual ~BufferGPUSync() {
        gpu_runtime::check(gpu_runtime::freeMemory(this->ptr));
    }
};

class BufferView : public Buffer {
public:
    BufferView(std::shared_ptr<Buffer> reference, size_t offset, size_t size) : reference(reference) {
        assert(offset + size <= reference->getSize());
        this->ptr    = (void *)((std::uint8_t *)reference->getPtr() + offset);
        this->size   = size;
        this->device = reference->getDevice();
    }

private:
    std::shared_ptr<Buffer> reference;
};

struct TensorShape {
    std::vector<int> dataExtent;
    std::vector<int> dataStride;
    int64_t offset = 0;

    TensorShape() {}
    TensorShape(std::vector<int> shape) : dataExtent(std::move(shape)) {}
    TensorShape(std::initializer_list<int> dims) : dataExtent(dims) {}

    bool is_contiguous() const {
        if (dataStride.empty()) {
            return true;
        }
        if (size() == 0) {
            return true;
        }
        int64_t prod = 1;
        for (int i = dataExtent.size() - 1; i >= 0; i--) {
            if (dataExtent[i] > 1 && dataStride[i] != prod) {
                return false;
            }
            prod *= dataExtent[i];
        }
        return true;
    }
    int ndims() const {
        return dataExtent.size();
    }
    const int &operator[](int idx) const {
        if (idx < 0) {
            return dataExtent.at(dataExtent.size() + idx);
        } else {
            return dataExtent.at(idx);
        }
    }
    int &operator[](int idx) {
        return const_cast<int &>(const_cast<const TensorShape *>(this)->operator[](idx));
    }

    size_t stride(int idx) const {
        if (!dataStride.empty()) {
            if (idx < 0) {
                return dataStride.at(dataStride.size() + idx);
            } else {
                return dataStride.at(idx);
            }
        }

        if (idx < 0) {
            idx = dataExtent.size() + idx;
        }
        assert(idx >= 0 && (size_t)idx < dataExtent.size());
        size_t result = 1;
        for (size_t i = idx + 1; i < dataExtent.size(); i++) {
            assert(dataExtent[i] >= 0);
            result *= dataExtent[i];
        }
        return result;
    }

    size_t size() const {
        if (dataExtent.empty()) {
            return 0;
        }
        size_t result = 1;
        for (int dim : dataExtent) {
            assert(dim >= 0);
            result *= dim;
        }
        return result;
    }

    std::string str() const {
        if (dataExtent.empty()) {
            return "[]";
        }
        std::stringstream ss;
        ss << "[" << dataExtent[0];
        for (size_t i = 1; i < dataExtent.size(); i++) {
            ss << ", " << dataExtent[i];
        }
        ss << "]";
        return ss.str();
    }
};

class Tensor {
public:
    enum ScalarType {
        INVALID_SCALAR_TYPE,
        INT8,
        INT16,
        INT32,
        INT64,
        FP16,
        FP32,
        BF16,
        FP8_E4M3,
        FP8_E5M2,
    };

    struct TensorOptions {
        Device device_;
        ScalarType dtype_;

        Device device() const {
            return device_;
        }
        ScalarType dtype() const {
            return dtype_;
        }

        TensorOptions device(Device dev) const {
            TensorOptions result(*this);
            result.device_ = dev;
            return result;
        }
        TensorOptions dtype(ScalarType type) const {
            TensorOptions result(*this);
            result.dtype_ = type;
            return result;
        }
    };

    static const std::map<ScalarType, size_t> scalarSize;

public:
    TensorShape shape;
    ScalarType scalarType;
    std::shared_ptr<Buffer> buffer;

public:
    bool valid() const {
        return shape.dataExtent.size() > 0;
    }
    int size(int dim) const {
        return shape[dim];
    }
    bool is_contiguous() const {
        return shape.is_contiguous();
    }
    std::vector<int> sizes() const {
        return shape.dataExtent;
    }

    bool is_cuda() const {
        return device().type == Device::CUDA;
    }
    bool is_gpu() const {
        return device().is_gpu();
    }

    TensorOptions options() const {
        return TensorOptions{device(), dtype()};
    }
    int get_device() const {
        return device().idx;
    }

    template<typename T>
    T *data_ptr() {
        return reinterpret_cast<T *>(data_ptr());
    }
    template<typename T>
    const T *data_ptr() const {
        return reinterpret_cast<const T *>(data_ptr());
    }

    const void *data_ptr() const {
        return buffer->getPtr<char>() + shape.offset * scalar_size();
    }
    void *data_ptr() {
        return buffer->getPtr<char>() + shape.offset * scalar_size();
    }

    Device device() const {
        return buffer->getDevice();
    }

    ScalarType scalar_type() const {
        return scalarType;
    }
    ScalarType dtype() const {
        return scalar_type();
    }

    size_t stride(int dim) const {
        return shape.stride(dim);
    }

    size_t numel() const {
        return shape.size();
    }
    size_t ndims() const {
        return shape.ndims();
    }

    size_t dim() const {
        return ndims();
    }

    size_t scalar_size() const {
        return scalarSize.at(scalarType);
    }

    Tensor operator[](int idx) const {
        assert(ndims() > 1);
        Tensor result;
        result.shape      = std::vector<int>(this->shape.dataExtent.begin() + 1, this->shape.dataExtent.end());
        size_t size       = stride(0) * scalar_size();
        result.buffer     = std::make_shared<BufferView>(this->buffer, idx * size, size);
        result.scalarType = this->scalarType;
        return result;
    }

    template<typename T>
    const T &at(const std::vector<int> &idx) const {
        assert(ndims() == idx.size());
        int64_t offset = 0;
        for (size_t i = 0; i < ndims(); i++) {
            offset += idx.at(i) * stride(i);
        }
        assert(offset >= 0 && offset < numel());
        return this->data_ptr<T>()[offset];
    }

    template<typename T>
    T &at(const std::vector<int> &idx) {
        return const_cast<T &>(const_cast<const Tensor *>(this)->at<T>(idx));
    }

    Tensor slice(int dim, int from, int to) const {
        assert(from <= to);
        Tensor result;
        result.buffer     = this->buffer;
        result.scalarType = this->scalarType;

        result.shape      = TensorShape(this->shape.dataExtent);
        result.shape[dim] = to - from;
        result.shape.dataStride.resize(result.shape.ndims());
        for (int i = 0; i < result.shape.ndims(); i++) {
            result.shape.dataStride[i] = this->shape.stride(i);
        }
        result.shape.offset = this->shape.offset + this->shape.stride(dim) * from;

        return result;
    }
    Tensor transpose(int dim1, int dim2) const {
        Tensor result;
        result.buffer     = this->buffer;
        result.scalarType = this->scalarType;

        result.shape = TensorShape(this->shape.dataExtent);
        result.shape.dataStride.resize(result.shape.ndims());
        for (int i = 0; i < result.shape.ndims(); i++) {
            result.shape.dataStride[i] = this->shape.stride(i);
        }
        result.shape.offset = this->shape.offset;

        std::swap(result.shape.dataExtent[dim1], result.shape.dataExtent[dim2]);
        std::swap(result.shape.dataStride[dim1], result.shape.dataStride[dim2]);

        return result;
    }

    Tensor view(TensorShape shape) const {
        assert(shape.size() == this->shape.size());
        assert(this->is_contiguous());
        Tensor result;
        result.buffer       = this->buffer;
        result.scalarType   = this->scalarType;
        result.shape        = shape;
        result.shape.offset = this->shape.offset;
        return result;
    }
    Tensor reshape(TensorShape shape) const {
        return view(shape);
    }

    // // NOT IMPLEMENTED!!! DONT USE
    // Tensor transpose(int a, int b) const {
    //     throw std::runtime_error("Not implemented");
    // }

    Tensor &zero_() {
        assert(this->is_contiguous());
        gpu_runtime::check(gpu_runtime::memsetAsync(
            data_ptr<char>() + shape.offset * scalar_size(), 0, shape.size() * scalar_size(), getCurrentGpuStream()));
        return *this;
    }
    Tensor &copy_(Tensor other) {
        assert(this->is_contiguous());
        assert(other.is_contiguous());
        assert(this->shape.dataExtent == other.shape.dataExtent);
        assert(this->dtype() == other.dtype());

        assert((shape.offset + shape.size()) * scalar_size() <= buffer->getSize());
        assert((other.shape.offset + shape.size()) * scalar_size() <= other.buffer->getSize());

        if (shape.size() == 0) {
            return *this;
        }

        std::optional<GpuDeviceContext> operation_ctx_guard;

        if (this->device().is_gpu()) {
            operation_ctx_guard.emplace(this->device().idx);
        } else if (other.device().is_gpu()) {
            operation_ctx_guard.emplace(other.device().idx);
        }

        if (this->device().type == Device::CPU && other.device().type == Device::CPU) {
            memcpy(data_ptr<char>(), other.data_ptr<char>(), shape.size() * scalar_size());
            return *this;
        }

        lockBuffer(this->buffer, getCurrentGpuStream());
        lockBuffer(other.buffer, getCurrentGpuStream());
        gpu_runtime::check(gpu_runtime::memcpyAsync(data_ptr<char>(),
                                                    other.data_ptr<char>(),
                                                    shape.size() * scalar_size(),
                                                    getCopyKind(this->device(), other.device()),
                                                    getCurrentGpuStream()));
        return *this;
    }

    // NOT IMPLEMENTED!!! DONT USE
    template<typename T>
    Tensor &fill_(T val) {
        throw std::runtime_error("Not implemented");
        return *this;
    }
    // NOT IMPLEMENTED!!! DONT USE
    Tensor index(std::vector<std::any> whatever) {
        throw std::runtime_error("Not implemented");
    }

public:
    static Tensor allocate(TensorShape shape, ScalarType scalarType, Device device, bool fill = false) {
        Tensor result;
        assert(shape.is_contiguous());
        if (device.type == Device::CPU) {
            result.buffer = std::make_shared<BufferMalloc>(shape.size() * scalarSize.at(scalarType));
        } else if (device.is_gpu()) {
            // TODO: cross device allocate
            GpuDeviceContext ctx(device.idx);
            result.buffer = std::make_shared<BufferGPU>(shape.size() * scalarSize.at(scalarType));
        } else {
            assert(false);
        }
        result.scalarType = scalarType;
        result.shape      = shape;

        if (fill) {
            if (device.type == Device::CPU) {
                memset(result.buffer->getPtr(), 0xCC, result.buffer->getSize());
            } else if (device.is_gpu()) {
                GpuDeviceContext ctx(device.idx);
                gpu_runtime::check(gpu_runtime::memsetAsync(
                    result.buffer->getPtr(), 0xCC, result.buffer->getSize(), getCurrentGpuStream()));
            }
        }

        return result;
    }
    static Tensor empty(TensorShape shape, ScalarType scalarType, Device device) {
        return allocate(shape, scalarType, device);
    }
    static Tensor empty_like(const Tensor &tensor) {
        return empty(TensorShape(tensor.shape.dataExtent), tensor.scalarType, tensor.device());
    }
    static Tensor ones(TensorShape shape, ScalarType scalarType, Device device) {
        Tensor result = allocate(shape, scalarType, device);
        // FIXME FIXME FIXME
        gpu_runtime::check(gpu_runtime::memsetAsync(
            result.buffer->getPtr(), 1, result.buffer->getSize(), getCurrentGpuStream()));
        return result;
    }
    static Tensor
    allocate_view(TensorShape shape, ScalarType scalarType, std::shared_ptr<Buffer> buffer, size_t offset = 0) {
        Tensor result;
        result.buffer     = std::make_shared<BufferView>(buffer, offset, shape.size() * scalarSize.at(scalarType));
        result.scalarType = scalarType;
        result.shape      = shape;
        return result;
    }

public:
    Tensor copy(Device device) const {
        if (!buffer) {
            return *this;
        }
        Tensor result = allocate(this->shape.dataExtent, this->scalarType, device);
        result.copy_(*this);

        // lockBuffer(this->buffer, getCurrentGpuStream());
        // lockBuffer(result.buffer, getCurrentGpuStream());
        // checkCUDA(cudaMemcpyAsync(result.data_ptr(), this->data_ptr(), result.buffer->getSize(), cudaMemcpyDefault,
        // getCurrentGpuStream())); if (this->device().type == Device::CPU && device.type == Device::CUDA) {
        //     checkCUDA(cudaMemcpyAsync(result.data_ptr(), this->data_ptr(), result.buffer->getSize(),
        //     cudaMemcpyHostToDevice, getCurrentGpuStream()));
        // } else if (this->device().type == Device::CUDA && device.type == Device::CPU) {
        //     checkCUDA(cudaMemcpyAsync(result.data_ptr(), this->data_ptr(), result.buffer->getSize(),
        //     cudaMemcpyDeviceToHost, getCurrentGpuStream()));
        // } else {
        //     checkCUDA(cudaMemcpyAsync(result.data_ptr(), this->data_ptr(), result.buffer->getSize(),
        //     cudaMemcpyDefault, getCurrentGpuStream()));
        // }
        return result;
    }

    // void copy_range(Tensor &dst, int dim, int lower_bound, int upper_bound) {
    //     if (upper_bound > shape[dim]) {
    //         upper_bound = shape[dim];
    //     }
    //     if (lower_bound >= upper_bound) {
    //         return;
    //     }
    //     auto shapeOut = this->shape;
    //     shapeOut[dim] = upper_bound - lower_bound;
    //     assert(dst.shape.data == shapeOut.data);
    //     checkCUDA(cudaMemcpy2DAsync(
    //         dst.
    //     ));
    // }

private:
    static gpu_runtime::MemcpyKind getCopyKind(Device dst, Device src) {
        const bool srcGpu = src.is_gpu();
        const bool dstGpu = dst.is_gpu();
        if (!srcGpu && dstGpu) {
            return gpu_runtime::MemcpyHostToDevice;
        }
        if (srcGpu && !dstGpu) {
            return gpu_runtime::MemcpyDeviceToHost;
        }
        if (srcGpu && dstGpu) {
            return gpu_runtime::MemcpyDeviceToDevice;
        }
        if (!srcGpu && !dstGpu) {
            return gpu_runtime::MemcpyHostToHost;
        }
        return gpu_runtime::MemcpyDefault;
    }

    // static bool isAsyncBuffer(Buffer *buffer) {
    //     return dynamic_cast<BufferGPU *>(buffer);
    // }

    static inline std::map<gpu_runtime::Stream, std::set<std::shared_ptr<Buffer>>> lockedBuffers;

public:
    // before launching an async operation, make sure to lock the buffer in case the buffer is freed before GPU
    // completes
    static void lockBuffer(std::shared_ptr<Buffer> buffer, gpu_runtime::Stream stream) {
        if (!buffer->isAsyncBuffer()) {
            lockedBuffers[stream].insert(buffer);
        }
    }

    // we could unlock buffers after sync with GPU
    static void unlockBuffers() {
        lockedBuffers.clear();
    }
    static void unlockBuffers(gpu_runtime::Stream stream) {
        lockedBuffers[stream].clear();
    }

    static void synchronizeDevice() {
        gpu_runtime::check(gpu_runtime::deviceSynchronize());
        unlockBuffers();
    }
    static void synchronizeStream(gpu_runtime::Stream stream) {
        gpu_runtime::check(gpu_runtime::streamSynchronize(stream));
        unlockBuffers(stream);
    }
};

inline const std::map<Tensor::ScalarType, size_t> Tensor::scalarSize = {
    {INT8, 1},
    {INT16, 2},
    {INT32, 4},
    {INT64, 8},
    {FP16, 2},
    {FP32, 4},
    {BF16, 2},
    {FP8_E4M3, 1},
    {FP8_E5M2, 1},
};

struct TensorsProvider {
    virtual ~TensorsProvider() {}
    virtual bool contains(const std::string &key) const = 0;
    virtual Tensor getTensor(const std::string &key)    = 0;
};
