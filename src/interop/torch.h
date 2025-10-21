#pragma once

#include <torch/extension.h>

#include "common.h"
#include "Tensor.h"

class BufferTorchTensor : public Buffer {
public:
    BufferTorchTensor(at::Tensor tensor) : tensor(std::move(tensor)) {
        this->size        = this->tensor.numel() * this->tensor.itemsize();
        this->ptr         = this->tensor.data_ptr();
        const int deviceIdx = this->tensor.device().has_index() ? this->tensor.device().index() : 0;
        if (this->tensor.device().is_cuda()) {
            this->device = Device::cuda(deviceIdx);
#if defined(NUNCHAKU_USE_HIP)
        } else if (this->tensor.device().type() == c10::DeviceType::HIP) {
            this->device = Device::rocm(deviceIdx);
#endif
        } else {
            this->device = Device::cpu();
        }
        this->device.idx = deviceIdx;
    }
    virtual bool isAsyncBuffer() override {
        // TODO: figure out how torch manages memory
        return this->device.is_gpu();
    }

private:
    at::Tensor tensor;
};

class TorchOpContext {
public:
    TorchOpContext();
    TorchOpContext(const TorchOpContext &) = delete;
    TorchOpContext(TorchOpContext &&)      = delete;
    ~TorchOpContext();
};

Tensor from_torch(at::Tensor input);
at::Tensor to_torch(Tensor input);

class TensorsProviderTorch : public TensorsProvider {
public:
    TensorsProviderTorch(std::map<std::string, at::Tensor> dict) : storage(std::move(dict)) {}

    virtual bool contains(const std::string &key) const override {
        return storage.contains(key);
    }
    virtual Tensor getTensor(const std::string &key) override {
        if (!storage.contains(key)) {
            return Tensor{};
        }
        return from_torch(storage.at(key));
    }

private:
    std::map<std::string, at::Tensor> storage;
};
