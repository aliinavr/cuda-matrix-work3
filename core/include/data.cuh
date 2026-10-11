
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>


inline void cuda_check(cudaError_t error, const char* message)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(message) + ": " +
            cudaGetErrorString(error)
        );
    }
}


template <typename AtomT>
class Data
{
private:
    std::size_t size_;
    AtomT* data_;

public:
    explicit Data(std::size_t size)
        : size_(size),
          data_(nullptr)
    {
        if (size_ != 0)
        {
            cuda_check(
                cudaMalloc(
                    reinterpret_cast<void**>(&data_),
                    size_ * sizeof(AtomT)
                ),
                "cudaMalloc failed"
            );
        }
    }


    Data(const Data& other)
        : Data(other.size_)
    {
        if (size_ != 0)
        {
            cuda_check(
                cudaMemcpy(
                    data_,
                    other.data_,
                    size_ * sizeof(AtomT),
                    cudaMemcpyDeviceToDevice
                ),
                "cudaMemcpy DeviceToDevice failed"
            );
        }
    }


    Data(Data&& other) noexcept
        : size_(other.size_),
          data_(other.data_)
    {
        other.size_ = 0;
        other.data_ = nullptr;
    }


    Data& operator=(const Data& other)
    {
        if (this == &other)
        {
            return *this;
        }

        Data temp(other);

        std::swap(size_, temp.size_);
        std::swap(data_, temp.data_);

        return *this;
    }


    Data& operator=(Data&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        if (data_ != nullptr)
        {
            cudaFree(data_);
        }

        size_ = other.size_;
        data_ = other.data_;

        other.size_ = 0;
        other.data_ = nullptr;

        return *this;
    }


    AtomT* data()
    {
        return data_;
    }


    const AtomT* data() const
    {
        return data_;
    }


    std::size_t size() const
    {
        return size_;
    }


    void copy_to_host(AtomT* host_ptr) const
    {
        if (size_ != 0)
        {
            cuda_check(
                cudaMemcpy(
                    host_ptr,
                    data_,
                    size_ * sizeof(AtomT),
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpy DeviceToHost failed"
            );
        }
    }


    void copy_from_host(const AtomT* host_ptr)
    {
        if (size_ != 0)
        {
            cuda_check(
                cudaMemcpy(
                    data_,
                    host_ptr,
                    size_ * sizeof(AtomT),
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpy HostToDevice failed"
            );
        }
    }


    ~Data()
    {
        if (data_ != nullptr)
        {
            cudaFree(data_);
        }
    }
};
