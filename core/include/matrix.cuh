
#pragma once

#include "data.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>


constexpr int BLOCK_SIZE = 16;


template <typename AtomT>
class MatrixView
{
private:
    AtomT* data_;
    std::size_t nrows_;
    std::size_t ncols_;

public:
    __host__ __device__
    MatrixView(
        AtomT* data = nullptr,
        std::size_t nrows = 0,
        std::size_t ncols = 0
    )
        : data_(data),
          nrows_(nrows),
          ncols_(ncols)
    {
    }


    __host__ __device__
    std::size_t size() const
    {
        return nrows_ * ncols_;
    }


    __host__ __device__
    std::size_t nrows() const
    {
        return nrows_;
    }


    __host__ __device__
    std::size_t ncols() const
    {
        return ncols_;
    }


    __host__ __device__
    AtomT& operator[](std::size_t n)
    {
        return data_[n];
    }


    __host__ __device__
    const AtomT& operator[](std::size_t n) const
    {
        return data_[n];
    }


    __host__ __device__
    AtomT& operator()(
        std::size_t i,
        std::size_t j
    )
    {
        return data_[i * ncols_ + j];
    }


    __host__ __device__
    const AtomT& operator()(
        std::size_t i,
        std::size_t j
    ) const
    {
        return data_[i * ncols_ + j];
    }
};


static_assert(
    std::is_trivially_copyable_v<MatrixView<float>>,
    "MatrixView must be trivially copyable"
);


template <typename AtomT>
__global__
void kernel_matmul_naive(
    MatrixView<AtomT> A,
    MatrixView<AtomT> B,
    MatrixView<AtomT> C
)
{
    const std::size_t row =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    const std::size_t col =
        blockIdx.x * blockDim.x +
        threadIdx.x;


    if (row >= C.nrows() ||
        col >= C.ncols())
    {
        return;
    }


    AtomT sum = AtomT{};


    for (std::size_t k = 0;
         k < A.ncols();
         ++k)
    {
        sum +=
            A(row, k) *
            B(k, col);
    }


    C(row, col) = sum;
}


template <typename AtomT>
__global__
void kernel_matmul_shmem(
    MatrixView<AtomT> A,
    MatrixView<AtomT> B,
    MatrixView<AtomT> C
)
{
    __shared__
    AtomT tileA[BLOCK_SIZE][BLOCK_SIZE];

    __shared__
    AtomT tileB[BLOCK_SIZE][BLOCK_SIZE];


    const std::size_t row =
        blockIdx.y * BLOCK_SIZE +
        threadIdx.y;

    const std::size_t col =
        blockIdx.x * BLOCK_SIZE +
        threadIdx.x;


    AtomT sum = AtomT{};


    const std::size_t tile_count =
        (
            A.ncols() +
            BLOCK_SIZE - 1
        ) / BLOCK_SIZE;


    for (std::size_t tile = 0;
         tile < tile_count;
         ++tile)
    {
        const std::size_t a_col =
            tile * BLOCK_SIZE +
            threadIdx.x;

        const std::size_t b_row =
            tile * BLOCK_SIZE +
            threadIdx.y;


        if (row < A.nrows() &&
            a_col < A.ncols())
        {
            tileA
                [threadIdx.y]
                [threadIdx.x]
                =
                A(row, a_col);
        }
        else
        {
            tileA
                [threadIdx.y]
                [threadIdx.x]
                =
                AtomT{};
        }


        if (b_row < B.nrows() &&
            col < B.ncols())
        {
            tileB
                [threadIdx.y]
                [threadIdx.x]
                =
                B(b_row, col);
        }
        else
        {
            tileB
                [threadIdx.y]
                [threadIdx.x]
                =
                AtomT{};
        }


        __syncthreads();


        #pragma unroll
        for (int k = 0;
             k < BLOCK_SIZE;
             ++k)
        {
            sum +=
                tileA
                    [threadIdx.y]
                    [k]
                *
                tileB
                    [k]
                    [threadIdx.x];
        }


        __syncthreads();
    }


    if (row < C.nrows() &&
        col < C.ncols())
    {
        C(row, col) = sum;
    }
}


struct NaiveMatMulStrategy
{
    template <typename AtomT>
    void operator()(
        MatrixView<AtomT> A,
        MatrixView<AtomT> B,
        MatrixView<AtomT> C
    ) const
    {
        dim3 block(
            BLOCK_SIZE,
            BLOCK_SIZE
        );


        dim3 grid(
            static_cast<unsigned int>(
                (
                    C.ncols() +
                    BLOCK_SIZE - 1
                )
                / BLOCK_SIZE
            ),

            static_cast<unsigned int>(
                (
                    C.nrows() +
                    BLOCK_SIZE - 1
                )
                / BLOCK_SIZE
            )
        );


        kernel_matmul_naive
            <<<grid, block>>>(
                A,
                B,
                C
            );


        cuda_check(
            cudaGetLastError(),
            "kernel_matmul_naive launch failed"
        );
    }
};


struct SharedMatMulStrategy
{
    template <typename AtomT>
    void operator()(
        MatrixView<AtomT> A,
        MatrixView<AtomT> B,
        MatrixView<AtomT> C
    ) const
    {
        dim3 block(
            BLOCK_SIZE,
            BLOCK_SIZE
        );


        dim3 grid(
            static_cast<unsigned int>(
                (
                    C.ncols() +
                    BLOCK_SIZE - 1
                )
                / BLOCK_SIZE
            ),

            static_cast<unsigned int>(
                (
                    C.nrows() +
                    BLOCK_SIZE - 1
                )
                / BLOCK_SIZE
            )
        );


        kernel_matmul_shmem
            <<<grid, block>>>(
                A,
                B,
                C
            );


        cuda_check(
            cudaGetLastError(),
            "kernel_matmul_shmem launch failed"
        );
    }
};


template <typename AtomT>
class Matrix
{
private:
    std::shared_ptr<Data<AtomT>> data_;
    MatrixView<AtomT> view_;

public:
    Matrix(
        std::size_t nrows,
        std::size_t ncols
    )
        : data_(
            std::make_shared<Data<AtomT>>(
                nrows * ncols
            )
          ),
          view_(
            data_->data(),
            nrows,
            ncols
          )
    {
    }


    std::size_t size() const
    {
        return view_.size();
    }


    std::size_t nrows() const
    {
        return view_.nrows();
    }


    std::size_t ncols() const
    {
        return view_.ncols();
    }


    Data<AtomT>& data()
    {
        return *data_;
    }


    const Data<AtomT>& data() const
    {
        return *data_;
    }


    MatrixView<AtomT>& view()
    {
        return view_;
    }


    const MatrixView<AtomT>& view() const
    {
        return view_;
    }


    template <typename Strategy>
    void multiply_to(
        const Matrix& other,
        Matrix& result,
        Strategy strategy
    ) const
    {
        if (ncols() != other.nrows())
        {
            throw std::invalid_argument(
                "Matrix sizes are incompatible"
            );
        }


        if (result.nrows() != nrows() ||
            result.ncols() != other.ncols())
        {
            throw std::invalid_argument(
                "Invalid result matrix size"
            );
        }


        strategy(
            view_,
            other.view_,
            result.view_
        );
    }


    Matrix operator*(
        const Matrix& other
    ) const
    {
        if (ncols() != other.nrows())
        {
            throw std::invalid_argument(
                "Matrix sizes are incompatible"
            );
        }


        Matrix result(
            nrows(),
            other.ncols()
        );


        multiply_to(
            other,
            result,
            SharedMatMulStrategy{}
        );


        return result;
    }
};
