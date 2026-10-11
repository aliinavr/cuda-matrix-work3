
#include "matrix.cuh"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>


static void fill_data(
    std::vector<float>& data
)
{
    for (std::size_t i = 0;
         i < data.size();
         ++i)
    {
        data[i] =
            static_cast<float>(
                (i * 17) % 101
            ) / 101.0f;
    }
}


static void BM_Naive(
    benchmark::State& state
)
{
    const int n =
        static_cast<int>(
            state.range(0)
        );


    std::vector<float> hostA(
        static_cast<std::size_t>(n) * n
    );

    std::vector<float> hostB(
        static_cast<std::size_t>(n) * n
    );


    fill_data(hostA);
    fill_data(hostB);


    Matrix<float> A(n, n);
    Matrix<float> B(n, n);
    Matrix<float> C(n, n);


    A.data().copy_from_host(
        hostA.data()
    );

    B.data().copy_from_host(
        hostB.data()
    );


    A.multiply_to(
        B,
        C,
        NaiveMatMulStrategy{}
    );


    cuda_check(
        cudaDeviceSynchronize(),
        "Naive warm-up failed"
    );


    cudaEvent_t start;
    cudaEvent_t stop;


    cuda_check(
        cudaEventCreate(&start),
        "cudaEventCreate failed"
    );

    cuda_check(
        cudaEventCreate(&stop),
        "cudaEventCreate failed"
    );


    for (auto _ : state)
    {
        cuda_check(
            cudaEventRecord(start),
            "cudaEventRecord failed"
        );


        A.multiply_to(
            B,
            C,
            NaiveMatMulStrategy{}
        );


        cuda_check(
            cudaEventRecord(stop),
            "cudaEventRecord failed"
        );


        cuda_check(
            cudaEventSynchronize(stop),
            "cudaEventSynchronize failed"
        );


        float ms = 0.0f;


        cuda_check(
            cudaEventElapsedTime(
                &ms,
                start,
                stop
            ),
            "cudaEventElapsedTime failed"
        );


        state.SetIterationTime(
            ms / 1000.0
        );
    }


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


static void BM_Shared(
    benchmark::State& state
)
{
    const int n =
        static_cast<int>(
            state.range(0)
        );


    std::vector<float> hostA(
        static_cast<std::size_t>(n) * n
    );

    std::vector<float> hostB(
        static_cast<std::size_t>(n) * n
    );


    fill_data(hostA);
    fill_data(hostB);


    Matrix<float> A(n, n);
    Matrix<float> B(n, n);
    Matrix<float> C(n, n);


    A.data().copy_from_host(
        hostA.data()
    );

    B.data().copy_from_host(
        hostB.data()
    );


    A.multiply_to(
        B,
        C,
        SharedMatMulStrategy{}
    );


    cuda_check(
        cudaDeviceSynchronize(),
        "Shared warm-up failed"
    );


    cudaEvent_t start;
    cudaEvent_t stop;


    cuda_check(
        cudaEventCreate(&start),
        "cudaEventCreate failed"
    );

    cuda_check(
        cudaEventCreate(&stop),
        "cudaEventCreate failed"
    );


    for (auto _ : state)
    {
        cuda_check(
            cudaEventRecord(start),
            "cudaEventRecord failed"
        );


        A.multiply_to(
            B,
            C,
            SharedMatMulStrategy{}
        );


        cuda_check(
            cudaEventRecord(stop),
            "cudaEventRecord failed"
        );


        cuda_check(
            cudaEventSynchronize(stop),
            "cudaEventSynchronize failed"
        );


        float ms = 0.0f;


        cuda_check(
            cudaEventElapsedTime(
                &ms,
                start,
                stop
            ),
            "cudaEventElapsedTime failed"
        );


        state.SetIterationTime(
            ms / 1000.0
        );
    }


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


BENCHMARK(BM_Naive)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);


BENCHMARK(BM_Shared)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();
