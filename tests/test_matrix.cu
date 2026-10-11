
#include "matrix.cuh"

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <tuple>
#include <vector>


using TestParams =
    std::tuple<int, int, int>;


class MatrixMultiplicationTest
    : public ::testing::TestWithParam<TestParams>
{
};


TEST_P(
    MatrixMultiplicationTest,
    SharedMemoryMatchesEigen
)
{
    const auto [m, k, n] =
        GetParam();


    std::vector<float> hostA(
        static_cast<std::size_t>(m) * k
    );

    std::vector<float> hostB(
        static_cast<std::size_t>(k) * n
    );


    for (std::size_t i = 0;
         i < hostA.size();
         ++i)
    {
        const int value =
            static_cast<int>(i % 7) - 3;

        hostA[i] =
            static_cast<float>(value)
            * 0.001f;
    }


    for (std::size_t i = 0;
         i < hostB.size();
         ++i)
    {
        const int value =
            static_cast<int>(i % 5) - 2;

        hostB[i] =
            static_cast<float>(value)
            * 0.001f;
    }


    Matrix<float> A(m, k);
    Matrix<float> B(k, n);


    A.data().copy_from_host(
        hostA.data()
    );

    B.data().copy_from_host(
        hostB.data()
    );


    Matrix<float> C =
        A * B;


    std::vector<float> hostC(
        static_cast<std::size_t>(m) * n
    );


    C.data().copy_to_host(
        hostC.data()
    );


    using RowMajorMatrix =
        Eigen::Matrix<
            float,
            Eigen::Dynamic,
            Eigen::Dynamic,
            Eigen::RowMajor
        >;


    Eigen::Map<RowMajorMatrix>
        eigenA(
            hostA.data(),
            m,
            k
        );


    Eigen::Map<RowMajorMatrix>
        eigenB(
            hostB.data(),
            k,
            n
        );


    Eigen::Map<RowMajorMatrix>
        actualRowMajor(
            hostC.data(),
            m,
            n
        );


    Eigen::MatrixXf expected =
        eigenA * eigenB;


    Eigen::MatrixXf actual =
        actualRowMajor;


    ASSERT_TRUE(
        expected.isApprox(
            actual,
            1e-5f
        )
    )
        << "m = " << m
        << ", k = " << k
        << ", n = " << n;
}


INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    MatrixMultiplicationTest,

    ::testing::Combine(

        ::testing::Values(
            1,
            2,
            3,
            127,
            128,
            129,
            512
        ),

        ::testing::Values(
            1,
            2,
            3,
            127,
            128,
            129,
            512
        ),

        ::testing::Values(
            1,
            2,
            3,
            127,
            128,
            129,
            512
        )
    )
);
