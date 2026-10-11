
#include "matrix.cuh"

#include <iostream>
#include <vector>


int main()
{
    Matrix<float> A(2, 3);
    Matrix<float> B(3, 2);


    std::vector<float> hostA =
    {
        1, 2, 3,
        4, 5, 6
    };


    std::vector<float> hostB =
    {
        7, 8,
        9, 10,
        11, 12
    };


    A.data().copy_from_host(
        hostA.data()
    );

    B.data().copy_from_host(
        hostB.data()
    );


    Matrix<float> C = A * B;


    std::vector<float> hostC(
        C.size()
    );


    C.data().copy_to_host(
        hostC.data()
    );


    for (std::size_t i = 0;
         i < C.nrows();
         ++i)
    {
        for (std::size_t j = 0;
             j < C.ncols();
             ++j)
        {
            std::cout
                << hostC[
                    i * C.ncols() + j
                ]
                << " ";
        }

        std::cout << '\n';
    }


    return 0;
}
