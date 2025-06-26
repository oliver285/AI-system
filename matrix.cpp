// // Example usage:
#include "matrix.h"
int main() {
    // Example usage
    Matrix<2,2> mat;
    mat.fill(5.0);
    mat.print();

    auto random_mat = Matrix<2, 2>::random();
    random_mat.print();

    Matrix<2,2> identity;
    identity=random_mat.invert(random_mat);
    identity.print();

    return 0;
}
// //     Matrix<2, 2> A;
// //     A.fill(2);
// //     std::cout << "Matrix A:\n";
// //     A.print();

// 
//     std::cout << "\nMatrix E (filled with e):\n";
//     E.print();

//     // auto Product = Matrix<2, 2>::multiply(A, E);
//     std::cout << "\nA * E:\n";
//     // Product.print();

//     Matrix<3, 3> A;
// A.data = {
//     {2, 1, 1},
//     {1, 3, 2},
//     {1, 0, 0}
// };

// std::cout << "Original Matrix A:\n";
// A.print();

// Matrix<3, 3> A_inv = invert(A);
// std::cout << "\nInverse of A:\n";
// A_inv.print();

//     return 0;
// }