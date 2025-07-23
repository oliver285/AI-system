#include "matrix.h"
#include <iostream>

// int main() {
//     // Create and print random matrix
//     Matrix mat3 = Matrix::random(3, 3);
//     std::cout << "Original matrix:\n";
//     mat3.print();
    
//     // Test transpose
//     Matrix transposed = mat3.transpose();
//     std::cout << "\nTransposed matrix:\n";
//     transposed.print();
    
//     // Verify multiplication
//     Matrix mat(3, 3);
//     mat.fill(2);
//     Matrix mat2 = Matrix::multiply(mat, mat3);
//     std::cout << "\nMultiplication result:\n";
//     mat2.print();
//     Matrix mat4(3,3);
//     mat4.I();
//     mat2 = Matrix::multiply(mat4,mat2);
//     std::cout<<"\n Eye Matrix Result:\n";
//     mat2.print();

//     return 0;
// }
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