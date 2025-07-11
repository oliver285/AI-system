// // Example usage:
#include "matrix.h"
int main() {
    // Example usage
    // Matrix<2,2> mat;
    // mat.fill(5.0);
    // mat.print();

    // auto random_mat = Matrix<2, 2>::random();
    // random_mat.print();

// Static matrices
Matrix<2,3> m1;
m1=Matrix<2,3>::random();
// m1.print();
// m1.random();
auto m1_t = m1.transpose();  // 3x2 matrix

// m1_t.print();
// Dynamic matrices 
Matrix<2> m2(3); 
m2.print();  // 3x2 // 2x3
// m4.print();  // 2x2 result
auto m2_t = m2.transpose();
m2_t.print();  // 3x2
auto m4 = Matrix<2,3>::multiply(m1, m1_t);
// Operations
auto m3 = m1 + m1;  // element-wise
// auto m4 = Matrix<2,3>::multiply(m1, m1_t);  // 2x2 result
m3.print();
m4.print();
return 0;
}
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