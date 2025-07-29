     #include"../core/matrix.h"
    
    
    
    
    
     void test_relu() {
        std::cout << "Testing RELU...\n";
        Matrix m(2, 2);
        m(0,0) = -1.0; m(0,1) = 0.0;
        m(1,0) = 2.0;  m(1,1) = -0.5;
        
        Matrix result = m.RELU();
        assert(result(0,0) == 0.0);
        assert(result(0,1) == 0.0);
        assert(result(1,0) == 2.0);
        assert(result(1,1) == 0.0);
        std::cout << "RELU passed!\n\n";
    }
    
    void test_leaky_relu() {
        std::cout << "Testing Leaky RELU...\n";
        Matrix m(1, 3);
        m(0,0) = -2.0; m(0,1) = 0.0; m(0,2) = 1.0;
        
        Matrix result = m.leaky_RELU(0.1);
        assert(fabs(result(0,0) - (-0.2) < 1e-6));
        assert(result(0,1) == 0.0);
        assert(result(0,2) == 1.0);
        std::cout << "Leaky RELU passed!\n\n";
    }
    
    void test_relu_derivative() {
        std::cout << "Testing RELU Derivative...\n";
        Matrix m(1, 3);
        m(0,0) = -1.0; m(0,1) = 0.0; m(0,2) = 1.0;
        
        Matrix result = m.deriv_RELU();
        assert(result(0,0) == 0.0);
        assert(result(0,1) == 1.0);  // Derivative at 0 is defined as 1
        assert(result(0,2) == 1.0);
        std::cout << "RELU Derivative passed!\n\n";
    }
    
    void test_leaky_relu_derivative() {
        std::cout << "Testing Leaky RELU Derivative...\n";
        Matrix m(1, 3);
        m(0,0) = -1.0; m(0,1) = 0.0; m(0,2) = 1.0;
        
        Matrix result = m.deriv_leaky_RELU(0.1);
        assert(fabs(result(0,0) - 0.1 < 1e-6));
        assert(result(0,1) == 1.0);  // Derivative at 0 is defined as 1
        assert(result(0,2) == 1.0);
        std::cout << "Leaky RELU Derivative passed!\n\n";
    }
    
    void test_softmax() {
        std::cout << "Testing Softmax...\n";
        Matrix m(3, 2);
        // Column 1: [1.0, 2.0, 3.0]
        // Column 2: [-1.0, 0.0, 1.0]
        m(0,0) = 1.0; m(0,1) = -1.0;
        m(1,0) = 2.0; m(1,1) = 0.0;
        m(2,0) = 3.0; m(2,1) = 1.0;
        
        Matrix result = Matrix::softmax(m);
        
        // Test column 1
        double sum1 = result(0,0) + result(1,0) + result(2,0);
        assert(fabs(sum1 - 1.0) < 1e-6);
        assert(result(2,0) > result(1,0));
        assert(result(1,0) > result(0,0));
        
        // Test column 2
        double sum2 = result(0,1) + result(1,1) + result(2,1);
        assert(fabs(sum2 - 1.0) < 1e-6);
        assert(result(2,1) > result(1,1));
        assert(result(1,1) > result(0,1));
        
        std::cout << "Softmax passed!\n\n";
    }
    
    void test_matrix_multiplication() {
        std::cout << "Testing Matrix Multiplication...\n";
        Matrix A(2, 3);
        A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
        A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
        
        Matrix B(3, 2);
        B(0,0) = 7; B(0,1) = 8;
        B(1,0) = 9; B(1,1) = 10;
        B(2,0) = 11; B(2,1) = 12;
        
        Matrix C = Matrix::multiply(A, B);
        assert(C(0,0) == 58);
        assert(C(0,1) == 64);
        assert(C(1,0) == 139);
        assert(C(1,1) == 154);
        std::cout << "Matrix Multiplication passed!\n\n";
    }
    
    void test_transpose() {
        std::cout << "Testing Transpose...\n";
        Matrix m(2, 3);
        m(0,0) = 1; m(0,1) = 2; m(0,2) = 3;
        m(1,0) = 4; m(1,1) = 5; m(1,2) = 6;
        
        Matrix t = m.transpose();
        assert(t(0,0) == 1);
        assert(t(0,1) == 4);
        assert(t(1,0) == 2);
        assert(t(1,1) == 5);
        assert(t(2,0) == 3);
        assert(t(2,1) == 6);
        std::cout << "Transpose passed!\n\n";
    }
    
    
    
    // int main() {
       
    //     test_relu();
    //     test_leaky_relu();
    //     test_relu_derivative();
    //     test_leaky_relu_derivative();
    //     test_softmax();
    //     test_matrix_multiplication();
    //     test_transpose();
        
    //     std::cout << "All tests passed!\n";
    //     return 0;
    // }