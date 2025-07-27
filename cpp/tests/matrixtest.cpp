     #include"../include/matrix.h"
    
    
    
    
    
    
    
    
    
    int main() {
        test_relu();
        test_leaky_relu();
        test_relu_derivative();
        test_leaky_relu_derivative();
        test_softmax();
        test_matrix_multiplication();
        test_transpose();
        
        std::cout << "All tests passed!\n";
        return 0;
    }