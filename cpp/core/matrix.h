#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cstddef>
#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <limits>

// ERROR enum outside class
enum Error {
    NO_ERROR = 0,
    INDEX_OUT_OF_RANGE,
    DIMENSION_MISMATCH,
    DIVIDE_BY_ZERO
};

class Matrix {
    size_t rows, cols;
    std::vector<float> data;

public:
    // Constructors
    Matrix();
    Matrix(size_t r, size_t c);

    // Accessors with optional error reporting
    float& operator()(size_t row, size_t col, Error* err = nullptr);
    const float& operator()(size_t row, size_t col, Error* err = nullptr) const;
    Matrix operator+(float val) const;
    size_t row_count() const;
    size_t col_count() const;
    size_t size() const;
    const float& no_bounds_check(size_t i) const;
    float& no_bounds_check(size_t i);
    void add_inplace(float val);
    // Matrix operations
    static Matrix random(size_t r, size_t c);
    float frobenius_norm() const;
    Matrix clip(float min_val, float max_val) const;
    Matrix add(const Matrix& other, Error* err = nullptr) const;
    Matrix subtract(const Matrix& other, Error* err = nullptr) const;
    void scale_inplace(float scalar);
    Matrix multiply_scalar(float scalar) const;
    Matrix divide_scalar(float scalar,Error* err) const;
    Matrix subtract_scalar(float scalar) const;
    static Matrix multiply(const Matrix& A, const Matrix& B, Error* err = nullptr);
    static Matrix divide(const Matrix& A, const Matrix& B, Error* err = nullptr);
    Matrix transpose() const;
    Matrix hadamard_product(const Matrix& A, Error* err = nullptr) const;
     Matrix hadamard_division(const Matrix& A, Error* err = nullptr) const;
    void multiply_scalar_inplace(float scalar);  
    // Activation functions
    Matrix RELU() const;
    Matrix deriv_RELU() const;
    Matrix leaky_RELU(float alpha = 0.01f) const;
    Matrix deriv_leaky_RELU(float alpha = 0.01f) const;
    static Matrix softmax(const Matrix& A);

    // Statistical operations
    float sum() const;
    float min() const;
    float max() const;
    float mean() const;
    Matrix& scale(float factor);
    static Matrix sum_cols(const Matrix& A);
    Matrix rowwise_mean() const;
    Matrix rowwise_std(float epsilon = 1e-8f) const;
    Matrix& subtract_rowwise(const Matrix& vec, Error* err = nullptr);

    // Slice columns safely (start..end inclusive)
    Matrix slice_cols(size_t start, size_t end,Error *err) const;
    Matrix sqrt() const;
    void sqrt_inplace();
    // Utility functions
    void fill(float value);
    void I(); // identity
    void print() const;
    
void add_inplace(const Matrix& other,float val);
void add_inplace_reg(float val);
void subtract_inplace(const Matrix& other,float val);
void subtract_inplace_element(const Matrix& other);
void hadamard_division_inplace(const Matrix& A, Error* err);
void add_inplace_squared(const Matrix& other ,float val);
void add_inplaceMat(const Matrix& other);


//CNN STUFF
    static Matrix Convolve2D(const Matrix& input, const Matrix& kernel, 
                            size_t stride = 1, Error* err = nullptr);
    
    // Max pooling
    static Matrix MaxPool2D(const Matrix& input, size_t pool_size, 
                           size_t stride, Error* err = nullptr);
    
    // Average pooling  
    static Matrix AveragePool2D(const Matrix& input, size_t pool_size,
                               size_t stride, Error* err = nullptr);

float Discrete_Convolution(const Matrix& I, const Matrix& K,size_t i, size_t j, Error* err= nullptr) const;
    // For handling multiple channels (simplified)
    static std::vector<Matrix> ConvolveMultiChannel(
        const std::vector<Matrix>& input_channels,
        const std::vector<Matrix>& kernels,
        size_t stride, Error* err = nullptr);


//  void rot180();

};
 Matrix rot180(const Matrix& mat);
#endif // MATRIX_H
