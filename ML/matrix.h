#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cstddef> // For size_t
#include <iostream>
// #include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
class Matrix {
    size_t rows, cols;
    std::vector<double> data;
    
public:
    // Constructors
    Matrix();
    Matrix(size_t r, size_t c);

    // Accessors
    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;
    size_t row_count() const;
    size_t col_count() const;
    size_t size() const;
    const double& no_bounds_check(size_t i) const;
    double& no_bounds_check(size_t i);

    // Matrix operations
    static Matrix random(size_t r, size_t c);
    double frobenius_norm() const;
    Matrix clip(double min_val, double max_val) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    void scale_inplace(double scalar);
    Matrix multiply_scalar(double scalar) const;
    Matrix subtract_scalar(double scalar) const;
    static Matrix multiply(const Matrix& A, const Matrix& B);
    Matrix transpose() const;
    Matrix hadamard_product(const Matrix& A) const;

    // Activation functions
    Matrix RELU() const;
    Matrix deriv_RELU() const;
    Matrix leaky_RELU(double alpha = 0.01) const;
    Matrix deriv_leaky_RELU(double alpha = 0.01) const;
    static Matrix softmax(const Matrix& A);

    // Statistical operations
    double sum() const;
    double min() const;
    double max() const;
    double mean() const;
    Matrix& scale(double factor);
    static Matrix sum_cols(const Matrix& A);

    // Utility functions
    void fill(double value);
    void I();
    void print() const;
};

#endif // MATRIX_H