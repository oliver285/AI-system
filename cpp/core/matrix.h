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
// ERROR enum OUTSIDE class
enum Error {
    NO_ERROR = 0,
    INDEX_OUT_OF_RANGE,
    DIMENSION_MISMATCH,
};

class Matrix {
    size_t rows, cols;
    std::vector<double> data;
    
public:
    // Constructors
    Matrix();
    Matrix(size_t r, size_t c);


// Modify accessors to return error status
double& operator()(size_t row, size_t col, Error* err = nullptr);
const double& operator()(size_t row, size_t col, Error* err = nullptr) const;
    size_t row_count() const;
    size_t col_count() const;
    size_t size() const;
    const double& no_bounds_check(size_t i) const;
    double& no_bounds_check(size_t i);

    // Matrix operations
    static Matrix random(size_t r, size_t c);
    double frobenius_norm() const;
    Matrix clip(double min_val, double max_val) const;
    Matrix add(const Matrix& other,Error* err= nullptr) const;
    Matrix subtract(const Matrix& other,Error* err= nullptr) const;
    void scale_inplace(double scalar);
    Matrix multiply_scalar(double scalar) const;
    Matrix subtract_scalar(double scalar) const;
    static Matrix multiply(const Matrix& A, const Matrix& B,Error* err= nullptr);
    Matrix transpose() const;
    Matrix hadamard_product(const Matrix& A, Error* err = nullptr) const;

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
   Matrix rowwise_mean() const;
  Matrix rowwise_std(double epsilon = 1e-8) const;
  Matrix& subtract_rowwise(const Matrix& vec,Error* err = nullptr);

    // Utility functions
    void fill(double value);
    void I();
    void print() const;
};

#endif // MATRIX_H