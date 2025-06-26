#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

template <int rows, int cols>
class Matrix {
public:
    std::vector<std::vector<double>> data;

    // Constructor
    Matrix() : data(rows, std::vector<double>(cols, 0)) {}

    // Random initialization with scaling
    static Matrix<rows, cols> random() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 1.0);

        Matrix<rows, cols> mat;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                mat.data[i][j] = dist(gen);
        return mat;
    }

    // Element-wise operations
    Matrix<rows, cols> operator+(const Matrix<rows, cols>& other) const {
        Matrix<rows, cols> result;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    Matrix<rows, cols> operator-(const Matrix<rows, cols>& other) const {
        Matrix<rows, cols> result;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] - other.data[i][j];
        return result;
    }

    // Scalar operations
    Matrix<rows, cols> multiply_scalar(double scalar) const {
        Matrix<rows, cols> result;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] * scalar;
        return result;
    }

    Matrix<rows, cols> subtract_scalar(double scalar) const {
        Matrix<rows, cols> result;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] - scalar;
        return result;
    }

    // Matrix multiplication
    template <int other_cols>
    static Matrix<rows, other_cols> multiply(const Matrix<rows, cols>& A, const Matrix<cols, other_cols>& B) {
        Matrix<rows, other_cols> result;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other_cols; ++j) {
                for (int k = 0; k < cols; ++k) {
                    result.data[i][j] += A.data[i][k] * B.data[k][j];
                }
            }
        }
        return result;
    }

    // Transpose
    Matrix<cols, rows> transpose() const {
        Matrix<cols, rows> result;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[j][i] = data[i][j];
        return result;
    }

    // Activation functions
    Matrix<rows, cols> RELU() const {
        Matrix<rows, cols> result;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = std::max(0.0, data[i][j]);
        return result;
    }

    Matrix<rows, cols> deriv_RELU() const {
        Matrix<rows, cols> result;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = (data[i][j] > 0) ? 1 : 0;
        return result;
    }

    // Softmax
    static Matrix<rows, cols> softmax(const Matrix<rows, cols>& A) {
        Matrix<rows, cols> result;
        // Find max for numerical stability
        double max_val = A.data[0][0];
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                if (A.data[i][j] > max_val) max_val = A.data[i][j];
        
        // Compute exponentials
        double sum = 0;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.data[i][j] = std::exp(A.data[i][j] - max_val);
                sum += result.data[i][j];
            }
        }
        
        // Normalize
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] /= sum;
                
        return result;
    }

    // Sum of all elements
    double sum() const {
        double total = 0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                total += data[i][j];
        return total;
    }

    // Column-wise sum (for batch processing)
    static Matrix<rows, 1> sum_cols(const Matrix<rows, cols>& A) {
        Matrix<rows, 1> result;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][0] += A.data[i][j];
        return result;
    }

    // Fill with value
    void fill(double value) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[i][j] = value;
    }

    // Print matrix
    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << "\n";
        }
    }
};







// int main() {
//     // Example usage
//     Matrix<2, 2> mat;
//     mat.fill(5.0);
//     mat.print();

//     auto random_mat = Matrix<2, 2>::random();
//     random_mat.print();

//     auto identity = invert(mat);
//     identity.print();

//     return 0;
// }




