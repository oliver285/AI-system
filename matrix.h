#include <iostream>
#include <vector>
#include <numbers>  // Requires C++20
#include <random>
#include <stdexcept>  // For runtime_error

template <int row, int col>
class Matrix {
public:
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

    // Constructor
    Matrix() : rows(row), cols(col), data(row, std::vector<double>(col, 0)) {}

    // Fill the matrix with a specific value
    void fill(double val) {
        for (uint16_t i = 0; i < rows; ++i)
            for (uint16_t j = 0; j < cols; ++j)
                data[i][j] = val;
    }

    // Create a matrix filled with random values from a normal distribution
    static Matrix<row, col> random() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 1.0);  // mean 0, std 1

        Matrix<row, col> mat;
        for (int i = 0; i < row; ++i)
            for (int j = 0; j < col; ++j)
                mat.data[i][j] = dist(gen);

        return mat;
    }

    // Add a value to each element of the matrix
    void add(int val) {
        for (uint16_t i = 0; i < rows; ++i)
            for (uint16_t j = 0; j < cols; ++j)
                data[i][j] += val;
    }

    // Apply the ReLU function (set negative values to 0)
    void RELU() {
        for (uint16_t i = 0; i < rows; ++i)
            for (uint16_t j = 0; j < cols; ++j)
                if (data[i][j] < 0)
                    data[i][j] = 0;
    }

    // Add two matrices
    static Matrix<row, col> Add_Matrix(const Matrix<row, col>& A, const Matrix<row, col>& B) {
        Matrix<row, col> added;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                added.data[i][j] = A.data[i][j] + B.data[i][j];
            }
        }
        return added;
    }

    // Subtract two matrices
    static Matrix<row, col> Sub_Matrix(const Matrix<row, col>& A, const Matrix<row, col>& B) {
        Matrix<row, col> subtracted;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                subtracted.data[i][j] = A.data[i][j] - B.data[i][j];
            }
        }
        return subtracted;
    }

    // Apply the exponential function to each element of the matrix
    static Matrix<row, col> exponential(const Matrix<row, col>& A) {
        Matrix<row, col> E;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                E.data[i][j] = std::exp(A.data[i][j]);
            }
        }
        return E;
    }

    // Compute the sum of all elements in the matrix
    double sum() const {
        double total = 0;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                total += data[i][j];
            }
        }
        return total;
    }

    // Softmax function for a matrix
    static Matrix<row, col> softmax(const Matrix<row, col>& A) {
        Matrix<row, col> max_value = max(A);
        Matrix<row, col> E_mat = exponential(A);
        double sum = E_mat.sum();
        Matrix<row, col> result;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                result.data[i][j] = E_mat.data[i][j] / sum;
            }
        }
        return result;
    }

    // Helper method to compute the max value in the matrix
    static Matrix<row, col> max(const Matrix<row, col>& A) {
        double max_val = A.data[0][0];
        Matrix<row, col> max_value;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                if (A.data[i][j] > max_val) {
                    max_val = A.data[i][j];
                }
            }
        }
        // Fill the result matrix with the max value
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                max_value.data[i][j] = max_val;
            }
        }
        return max_value;
    }

    // Print the matrix
    void print() const {
        for (const auto& r : data) {
            for (const auto& v : r) {
                std::cout << v << " ";
            }
            std::cout << "\n";
        }
    }

    // Matrix multiplication
    template <int shared, int col2>
    static Matrix<row, col2> multiply(const Matrix<row, shared>& A, const Matrix<shared, col2>& B) {
        Matrix<row, col2> result;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col2; ++j) {
                for (int k = 0; k < shared; ++k) {
                    result.data[i][j] += A.data[i][k] * B.data[k][j];
                }
            }
        }
        return result;
    }


// Inverse matrix using Gauss-Jordan elimination


Matrix<row, col> invert(const Matrix<row, col>& input) {
    Matrix<row, col> A = input;
    Matrix<row, col> I;

    // Initialize identity matrix
    for (int i = 0; i < row; ++i) {
        I.data[i][i] = 1.0;
    }

    // Gauss-Jordan elimination
    for (int i = 0; i < row; ++i) {
        if (A.data[i][i] == 0) {
            for (uint16_t j = i + 1; j < row; ++j) {
                if (A.data[j][i] != 0) {
                    std::swap(A.data[i], A.data[j]);
                    std::swap(I.data[i], I.data[j]);
                    break;
                }
            }
        }

        double pivot = A.data[i][i];
        if (pivot == 0) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }

        for (uint16_t j = 0; j < col; ++j) {
            A.data[i][j] /= pivot;
            I.data[i][j] /= pivot;
        }

        for (uint16_t k = 0; k < row; ++k) {
            if (k == i) continue;
            double factor = A.data[k][i];
            for (uint16_t j = 0; j < col; ++j) {
                A.data[k][j] -= factor * A.data[i][j];
                I.data[k][j] -= factor * I.data[i][j];
            }
        }
    }

    return I;
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




