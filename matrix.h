#include <iostream>
#include <vector>
#include <numbers>  // Requires C++20

template <uint16_t row, uint16_t col>
class Matrix {
public:
uint16_t rows;
uint16_t cols;
    std::vector<std::vector<double>> data;

    Matrix() : rows(row), cols(col), data(row, std::vector<double>(col, 0)) {}

    void fill(double val) {
        for (uint16_t i = 0; i < rows; ++i)
            for (uint16_t j = 0; j < cols; ++j)
                data[i][j] = val;
    }
   void Matrix<row, col> random() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 1.0);  // mean 0, std 1

        Matrix<row, col> mat;
        for (int i = 0; i < row; ++i)
            for (int j = 0; j < col; ++j)
            data[i][j] = dist(gen);

        return mat;
    }

    void add(int val){

        for (uint16_t i = 0; i < rows; ++i)
        for (uint16_t j = 0; j < cols; ++j){
    data[i][j]+=val;
    
        }
    }

    void subtract(int val){

        for (uint16_t i = 0; i < rows; ++i)
        for (uint16_t j = 0; j < cols; ++j){
    data[i][j]-=val;
    
        }
    }

    void print() const {
        for (const auto& r : data) {
            for (const auto& v : r)
                std::cout << v << " ";
            std::cout << "\n";
        }
    }

    static Matrix<row, col> expMatrix() {
        Matrix<row, col> mat;
        mat.fill(std::numbers::e);  // Fills the matrix with the constant e
        return mat;
    }

    template <int shared, int col2>
    static Matrix<row, col2> multiply(const Matrix<row, shared>& A, const Matrix<shared, col2>& B) {
        Matrix<row, col2> result;
        for ( i uint16_t= 0; i < row; ++i) {
            for (uint16_t j = 0; j < col2; ++j) {
                for (uint16_t k = 0; k < shared; ++k) {
                    result.data[i][j] += A.data[i][k] * B.data[k][j];
                }
            }
        }
        return result;
    }
};

template <int N>
static Matrix<N, N> invert(const Matrix<N, N>& input) {
    Matrix<N, N> A = input;
    Matrix<N, N> I;

    // Initialize identity matrix
    for (int i = 0; i < N; ++i)
        I.data[i][i] = 1.0;

    // Gauss-Jordan elimination
    for (int i = 0; i < N; ++i) {
        // Pivot check
        if (A.data[i][i] == 0) {
            // Find non-zero pivot and swap
            for (uint16_t j = i + 1; j < N; ++j) {
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

        // Normalize the row
        for (uint16_t j = 0; j < N; ++j) {
            A.data[i][j] /= pivot;
            I.data[i][j] /= pivot;
        }

        // Eliminate other rows
        for (uint16_t k = 0; k < N; ++k) {
            if (k == i) continue;
            double factor = A.data[k][i];
            for (uint16_t j = 0; j < N; ++j) {
                A.data[k][j] -= factor * A.data[i][j];
                I.data[k][j] -= factor * I.data[i][j];
            }
        }
    }

    return I; // Inverse matrix
}

