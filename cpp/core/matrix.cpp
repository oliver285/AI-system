#include "matrix.h"
#include <cassert>
    Matrix::Matrix(){}
    Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c) {}  // Fixed
  
  // Access element (i,j)
    // Access element (i,j) with bounds checking
    // Dual operator() for const and non-const access
    double& Matrix::operator()(size_t row, size_t col, Error* err) {
        if (row >= rows || col >= cols) {
            if (err) *err = INDEX_OUT_OF_RANGE;
            thread_local double dummy = 0.0;
            return dummy;
        }
        return data[row * cols + col];
    }
    
    const double& Matrix::operator()(size_t row, size_t col, Error* err) const {
        if (row >= rows || col >= cols) {
            if (err) *err = INDEX_OUT_OF_RANGE;
            thread_local double dummy = 0.0;
            return dummy;
        }
        return data[row * cols + col];
    }
// Accessors
size_t  Matrix::row_count() const  { return rows; }
size_t  Matrix::col_count() const  { return cols; }
size_t  Matrix::size() const {return data.size();}
const double&  Matrix::no_bounds_check(size_t i) const{
    return data[i];
}
double&  Matrix::no_bounds_check(size_t i){
    return data[i];
}



// double get(int i, int j) const { 
//                         if (i < 0 || i >= row_count() || j < 0 || j >= col_count())
//         throw std::out_of_range("Matrix index out of range");
                        
//                         return data[i][j]; }
//  void set(int i, int j, double value) {
//                         if (i < 0 || i >= row_count() || j < 0 || j >= col_count())
//                         throw std::out_of_range("Matrix index out of range");                    
//     data[i][j] = value; }
                    
    // Random initialization with scaling
// Random initialization (Gaussian)
 Matrix  Matrix::random(size_t r, size_t c) {
    Matrix mat(r, c);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            mat(i, j) = dist(gen);
    
    return mat;
}

// Add to Matrix class
double  Matrix::frobenius_norm() const {
    double val = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        val += data[i] * data[i];
    }
    return std::sqrt(val);
}

Matrix  Matrix::clip(double min_val, double max_val) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::max(min_val, std::min(data[i], max_val));
    }
    return result;
}



    // Element-wise operations
    Matrix Matrix::add(const Matrix& other, Error* err) const {
        Matrix result(rows, cols);
        if (rows != other.rows || cols != other.cols) {
            if (err) *err = DIMENSION_MISMATCH;
            result.fill(0.0);
            return result;
        }
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result(i, j) = (*this)(i, j) + other(i, j);
        return result;
    }
    
    Matrix Matrix::subtract(const Matrix& other, Error* err) const {
        Matrix result(rows, cols);
        if (rows != other.rows || cols != other.cols) {
            if (err) *err = DIMENSION_MISMATCH;
            result.fill(0.0);
            return result;
        }
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result(i, j) = (*this)(i, j) - other(i, j);
        return result;
    }

        // Add this method
        void  Matrix::scale_inplace(double scalar) {
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] *= scalar;
            }
        }

    // Scalar operations
    Matrix  Matrix::multiply_scalar(double scalar) const {
        Matrix result(rows,cols);
 

        //  const int act_cols = cols==-1 ? dynamic_cols : cols;
        for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
        
        result(i,j)=(*this)(i,j)*scalar;
        return result;
    }


    // Alternative version using direct data access (for performance)
// Matrix multiply_scalar(double scalar) const {
//     Matrix result(rows, cols);
    
//     // Direct memory access - faster but less safe
//     for (size_t i = 0; i < data.size(); ++i) {
//         result.data[i] = data[i] * scalar;
//     }
    
//     return result;
// }

Matrix  Matrix::subtract_scalar(double scalar) const {
    Matrix result(rows,cols);
 

    //  const int act_cols = cols==-1 ? dynamic_cols : cols;
    for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
    
    result(i,j)=(*this)(i,j)-scalar;
    return result;
}


 // Matrix multiplication with proper access
 
 Matrix Matrix::multiply(const Matrix& A, const Matrix& B, Error* err) {
    if (A.cols != B.rows) {
        if (err) *err = DIMENSION_MISMATCH;
        return Matrix();
    }

    Matrix result(A.rows, B.cols);
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}
    // Transpose
    // Transpose with proper access
// Static transpose (when neither rows nor cols is dynamic)
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // Use safe access instead of direct data manipulation
            result(j, i) = (*this)(i, j);  // ✅ Better maintainability
        }
    }
    return result;
}


    // Activation functions
// ReLU activation function (optimized version)
Matrix  Matrix::RELU() const {
    Matrix result(rows, cols);
    
    // Direct data access for better performance
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::max(0.0, data[i]);
    }
    
    return result;
}

    Matrix  Matrix::deriv_RELU() const {
        Matrix result(rows,cols);
 
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] >= 0 ? 1.0 : 0.0;
             
        }
        return result;
    }


    Matrix  Matrix::leaky_RELU(double alpha) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = (data[i] > 0) ? data[i] : alpha * data[i];
        }
        return result;
    }
    
    Matrix  Matrix::deriv_leaky_RELU(double alpha) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = (data[i] >= 0) ? 1.0 : alpha;
        }
        return result;
    }
//Potential parallelism implementation
    // Matrix deriv_RELU_parallel() const {
    //     Matrix result(rows, cols);
    //     #pragma omp parallel for
    //     for (size_t i = 0; i < data.size(); ++i) {
    //         result.data[i] = data[i] > 0.0 ? 1.0 : 0.0;
    //     }
    //     return result;
    // }


 Matrix  Matrix::softmax(const Matrix& A) {
        Matrix result(A.rows, A.cols);  // Initialize result matrix
    
        // 1. Find max for numerical stability (per column)
        for (size_t j = 0; j < A.cols; ++j) {
            double max_val = -std::numeric_limits<double>::max();
            for (size_t i = 0; i < A.rows; ++i) {
                if (A(i, j) > max_val) max_val = A(i, j);
            }
    
            // 2. Compute exponentials and sum (per column)
            double sum = 0.0;
            for (size_t i = 0; i < A.rows; ++i) {
                result(i, j) = std::exp(A(i, j) - max_val);
                sum += result(i, j);
            }
    
            // 3. Normalize (per column)
            for (size_t i = 0; i < A.rows; ++i) {
                result(i, j) /= sum;
            }
        }
    
        return result;
    }


/* Softmax function alternative 
static -Matrix log_softmax(const Matrix& A) {
    Matrix result(A.rows, A.cols);
    
    for (size_t j = 0; j < A.cols; ++j) {
        double max_val = A(0, j);
        for (size_t i = 0; i < A.rows; ++i) {
            max_val = std::max(max_val, A(i, j));
        }

        double sum = 0.0;
        for (size_t i = 0; i < A.rows; ++i) {
            sum += std::exp(A(i, j) - max_val);
        }

        const double log_sum = max_val + std::log(sum);
        for (size_t i = 0; i < A.rows; ++i) {
            result(i, j) = A(i, j) - log_sum;
        }
    }
    
    return result;
}*/


    // Sum of all elements
  // Sum of all elements (optimized)
double  Matrix::sum() const {
    double total = 0.0;
    
    // Direct data access for better performance
    for (const auto& val : data) {
        total += val;
    }
    
    return total;
}

double  Matrix::min() const {
    if (size() == 0) return 0.0;
    double min_val = no_bounds_check(0);
    for (size_t i = 1; i < size(); i++) {
        if (no_bounds_check(i) < min_val) {
            min_val = no_bounds_check(i);
        }
    }
    return min_val;
}

double  Matrix::max() const {
    if (size() == 0) return 0.0;
    double max_val = no_bounds_check(0);
    for (size_t i = 1; i < size(); i++) {
        if (no_bounds_check(i) > max_val) {
            max_val = no_bounds_check(i);
        }
    }
    return max_val;
}
 
double  Matrix::mean() const {
    double sum = 0.0;
    for (double val : data) sum += val;
    return sum / data.size();
}
Matrix&  Matrix::scale(double factor) {
    for (double& val : data) val *= factor;
    return *this;
}


 // Column-wise sum (optimized for batch processing)
 Matrix  Matrix::sum_cols(const Matrix& A) {
    Matrix result(A.rows, 1);  // Column vector to store sums
    
    for (size_t i = 0; i < A.rows; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < A.cols; ++j) {
            row_sum += A(i, j);  // Sum across columns for each row
        }
        result(i, 0) = row_sum;
    }
    
    return result;
}


Matrix Matrix::rowwise_mean() const {
    Matrix result(rows, 1);
    for (size_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (size_t c = 0; c < cols; ++c)
            sum += data[r * cols + c];
        result(r, 0) = sum / cols;
    }
    return result;
}

Matrix Matrix::rowwise_std(double epsilon) const {
    Matrix means = this->rowwise_mean();
    Matrix result(rows, 1);
    
    for (size_t r = 0; r < rows; ++r) {
        double sq_sum = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            double diff = data[r * cols + c] - means(r, 0);
            sq_sum += diff * diff;
        }
        result(r, 0) = std::sqrt(sq_sum / cols + epsilon);
    }
    return result;
}

Matrix& Matrix::subtract_rowwise(const Matrix& vec, Error* err) {
    if (vec.rows != rows || vec.cols != 1) {
        if (err) *err = DIMENSION_MISMATCH;
        return *this; // Safe return
    }
    
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            (*this)(r,c) -= vec(r, 0);
    
    return *this;
}

    // Fill with value
    void  Matrix::fill(double value) {


        // const int actual_cols = cols==-1 ? dynamic_cols:cols;

       for(size_t i=0;i<data.size();i++){
        data[i]=value;
       }
    
    // set(i,j,value);
     
    
    }

    void  Matrix::I(){

        fill(0);
        for (size_t i = 0; i < rows; ++i){
            for (size_t j = 0; j < cols; ++j) {
            if(i==j) (*this)(i,j)=1;

            }
        }

    }

    // Print matrix
    void  Matrix::print() const {

        for (size_t i = 0; i < rows; ++i){
            for (size_t j = 0; j < cols; ++j) {
                std::cout << (*this)(i,j)<< " ";
            }
            std::cout << "\n";
        }
    }

    Matrix  Matrix::hadamard_product(const Matrix& A,Error* err) const {
        // Check for matching dimensions
        if (rows != A.rows || cols != A.cols) {
            if (err) *err = DIMENSION_MISMATCH;
            return Matrix();
        }
    
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) * A(i, j);
            }
        }
        return result;
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
//     A.fill(2);
//     std::cout << "Matrix A:\n";
//     A.print();
