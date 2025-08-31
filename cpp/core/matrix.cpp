#include "matrix.h"
#include <cassert>
    Matrix::Matrix(){}
    Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c) {
            if (rows * cols > 1e8) { // adjust threshold
        std::cerr << "⚠️  HUGE allocation: " << rows << " x " << cols << " = " << rows*cols << " elements\n";
    }
    }  // Fixed
  
  // Access element (i,j)
    // Access element (i,j) with bounds checking
    // Dual operator() for const and non-const access
 float& Matrix::operator()(size_t row, size_t col, Error* err) {
    if (row >= rows || col >= cols) {
        if (err) *err = INDEX_OUT_OF_RANGE;
        thread_local float dummy = 0.0f;
        return dummy;
    }
    return data[row * cols + col];
}

const float& Matrix::operator()(size_t row, size_t col, Error* err) const {
    if (row >= rows || col >= cols) {
        if (err) *err = INDEX_OUT_OF_RANGE;
        thread_local float dummy = 0.0f;
        return dummy;
    }
    return data[row * cols + col];
}

Matrix Matrix::operator+(float val) const {
    Matrix res(rows, cols);
    size_t n = data.size();

    for (size_t i = 0; i < n; ++i) {
        res.data[i] = data[i] + val;
    }

    return res;
}

void Matrix::add_inplace(float val) {


    for (size_t i = 0; i < data.size(); ++i) {
         data[i] += val;
    }

}

// Accessors
size_t  Matrix::row_count() const  { return rows; }
size_t  Matrix::col_count() const  { return cols; }
size_t  Matrix::size() const {return data.size();}
const float&  Matrix::no_bounds_check(size_t i) const{
    return data[i];
}
float&  Matrix::no_bounds_check(size_t i){
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
// Random initialization with Gaussian
Matrix Matrix::random(size_t r, size_t c) {
    Matrix mat(r, c);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            mat(i, j) = dist(gen);

    return mat;
}

// Add to Matrix class
float Matrix::frobenius_norm() const {
    float val = 0.0f;
    for (size_t i = 0; i < data.size(); ++i)
        val += data[i] * data[i];
    return std::sqrt(val);
}

Matrix Matrix::clip(float min_val, float max_val) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i)
        result.data[i] = std::max(min_val, std::min(data[i], max_val));
    return result;
}


Matrix Matrix::slice_cols(size_t start, size_t end,Error *err) const {
    //     size_t new_cols = end - start;
    // std::cout << "slice_cols: start=" << start
    //       << " end=" << end
    //       << " rows=" << rows
    //       << " cols=" << cols
    //       << " new_cols=" << new_cols
    //       << " => result: " << rows << " x " << new_cols
    //       << "\n";
    if (start >= cols || end > cols || start >= end) {
           if (err) *err = DIMENSION_MISMATCH;
           
    return Matrix(0, 0);
    }
    size_t new_cols = end - start;
    Matrix result(rows, new_cols);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < new_cols; ++c) {
            result(r, c) = (*this)(r, start + c);
        }
    }
    return result;
}



Matrix Matrix::add(const Matrix& other, Error* err) const {
    Matrix result(rows, cols);
    if (rows != other.rows || cols != other.cols) {
        if (err) *err = DIMENSION_MISMATCH;
        result.fill(0.0f);
        return result;
    }
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result(i, j) = (*this)(i, j) + other(i, j);
    return result;
}

// Same for subtract, multiply_scalar, subtract_scalar, scale_inplace

    
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
        void  Matrix::scale_inplace(float scalar) {
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] *= scalar;
            }
        }

    // Scalar operations
Matrix Matrix::multiply_scalar(float scalar) const {
    Matrix result(rows, cols);
    result.data.resize(rows * cols);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Matrix Matrix::divide_scalar(float scalar,Error* err) const {
    if (scalar == 0.0f) {
      if(err) *err=DIVIDE_BY_ZERO;
    }

    Matrix result(rows, cols);
    result.data.resize(rows * cols);

    float inv = 1.0f / scalar;  // multiply instead of divide for speed
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * inv;
    }
    return result;
}

void Matrix::multiply_scalar_inplace(float scalar) {
    for (size_t i = 0; i < data.size(); ++i) data[i] *= scalar;
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

Matrix  Matrix::subtract_scalar(float scalar) const {
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
        result.data[i] = std::max(0.0f, data[i]);
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


    Matrix  Matrix::leaky_RELU(float alpha) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = (data[i] > 0) ? data[i] : alpha * data[i];
        }
        return result;
    }
    
    Matrix  Matrix::deriv_leaky_RELU(float alpha) const {
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


Matrix Matrix::softmax(const Matrix& A) {
    Matrix result(A.rows, A.cols);
    for (size_t j = 0; j < A.cols; ++j) {
        float max_val = -std::numeric_limits<float>::max();
        for (size_t i = 0; i < A.rows; ++i)
            if (A(i, j) > max_val) max_val = A(i, j);

        float sum = 0.0f;
        for (size_t i = 0; i < A.rows; ++i) {
            result(i, j) = std::exp(A(i, j) - max_val);
            sum += result(i, j);
        }
        for (size_t i = 0; i < A.rows; ++i)
            result(i, j) /= sum;
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
float Matrix::sum() const {
    float total = 0.0f;
    for (const auto& val : data) total += val;
    return total;
}

float Matrix::min() const {
    if (size() == 0) return 0.0f;
    float min_val = data[0];
    for (size_t i = 1; i < size(); ++i)
        if (data[i] < min_val) min_val = data[i];
    return min_val;
}

float Matrix::max() const {
    if (size() == 0) return 0.0f;
    float max_val = data[0];
    for (size_t i = 1; i < size(); ++i)
        if (data[i] > max_val) max_val = data[i];
    return max_val;
}

float Matrix::mean() const {
    return sum() / data.size();
}


Matrix&  Matrix::scale(float factor) {
    for (float& val : data) val *= factor;
    return *this;
}


 // Column-wise sum (optimized for batch processing)
 Matrix  Matrix::sum_cols(const Matrix& A) {
    Matrix result(A.rows, 1);  // Column vector to store sums
    
    for (size_t i = 0; i < A.rows; ++i) {
        float row_sum = 0.0f;
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

Matrix Matrix::rowwise_std(float epsilon) const {
    Matrix means = rowwise_mean();
    Matrix result(rows, 1);
    for (size_t r = 0; r < rows; ++r) {
        float sq_sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float diff = data[r * cols + c] - means(r, 0);
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
    void  Matrix::fill(float value) {


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

Matrix Matrix::hadamard_product(const Matrix& A, Error* err) const {
    if (rows != A.rows || cols != A.cols) {
        if (err) *err = DIMENSION_MISMATCH;
        return Matrix();
    }

    Matrix result(rows, cols);
    size_t total = rows * cols;

    for (size_t idx = 0; idx < total; ++idx) {
        result.data[idx] = this->data[idx] * A.data[idx];
    }

    return result;
}

Matrix Matrix::sqrt() const {
    Matrix res(rows, cols);
    for (size_t i = 0; i < data.size(); i++) {
        res.data[i] = std::sqrt(data[i]);
    }
    return res;
}

 void Matrix::sqrt_inplace() {

    for (size_t i = 0; i < data.size(); i++) {
        data[i] = std::sqrt(data[i]);
    }
   
 }


     Matrix Matrix::hadamard_division(const Matrix& A, Error* err) const {
    if (rows != A.rows || cols != A.cols) {
        if (err) *err = DIMENSION_MISMATCH;
        return Matrix();
    }

    Matrix result(rows, cols);
    size_t total = rows * cols;

    for (size_t idx = 0; idx < total; ++idx) {
        result.data[idx] = this->data[idx] / A.data[idx]; 
        // Cleaner than multiply by reciprocal, compiler will optimize
    }

    return result;
}

void Matrix::hadamard_division_inplace(const Matrix& A, Error* err) {
    if (rows != A.rows || cols != A.cols) {
        if (err) *err = DIMENSION_MISMATCH;
        return;
    }
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] /= A.data[i];
    }
}



void Matrix::add_inplace(const Matrix& other,float alpha) {
    // float alpha=1.0f;
    // std::cout << "add_inplace: this(" << rows << "," << cols << ") other(" 
    //           << other.rows << "," << other.cols << ")\n";
    assert(rows == other.rows && cols == other.cols);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += alpha * other.data[i];
    }
}

void Matrix::add_inplace_reg(float val){
      for (size_t i = 0; i < data.size(); ++i) {
        data[i] += val;
    }
}

void Matrix::subtract_inplace(const Matrix& other,float alpha) {
    //  float alpha=1.0f;
    assert(rows == other.rows && cols == other.cols);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= alpha * other.data[i];
    }
}


void Matrix::subtract_inplace_element(const Matrix& other) {
    //  float alpha=1.0f;
     assert(rows == other.rows && cols == other.cols);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -=  other.data[i];
    }
}


void Matrix::add_inplace_squared(const Matrix& other, float alpha) {
    assert(rows == other.rows && cols == other.cols);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += alpha * (other.data[i] * other.data[i]); // Correct: add squared gradients
    }
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
