#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

class Matrix {
   
   size_t rows , cols;
    std::vector<double> data;
    
public:
Matrix(){}
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c) {}  // Fixed
  
  // Access element (i,j)
    // Access element (i,j) with bounds checking
    // Dual operator() for const and non-const access
    double& operator()(size_t i, size_t j) {
        if (i >= rows || j >= cols)
            throw std::out_of_range("Matrix index out of range");
        return data[i * cols + j];
    }
    
    const double& operator()(size_t i, size_t j) const {
        if (i >= rows || j >= cols)
            throw std::out_of_range("Matrix index out of range");
        return data[i * cols + j];
    }
// Accessors
size_t row_count() const  { return rows; }
size_t col_count() const  { return cols; }
size_t size() const {return data.size();}
const double& no_bounds_check(size_t i) const{
    return data[i];
}
double& no_bounds_check(size_t i){
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
static Matrix random(size_t r, size_t c) {
    Matrix mat(r, c);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            mat(i, j) = dist(gen);
    
    return mat;
}

double frobenius_norm() const {
    double val = 0;
    for (size_t i = 0; i < size(); ++i) {
        double element = no_bounds_check(i);
        val += element * element;
    }
    return std::sqrt(val);  // CORRECT: sqrt(sum of squares)
}



    // Element-wise operations
    Matrix operator+(const Matrix& other) const {
   // Check for matching dimensions
   if (rows != other.rows || cols != other.cols) {
    throw std::invalid_argument("Matrix dimensions must match for addition");
}
        Matrix result(rows,cols);

 for (size_t i = 0; i < rows; ++i)
 for (size_t j = 0; j < cols; ++j)
                result(i,j)=(*this)(i,j)+other(i,j);
        return result;
    }

    Matrix operator-(const Matrix& other) const {
if(other.cols!=cols || other.rows!=rows){
    throw std::invalid_argument("Matrix dimensions must match for subtraction you worthless basement lurking motherfucker");
}

        Matrix result(rows,cols);
 

        for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
        result(i,j)=(*this)(i,j)-other(i,j);
        return result;
    } 

    // Scalar operations
    Matrix multiply_scalar(double scalar) const {
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

Matrix subtract_scalar(double scalar) const {
    Matrix result(rows,cols);
 

    //  const int act_cols = cols==-1 ? dynamic_cols : cols;
    for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
    
    result(i,j)=(*this)(i,j)-scalar;
    return result;
}


 // Matrix multiplication with proper access
 
 static Matrix multiply(const Matrix& A, const Matrix& B) {
    // Validate matrix dimensions with detailed error message
    if (A.cols != B.rows) {
        throw std::invalid_argument(
            "Matrix multiplication dimension mismatch: A.cols (" + 
            std::to_string(A.cols) + ") != B.rows (" + 
            std::to_string(B.rows) + ")"
        );
    }

    Matrix result(A.rows, B.cols);
    const size_t inner_dim = A.cols;  // Consistent size type

    // Optimized multiplication with row-major ordering
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            double sum = 0.0;
            // Inner loop unrolling could be added here
            for (size_t k = 0; k < inner_dim; ++k) {
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
Matrix transpose() const {
    
        // Static case
        Matrix result(cols,rows);
        for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)

            // Direct data access for better performance
            result.data[j * rows + i] = data[i * cols + j];
                // result.set(j, i, get(i, j));
        return result;
    
}


    // Activation functions
// ReLU activation function (optimized version)
Matrix RELU() const {
    Matrix result(rows, cols);
    
    // Direct data access for better performance
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::max(0.0, data[i]);
    }
    
    return result;
}

    Matrix deriv_RELU() const {
        Matrix result(rows,cols);
 
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] > 0 ? 1.0 : 0.0;
             
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


    static Matrix softmax(const Matrix& A) {
        Matrix result(A.rows, A.cols);  // Initialize result matrix
    
        // 1. Find max for numerical stability (per column)
        for (size_t j = 0; j < A.cols; ++j) {
            double max_val = A(0, j);
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
double sum() const {
    double total = 0.0;
    
    // Direct data access for better performance
    for (const auto& val : data) {
        total += val;
    }
    
    return total;
}

double min() const {
    if (size() == 0) return 0.0;
    double min_val = no_bounds_check(0);
    for (size_t i = 1; i < size(); i++) {
        if (no_bounds_check(i) < min_val) {
            min_val = no_bounds_check(i);
        }
    }
    return min_val;
}

double max() const {
    if (size() == 0) return 0.0;
    double max_val = no_bounds_check(0);
    for (size_t i = 1; i < size(); i++) {
        if (no_bounds_check(i) > max_val) {
            max_val = no_bounds_check(i);
        }
    }
    return max_val;
}

double mean(){
double mean=0;
for(size_t i=0;i<size();i++){
mean+=no_bounds_check(i);
}

return mean/size();

}

 // Column-wise sum (optimized for batch processing)
static Matrix sum_cols(const Matrix& A) {
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

    // Fill with value
    void fill(double value) {


        // const int actual_cols = cols==-1 ? dynamic_cols:cols;

       for(size_t i=0;i<data.size();i++){
        data[i]=value;
       }
    
    // set(i,j,value);
     
    
    }

    void I(){

        fill(0);
        for (size_t i = 0; i < rows; ++i){
            for (size_t j = 0; j < cols; ++j) {
            if(i==j) (*this)(i,j)=1;

            }
        }

    }

    // Print matrix
    void print() const {

        for (size_t i = 0; i < rows; ++i){
            for (size_t j = 0; j < cols; ++j) {
                std::cout << (*this)(i,j)<< " ";
            }
            std::cout << "\n";
        }
    }

    Matrix hadamard_product(const Matrix& A) const {
        // Check for matching dimensions
        if (rows != A.rows || cols != A.cols) {
            throw std::invalid_argument(
                "Matrix dimensions must match for Hadamard product. " +
                std::to_string(rows) + "x" + std::to_string(cols) + " vs " +
                std::to_string(A.rows) + "x" + std::to_string(A.cols)
            );
        }
    
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) * A(i, j);
            }
        }
        return result;
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




