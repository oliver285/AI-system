#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

template <int rows, int cols = -1>  // cols=-1 means dynamic
class Matrix {
    int dynamic_cols;  // Only used if cols==-1
    std::vector<std::vector<double>> data;
    
public:
    // Constructor for dynamic columns
    Matrix(int c = -1) :
     dynamic_cols(cols == -1 ? c : cols), 
                       data(rows, std::vector<double>(dynamic_cols)) {}
                       int col_count() const { return cols == -1 ? dynamic_cols : cols; }
                       int row_count() const { return rows; }
                       double get(int i, int j) const { 
                        if (i < 0 || i >= row_count() || j < 0 || j >= col_count())
        throw std::out_of_range("Matrix index out of range");
                        
                        return data[i][j]; }
                       void set(int i, int j, double value) {
                        if (i < 0 || i >= row_count() || j < 0 || j >= col_count())
                        throw std::out_of_range("Matrix index out of range");                    
    data[i][j] = value; }
                    
    // Random initialization with scaling
    static Matrix<rows, cols> random(int runtime_cols=1) {
        Matrix<rows, cols> mat;
        if constexpr (cols==-1){

mat = Matrix<rows,cols>(runtime_cols);
 }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 1.0);

       
        for (int i = 0; i < mat.row_count(); ++i)
            for (int j = 0; j < mat.col_count(); ++j)
                mat.set(i,j,dist(gen));
        return mat;

        
    }

    // Element-wise operations
    Matrix<rows, cols> operator+(const Matrix<rows, cols>& other) const {

        Matrix<rows, cols> result;
        if constexpr (cols==-1){
            assert(col_count() == other.col_count());
result = Matrix<rows,cols>(other.col_count());
 }
//  const int actual_cols = cols==-1 ? dynamic_cols:cols;

 for (int i = 0; i < row_count(); ++i)
 for (int j = 0; j < col_count(); ++j)
                result.set(i,j,get(i,j) + other.get(i,j));
        return result;
    }

    Matrix<rows, cols> operator-(const Matrix<rows, cols>& other) const {


        Matrix<rows, cols> result;
        if constexpr (cols==-1){
            assert(col_count() == other.col_count());
result = Matrix<rows,cols>(other.col_count());
 }
 

 for (int i = 0; i < row_count(); ++i)
 for (int j = 0; j < col_count(); ++j)
 result.set(i,j,get(i,j)-  other.get(i,j));
        return result;
    }

    // Scalar operations
    Matrix<rows, cols> multiply_scalar(double scalar) const {
        Matrix<rows, cols> result;
        if constexpr (cols==-1){
           
result = Matrix<rows,cols>(col_count());
 }

        //  const int act_cols = cols==-1 ? dynamic_cols : cols;
         for (int i = 0; i < row_count(); ++i)
         for (int j = 0; j < col_count(); ++j)
         result.set(i,j,get(i,j)*scalar);
        return result;
    }

    Matrix<rows, cols> subtract_scalar(double scalar) const {
        Matrix<rows, cols> result;
        
        if constexpr (cols==-1){
          
result = Matrix<rows,cols>(col_count());
 }


//  const int act_cols = cols==-1 ? dynamic_cols : cols;
 for (int i = 0; i < row_count(); ++i)
 for (int j = 0; j < col_count(); ++j)
 result.set(i,j,get(i,j) - scalar);
        return result;
    }

 // Matrix multiplication with proper access
 template <int other_cols>
 static Matrix<rows, other_cols> multiply(const Matrix<rows, cols>& A,
                                        const Matrix<cols, other_cols>& B) {
    //  Matrix<rows, other_cols> result;
    // if constexpr (cols == -1 || other_cols == -1) {
    //     if (A.col_count() != B.row_count())
    //         throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    // }
     if constexpr (other_cols == -1) {
        

     Matrix<rows, other_cols> Results(B.col_count());
     }

     Matrix<rows, other_cols> Results(B.col_count());
     const int inner_dim = cols == -1 ? A.col_count() : cols;
     const int result_cols = other_cols == -1 ? B.col_count() : other_cols;

     for (int i = 0; i < A.row_count(); ++i) {
         for (int j = 0; j < result_cols; ++j) {
             double sum = 0;
             for (int k = 0; k < inner_dim; ++k) {
                 sum += A.get(i, k) * B.get(k, j);
             }
             Results.set(i, j, sum);
         }
     }
     return Results;
 }
    // Transpose
    // Transpose with proper access
    Matrix<cols, rows> transpose() const {
 
        if constexpr (rows == -1) {

            const int ros=row_count();
            Matrix<cols, -1> result(ros);


        }
        Matrix<cols, rows> result(row_count());
        const int actual_cols = col_count();
        for (int i = 0; i < row_count(); ++i) {
            for (int j = 0; j < actual_cols; ++j) {
                result.set(j, i, get(i, j));
            }
        }
        return result;
    }

    // Activation functions
    Matrix<rows, cols> RELU() const {
        Matrix<rows, cols> result;
        if constexpr (cols==-1){

result = Matrix<rows,cols>(col_count());
 }
 const int actual_cols = col_count();
        for (int i = 0; i < row_count(); ++i)
            for (int j = 0; j < actual_cols; ++j)
                result.set(i,j,std::max(0.0,get(i,j)));
        return result;
    }

    Matrix<rows, cols> deriv_RELU() const {
        Matrix<rows, cols> result;
        if constexpr (cols==-1){

result = Matrix<rows,cols>(col_count());
 }
 const int actual_cols = col_count();
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < actual_cols; ++j)
                result.set(i,j, (get(i,j) > 0) ? 1 : 0);
        return result;
    }

    // Softmax
    static Matrix<rows, cols> softmax(const Matrix<rows, cols>& A) {
        Matrix<rows, cols> result;
        if constexpr (cols==-1){

result = Matrix<rows,cols>(A.col_count());
 }
//  const int actual_cols = cols==-1 ? dynamic_cols:cols;
        // Find max for numerical stability
        double max_val = A.get(0,0);
        for (int i = 0; i < A.row_count(); ++i)
            for (int j = 0; j < A.col_count(); ++j)
                if (A.get(i,j) > max_val) max_val = A.get(i,j);
        
        // Compute exponentials
        double sum = 0;
        for (int i = 0; i < A.row_count(); ++i){
        for (int j = 0; j < A.col_count(); ++j) {
                result.set(i,j,std::exp(A.get(i,j) - max_val));
                sum += result.get(i,j);
            }
        }
        
        // Normalize
        for (int i = 0; i < A.row_count(); ++i)
            for (int j = 0; j < A.col_count(); ++j)
                result.set(i,j,result.get(i,j) / sum);
                
        return result;
    }

    // Sum of all elements
    double sum() const {
        // const int actual_cols = cols==-1 ? dynamic_cols:cols;
        double total = 0;
        for (int i = 0; i < row_count(); ++i)
            for (int j = 0; j < col_count(); ++j)
                total += get(i,j);
        return total;
    }

    // Column-wise sum (for batch processing)
    static Matrix<rows, 1> sum_cols(const Matrix<rows, cols>& A) {
        Matrix<rows, 1> result;
        const int actual_cols =  A.col_count();
         double sum =  0;
        for (int i = 0; i < A.row_count(); ++i){
            sum=0;
        for (int j = 0; j < actual_cols; ++j){
    sum+=A.get(i,j);
               
                }
                result.set(i,0, sum);


            }
        return result;
    }

    // Fill with value
    void fill(double value) {


        // const int actual_cols = cols==-1 ? dynamic_cols:cols;

        for (int i = 0; i < row_count(); ++i)
        for (int j = 0; j < col_count(); ++j) 
    set(i,j,value);
                
    }

    // Print matrix
    void print() const {

        for (int i = 0; i < row_count(); ++i){
            for (int j = 0; j < col_count(); ++j) {
                std::cout << get(i,j)<< " ";
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




