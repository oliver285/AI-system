#ifndef MLP_H
#define MLP_H
// #define UNIT_TESTING
#ifdef UNIT_TESTING
class MLPTest;  // Forward declaration
#endif
#include "matrix.h"

class MLP {
private:
    // Weights and gradients
    Matrix W1, vW1, W2, vW2, dW1, dW2;
    Matrix b1, vb1, b2, vb2, db1, db2;
    
    // Activations and intermediates
    Matrix Z1, A1, dZ1;
    Matrix Z2, A2, dZ2;
    
    // size_t batch_size; Defined locally since depends on function

public:
    // Constructor
    MLP(size_t input_size, size_t hidden_size, size_t output_size);

//Accessors
// double getA1(size_t row,size_t col) {return A1(row,col);}
// double getA2(size_t row,size_t col) {return A2(row,col);}
// double getZ1(size_t row,size_t col) {return Z1(row,col);}
// double getZ2(size_t row,size_t col) {return Z2(row,col);}
// double getW1(size_t row,size_t col) {return W1(row,col);}
// double getW2(size_t row,size_t col) {return W2(row,col);}
// double getb1(size_t row,size_t col) {return b1(row,col);}
// double getb2(size_t row,size_t col) {return b2(row,col);}

    
    // Core operations
    Matrix forward_prop(const Matrix& X);
    void back_prop(const Matrix& X, const Matrix& Y);
    void update_params(double learning_rate);
    double compute_loss(const Matrix& Y, const Matrix& A2);
    
    // Accessors
    const Matrix& get_A1() const { return A1; }
    const Matrix& get_A2() const { return A2; }
    const Matrix& get_dZ2() const { return dZ2; }
    const Matrix& get_dW1() const { return dW1; }
    const Matrix& get_db1() const { return db1; }
    const Matrix& get_dW2() const { return dW2; }
    const Matrix& get_db2() const { return db2; }
    const Matrix& get_W1() const { return W1; }
    const Matrix& get_b1() const { return b1; }
    const Matrix& get_W2() const { return W2; }
    const Matrix& get_b2() const { return b2; }
    const Matrix& get_vW1() const { return vW1; }
    const Matrix& get_vb1() const { return vb1; }
    const Matrix& get_vW2() const { return vW2; }
    const Matrix& get_vb2() const { return vb2; }
       // Add similar accessors for other needed matrices
   
       // Const matrix views
       const Matrix& view_dW1() const { return dW1; }
       const Matrix& view_db1() const { return db1; }
       // Add similar views for other needed matrices
   
       // Setters for gradients (for ParameterUpdate test)
      void set_W1(const Matrix& weights) { W1 = weights; }
      void set_W2(const Matrix& weights) { W2 = weights; }
      void set_b1(const Matrix& bias) { b1 = bias; }
      void set_b2(const Matrix& bias) { b2 = bias; }
      void set_dW1(const Matrix& m) { dW1 = m; }
      void set_db1(const Matrix& m) { db1 = m; }
      void set_dW2(const Matrix& m) { dW2 = m; }
      void set_db2(const Matrix& m) { db2 = m; }
      void set_vW1(const Matrix& m) { vW1 = m; }
      void set_vb1(const Matrix& m) { vb1 = m; }
      void set_vW2(const Matrix& m) { vW2 = m; }
      void set_vb2(const Matrix& m) { vb2 = m; }
       // Utility functions
    Matrix one_hot(const Matrix& Y, size_t num_classes = 2);
    Matrix get_predictions(const Matrix& A);
    double get_accuracy(const Matrix& predictions, const Matrix& labels);
   double cross_entropy_loss(const Matrix& Y_pred, const Matrix& Y_true);
    // Training loop
    void gradient_descent(Matrix& X,Matrix& Y, 
                         size_t iterations, double learning_rate);

                         
    #ifdef UNIT_TESTING
    friend class MLPTest;  // Grant test access
    #endif
};

#endif // MLP_H