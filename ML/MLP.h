#ifndef MLP_H
#define MLP_H

#include "matrix.h"

class MLP {
private:
    // Weights and gradients
    Matrix W1, vW1, W2, vW2, dW1, dW2;
    Matrix b1, vb1, b2, vb2, db1, db2;
    
    // Activations and intermediates
    Matrix Z1, A1, dZ1;
    Matrix Z2, A2, dZ2;
    
    size_t batch_size;

public:
    // Constructor
    MLP(size_t input_size, size_t hidden_size, size_t output_size);
    
    // Core operations
    Matrix forward_prop(const Matrix& X);
    void back_prop(const Matrix& X, const Matrix& Y);
    void update_params(double learning_rate);
    double compute_loss(const Matrix& Y, const Matrix& A2)   ;
    // Utility functions
    Matrix one_hot(const Matrix& Y, size_t num_classes = 2);
    Matrix get_predictions(const Matrix& A);
    double get_accuracy(const Matrix& predictions, const Matrix& labels);
   double cross_entropy_loss(const Matrix& Y_pred, const Matrix& Y_true);
    // Training loop
    void gradient_descent(Matrix& X,Matrix& Y, 
                         size_t iterations, double learning_rate);
};

#endif // MLP_H