#ifndef MLP_H
#define MLP_H

#ifdef UNIT_TESTING
class MLPTest; // Forward declaration
  void update_layer_params(param, v, s, lr_corrected, epsilon, err);
#endif

#include "../core/matrix.h"
 void update_layer_params(Matrix& param, Matrix& v, Matrix& s, 
                               float lr_corrected, float epsilon, Error* err);
class MLP
{
private:
    // Weights, biases, gradients, and velocities (all float)
    Matrix W1, vW1, vW1_hat, W2, vW2, vW2_hat, sW1, sW1_hat, sW2, sW2_hat,W3, vW3, vW3_hat, sW3, sW3_hat;
    Matrix dW1, dW2,dW3;
    Matrix b1, vb1, vb1_hat, sb1, sb1_hat, b2, vb2, vb2_hat, sb2, sb2_hat, b3, vb3, vb3_hat, sb3, sb3_hat;
    Matrix db1, db2,db3;

    // Activations and intermediates
    Matrix Z1, A1, dZ1;
    Matrix Z2, A2, dZ2;
    Matrix Z3, A3, dZ3;
    float hiddensize2;

public:
    // Constructor
    MLP(size_t input_size, size_t hidden_size,size_t hidden_size2, size_t output_size);

    // Core operations
    Matrix forward_prop(const Matrix &X);
    void back_prop(const Matrix &X, const Matrix &Y);
    void update_params(float learning_rate);
    float compute_loss(const Matrix &Y, const Matrix &A2);

   // Accessors
const Matrix& get_A1() const { return A1; }
const Matrix& get_A2() const { return A2; }
const Matrix& get_A3() const { return A3; }
const Matrix& get_Z1() const { return Z1; }
const Matrix& get_Z2() const { return Z2; }
const Matrix& get_Z3() const { return Z3; }
const Matrix& get_dZ1() const { return dZ1; }
const Matrix& get_dZ2() const { return dZ2; }
const Matrix& get_dZ3() const { return dZ3; }
const Matrix& get_dW1() const { return dW1; }
const Matrix& get_db1() const { return db1; }
const Matrix& get_dW2() const { return dW2; }
const Matrix& get_db2() const { return db2; }
const Matrix& get_dW3() const { return dW3; }
const Matrix& get_db3() const { return db3; }
const Matrix& get_W1() const { return W1; }
const Matrix& get_b1() const { return b1; }
const Matrix& get_W2() const { return W2; }
const Matrix& get_b2() const { return b2; }
const Matrix& get_W3() const { return W3; }
const Matrix& get_b3() const { return b3; }
const Matrix& get_vW1() const { return vW1; }
const Matrix& get_vb1() const { return vb1; }
const Matrix& get_vW2() const { return vW2; }
const Matrix& get_vb2() const { return vb2; }
const Matrix& get_vW3() const { return vW3; }
const Matrix& get_vb3() const { return vb3; }
const Matrix& get_sW1() const { return sW1; }
const Matrix& get_sb1() const { return sb1; }
const Matrix& get_sW2() const { return sW2; }
const Matrix& get_sb2() const { return sb2; }
const Matrix& get_sW3() const { return sW3; }
const Matrix& get_sb3() const { return sb3; }

// Setters for testing and parameter updates
void set_W1(const Matrix& weights) { W1 = weights; }
void set_W2(const Matrix& weights) { W2 = weights; }
void set_W3(const Matrix& weights) { W3 = weights; }
void set_b1(const Matrix& bias) { b1 = bias; }
void set_b2(const Matrix& bias) { b2 = bias; }
void set_b3(const Matrix& bias) { b3 = bias; }
void set_dW1(const Matrix& m) { dW1 = m; }
void set_db1(const Matrix& m) { db1 = m; }
void set_dW2(const Matrix& m) { dW2 = m; }
void set_db2(const Matrix& m) { db2 = m; }
void set_dW3(const Matrix& m) { dW3 = m; }
void set_db3(const Matrix& m) { db3 = m; }
void set_dZ1(const Matrix& m) { dZ1 = m; }
void set_dZ2(const Matrix& m) { dZ2 = m; }
void set_dZ3(const Matrix& m) { dZ3 = m; }
void set_vW1(const Matrix& m) { vW1 = m; }
void set_vb1(const Matrix& m) { vb1 = m; }
void set_vW2(const Matrix& m) { vW2 = m; }
void set_vb2(const Matrix& m) { vb2 = m; }
void set_vW3(const Matrix& m) { vW3 = m; }
void set_vb3(const Matrix& m) { vb3 = m; }
void set_sW1(const Matrix& m) { sW1 = m; }
void set_sb1(const Matrix& m) { sb1 = m; }
void set_sW2(const Matrix& m) { sW2 = m; }
void set_sb2(const Matrix& m) { sb2 = m; }
void set_sW3(const Matrix& m) { sW3 = m; }
void set_sb3(const Matrix& m) { sb3 = m; }


    // Utility functions
    Matrix one_hot(const Matrix &Y, size_t num_classes = 2);
    Matrix get_predictions(const Matrix &A);
    float get_accuracy(const Matrix &predictions, const Matrix &labels);
    float cross_entropy_loss(const Matrix &Y_pred, const Matrix &Y_true);

    // Training loop
    void gradient_descent(Matrix &X, Matrix &Y,
                          size_t iterations, float learning_rate);

// inline void update_layer_params(Matrix& param, Matrix& v, Matrix& s, 
//                                float lr_corrected, float epsilon, Error* err);

#ifdef UNIT_TESTING
    friend class MLPTest; // Grant test access
      void test_update_layer_params(Matrix& param, Matrix& v, Matrix& s, 
                                 float lr_corrected, float epsilon, Error* err) {
         void update_layer_params(param, v, s, lr_corrected, epsilon, err);
    }
      void update_layer_params(param, v, s, lr_corrected, epsilon, err);
#endif
};

#endif // MLP_H
