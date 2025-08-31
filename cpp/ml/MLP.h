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
    Matrix W1, vW1, vW1_hat, W2, vW2, vW2_hat, sW1, sW1_hat, sW2, sW2_hat;
    Matrix dW1, dW2;
    Matrix b1, vb1, vb1_hat, sb1, sb1_hat, b2, vb2, vb2_hat, sb2, sb2_hat;
    Matrix db1, db2;

    // Activations and intermediates
    Matrix Z1, A1, dZ1;
    Matrix Z2, A2, dZ2;

public:
    // Constructor
    MLP(size_t input_size, size_t hidden_size, size_t output_size);

    // Core operations
    Matrix forward_prop(const Matrix &X);
    void back_prop(const Matrix &X, const Matrix &Y);
    void update_params(float learning_rate);
    float compute_loss(const Matrix &Y, const Matrix &A2);

    // Accessors
    const Matrix &get_A1() const { return A1; }
    const Matrix &get_A2() const { return A2; }
    const Matrix &get_dZ2() const { return dZ2; }
    const Matrix &get_dW1() const { return dW1; }
    const Matrix &get_db1() const { return db1; }
    const Matrix &get_dW2() const { return dW2; }
    const Matrix &get_db2() const { return db2; }
    const Matrix &get_W1() const { return W1; }
    const Matrix &get_b1() const { return b1; }
    const Matrix &get_W2() const { return W2; }
    const Matrix &get_b2() const { return b2; }
    const Matrix &get_vW1() const { return vW1; }
    const Matrix &get_vb1() const { return vb1; }
    const Matrix &get_vW2() const { return vW2; }
    const Matrix &get_vb2() const { return vb2; }

    // Setters for testing and parameter updates
    void set_W1(const Matrix &weights) { W1 = weights; }
    void set_W2(const Matrix &weights) { W2 = weights; }
    void set_b1(const Matrix &bias) { b1 = bias; }
    void set_b2(const Matrix &bias) { b2 = bias; }
    void set_dW1(const Matrix &m) { dW1 = m; }
    void set_db1(const Matrix &m) { db1 = m; }
    void set_dW2(const Matrix &m) { dW2 = m; }
    void set_db2(const Matrix &m) { db2 = m; }
    void set_vW1(const Matrix &m) { vW1 = m; }
    void set_vb1(const Matrix &m) { vb1 = m; }
    void set_vW2(const Matrix &m) { vW2 = m; }
    void set_vb2(const Matrix &m) { vb2 = m; }
    void set_sW1(const Matrix &m) { sW1 = m; }
    void set_sW2(const Matrix &m) { sW2 = m; }
    void set_sb1(const Matrix &m) { sb1 = m; }
    void set_sb2(const Matrix &m) { sb2 = m; };


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
