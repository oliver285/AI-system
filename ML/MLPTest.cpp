// mlp_test.cpp
#include <gtest/gtest.h>
#include "MLP.h"
#include "matrix.h"
#include <cmath>  // For log()

class MLPTest : public ::testing::Test {
protected:
    size_t input_size = 2;
    size_t hidden_size = 3;
    size_t output_size = 2;
    MLP mlp;


    MLPTest() : mlp(input_size, hidden_size, output_size) {
        Matrix test_W1(hidden_size,input_size);
Matrix test_W2(output_size,hidden_size);
Matrix test_b1(1,hidden_size);
Matrix test_b2(1,output_size);
test_W1.fill(0.1);
test_W2.fill(0.2);
test_b1.fill(0.3);
test_b2.fill(0.4);
mlp.set_W1(test_W1);
mlp.set_W2(test_W2);
mlp.set_b1(test_b1);
mlp.set_b2(test_b2);

        // mlp.W1.fill(0.1);
        // mlp.W2.fill(0.2);
        // mlp.b1.fill(0.3);
        // mlp.b2.fill(0.4);

        // : W1(hidden_size, input_size),   // Correct: (hidden_size, input_size)
        // W2(output_size, hidden_size),  // Correct: (output_size, hidden_size)
        // b1(1, hidden_size),            // Changed to (1, hidden_size)
        // b2(1, output_size), 

    }
};
// Updated test methods only - replace existing ones with these

TEST_F(MLPTest, ForwardPropKnownValues) {
    Matrix X(input_size, 1);
    X(0, 0) = 0.5; X(1, 0) = -0.5;

    mlp.forward_prop(X);

    for (size_t i = 0; i < hidden_size; i++) {
        double z = 0.1 * 0.5 + 0.1 * (-0.5) + 0.3;
        double expected = std::max(0.0, z);
        ASSERT_NEAR(mlp.get_A1()(i, 0), expected, 1e-6);
    }
}

TEST_F(MLPTest, BackpropGradientShapes) {
    Matrix X(input_size, 1);
    Matrix Y(1, 1);
    mlp.forward_prop(X);
    mlp.back_prop(X, Y);

    ASSERT_EQ(mlp.get_dW1().row_count(), hidden_size);
    ASSERT_EQ(mlp.get_dW1().col_count(), input_size);
    ASSERT_EQ(mlp.get_db1().row_count(), 1);
    ASSERT_EQ(mlp.get_db1().col_count(), hidden_size);
    ASSERT_EQ(mlp.get_dW2().row_count(), output_size);
    ASSERT_EQ(mlp.get_dW2().col_count(), hidden_size);
    ASSERT_EQ(mlp.get_db2().row_count(), 1);
    ASSERT_EQ(mlp.get_db2().col_count(), output_size);
}

TEST_F(MLPTest, BackpropGradientValues) {
    Matrix X(input_size, 1);
    Matrix Y(1, 1);
    X.fill(0.5);
    Y(0, 0) = 1;

    mlp.forward_prop(X);
    mlp.back_prop(X, Y);

    Matrix probs = mlp.get_A2();
    // Add dimension check
    ASSERT_EQ(probs.row_count(), output_size);
    ASSERT_EQ(probs.col_count(), 1);

    Matrix expected_dZ2 = probs;
    
    // Add safety check
    size_t label_index = static_cast<size_t>(Y(0, 0));
    if (label_index < expected_dZ2.row_count()) {
        expected_dZ2(label_index, 0) -= 1.0;
    } else {
        FAIL() << "Label index out of range: " << label_index;
    }

    for (size_t i = 0; i < output_size; i++) {
        ASSERT_NEAR(mlp.get_dZ2()(i, 0), expected_dZ2(i, 0), 1e-6);
    }
}

TEST_F(MLPTest, ParameterUpdate) {
    // Create matrices with correct dimensions

    Matrix dW1(hidden_size,input_size);
    Matrix dW2(output_size,hidden_size);
    Matrix db1(1,hidden_size);
    Matrix db2(1,output_size);
    // Matrix dW1(mlp.get_dW1().dimensions());
    // Matrix db1(mlp.get_db1().dimensions());
    // Matrix dW2(mlp.get_dW2().dimensions());
    // Matrix db2(mlp.get_db2().dimensions());
    
    dW1.fill(0.01);
    db1.fill(0.02);
    dW2.fill(0.03);
    db2.fill(0.04);
    
    mlp.set_dW1(dW1);
    mlp.set_db1(db1);
    mlp.set_dW2(dW2);
    mlp.set_db2(db2);

    // Save original parameters
    Matrix orig_W1 = mlp.get_W1();
    Matrix orig_b1 = mlp.get_b1();
    Matrix orig_W2 = mlp.get_W2();
    Matrix orig_b2 = mlp.get_b2();

    // Initialize velocities to zero
    Matrix zero_vW1(hidden_size,input_size);
    Matrix zero_vW2(output_size,hidden_size);
    Matrix zero_vb1(1,hidden_size);
    Matrix zero_vb2(1,output_size);
    // Matrix zero_vW1(mlp.get_vW1().dimensions());
    // Matrix zero_vb1(mlp.get_vb1().dimensions());
    // Matrix zero_vW2(mlp.get_vW2().dimensions());
    // Matrix zero_vb2(mlp.get_vb2().dimensions());
    
    zero_vW1.fill(0.0);
    zero_vb1.fill(0.0);
    zero_vW2.fill(0.0);
    zero_vb2.fill(0.0);
    
    mlp.set_vW1(zero_vW1);
    mlp.set_vb1(zero_vb1);
    mlp.set_vW2(zero_vW2);
    mlp.set_vb2(zero_vb2);
    
    mlp.update_params(0.1);

    // Check W1 update
    for (size_t i = 0; i < mlp.get_W1().size(); ++i) {
        double expected = orig_W1.no_bounds_check(i) - 0.1 * 0.01;
        ASSERT_NEAR(mlp.get_W1().no_bounds_check(i), expected, 1e-6);
    }
    
    // Check b1 update
    for (size_t i = 0; i < mlp.get_b1().size(); ++i) {
        double expected = orig_b1.no_bounds_check(i) - 0.1 * 0.02;
        ASSERT_NEAR(mlp.get_b1().no_bounds_check(i), expected, 1e-6);
    }
}