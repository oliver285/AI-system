// mlp_test.cpp
#include <gtest/gtest.h>
#include"../core/matrix.h"
#include"../ml/MLP.h"
// #include "MLP.h"
// #include "matrix.h"
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

    // Verify calculations account for proper bias handling
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

// mlp_test.cpp - Update BackpropGradientValues test
TEST_F(MLPTest, BackpropGradientValues) {
    Matrix X(input_size, 1);
    Matrix Y(1, 1);
    X(0,0) = 0.5; X(1,0) = 0.5;  // Explicit values
    Y(0,0) = 1;

    mlp.forward_prop(X);
    mlp.back_prop(X, Y);

    // Manually computed expected values
    Matrix expected_dZ2(output_size, 1);
    expected_dZ2(0,0) = 0.5;    // Expected gradient for class 0
    expected_dZ2(1,0) = -0.5;   // Expected gradient for class 1

    Matrix checkDZ2 = mlp.get_dZ2();
    for (size_t i = 0; i < checkDZ2.row_count(); ++i) {
        for (size_t j = 0; j < checkDZ2.col_count(); ++j) {
            ASSERT_NEAR(checkDZ2(i, j), expected_dZ2(i, j), 1e-6);
        }
    }
}TEST_F(MLPTest, ParameterUpdate) {
    // Initialize with known values
    Matrix dW1(hidden_size, input_size);
    Matrix dW2(output_size, hidden_size);
    Matrix db1(1, hidden_size);
    Matrix db2(1, output_size);
    
    dW1.fill(0.01f);
    db1.fill(0.02f);
    dW2.fill(0.03f);
    db2.fill(0.04f);
    
    mlp.set_dW1(dW1);
    mlp.set_db1(db1);
    mlp.set_dW2(dW2);
    mlp.set_db2(db2);

    // Reset Adam moments to zero with correct dimensions for each
    Matrix zero_vW1(hidden_size, input_size);
    Matrix zero_vW2(output_size, hidden_size);
    Matrix zero_vb1(1, hidden_size);
    Matrix zero_vb2(1, output_size);
    Matrix zero_sW1(hidden_size, input_size);
    Matrix zero_sW2(output_size, hidden_size);
    Matrix zero_sb1(1, hidden_size);
    Matrix zero_sb2(1, output_size);
    
    zero_vW1.fill(0.0f);
    zero_vW2.fill(0.0f);
    zero_vb1.fill(0.0f);
    zero_vb2.fill(0.0f);
    zero_sW1.fill(0.0f);
    zero_sW2.fill(0.0f);
    zero_sb1.fill(0.0f);
    zero_sb2.fill(0.0f);
    
    mlp.set_vW1(zero_vW1);
    mlp.set_vW2(zero_vW2);
    mlp.set_vb1(zero_vb1);
    mlp.set_vb2(zero_vb2);
    mlp.set_sW1(zero_sW1);
    mlp.set_sW2(zero_sW2);
    mlp.set_sb1(zero_sb1);
    mlp.set_sb2(zero_sb2);

    Matrix orig_W1 = mlp.get_W1();
    mlp.update_params(0.1f);

    // For Adam with zero-initialized moments, the first update should be:
    // v = (1 - beta1) * gradient
    // s = (1 - beta2) * gradient^2
    // param = param - lr * v / (sqrt(s) + epsilon)
    for (size_t i = 0; i < mlp.get_W1().size(); ++i) {
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
        
        float v = (1 - beta1) * 0.01f;  // v = 0.1 * 0.01 = 0.001
        float s = (1 - beta2) * 0.01f * 0.01f;  // s = 0.001 * 0.0001 = 1e-7
        
        float expected_change = -0.1f * v / (sqrtf(s) + epsilon);
        float actual_change = mlp.get_W1().no_bounds_check(i) - orig_W1.no_bounds_check(i);
        
        ASSERT_NEAR(actual_change, expected_change, 1e-6f);
    }
}

// Add to your mlp_test.cpp

TEST(MatrixTest, SoftmaxStability) {
    Matrix logits(2, 1);
    logits(0, 0) = 1000.0;
    logits(1, 0) = 1001.0;

    Matrix probs = Matrix::softmax(logits);
    double sum = 0.0;
    for (size_t i = 0; i < probs.row_count(); ++i) {
        ASSERT_TRUE(std::isfinite(probs(i, 0)));
        ASSERT_GE(probs(i, 0), 0.0);
        ASSERT_LE(probs(i, 0), 1.0);
        sum += probs(i, 0);
    }
    ASSERT_NEAR(sum, 1.0, 1e-6);
}


TEST_F(MLPTest, GradientMagnitudeCheck) {



    Matrix test_W1=Matrix::random(hidden_size,input_size).multiply_scalar(.5);
    Matrix test_W2=Matrix::random(output_size,hidden_size).multiply_scalar(.5);;
    mlp.set_W1(test_W1);
    mlp.set_W2(test_W2);

    Matrix X(input_size, 2);
    Matrix Y(1, 2);
    X(0, 0) = 0.5; X(1, 0) = -0.3;  // ← some variance
    X(0, 1) = 1.2; X(1, 1) =  0.8;
    Y(0, 0) = 0; Y(0, 1) = 1; 
    
    EXPECT_EQ(X.col_count(), 2);
    EXPECT_EQ(Y.col_count(), 2);
    

    mlp.forward_prop(X);
    mlp.back_prop(X, Y);

    double norm_dW1 = mlp.get_dW1().frobenius_norm();
    double norm_dW2 = mlp.get_dW2().frobenius_norm();
    double norm_db1 = mlp.get_db1().frobenius_norm();
    double norm_db2 = mlp.get_db2().frobenius_norm();

    ASSERT_GT(norm_dW1, 1e-6);
    ASSERT_LT(norm_dW1, 100.0);

    ASSERT_GT(norm_dW2, 1e-6);
    ASSERT_LT(norm_dW2, 100.0);

    ASSERT_GT(norm_db1, 1e-6);
    ASSERT_LT(norm_db1, 100.0);

    ASSERT_GT(norm_db2, 1e-6);
    ASSERT_LT(norm_db2, 100.0);
}


TEST(MatrixTest, AddInplaceSquared) {
    Matrix m1(2, 2);
    m1.fill(1.0f); // Initialize with 1.0
    Matrix m2(2, 2);
    m2.fill(2.0f); // Initialize with 2.0

    // Test add_inplace_squared with alpha = 0.1
    m1.add_inplace_squared(m2, 0.1f);

    // Expected: m1[i] += 0.1 * (2.0 * 2.0) = 1.0 + 0.4 = 1.4
    for (size_t i = 0; i < m1.size(); ++i) {
        EXPECT_FLOAT_EQ(m1.no_bounds_check(i), 1.4f);
    }
}


// TEST_F(MLPTest, UpdateLayerParams) {
//     // Initialize matrices
//     Matrix param(2, 2);
//     param.fill(2.0f);
//     Matrix v(2, 2);
//     v.fill(1.0f);
//     Matrix s(2, 2);
//     s.fill(4.0f); // sqrt(4) = 2.0
//     float lr_corrected = 0.1f;
//     float epsilon = 1e-8f;
//     Error err = NO_ERROR;

//     // Call the function (ensure it's accessible, might need to be public or friend)
//     update_layer_params(param, v, s, lr_corrected, epsilon, &err);

//     // Check for errors
//     ASSERT_EQ(err, NO_ERROR);

//     // Verify parameter update:
//     // s becomes sqrt(4) + epsilon ≈ 2.00000001
//     // v becomes v / s ≈ 1.0 / 2.00000001 ≈ 0.5
//     // Then param = param - lr_corrected * v ≈ 2.0 - 0.1 * 0.5 = 1.95
//     for (size_t i = 0; i < param.size(); ++i) {
//         EXPECT_NEAR(param.no_bounds_check(i), 1.95f, 1e-6f);
//     }
// }



TEST_F(MLPTest, UpdateParams) {
    MLP mlp(2,3,1); // Create MLP object

    // Initialize parameters and gradients to known values for testing
    // mlp.W1.fill(2.0f);
    // mlp.W2.fill(2.0f);
    // mlp.b1.fill(2.0f);
    // mlp.b2.fill(2.0f);

    // mlp.dW1.fill(1.0f); // Set gradients to 1.0
    // mlp.dW2.fill(1.0f);
    // mlp.db1.fill(1.0f);
    // mlp.db2.fill(1.0f);

    // // Initialize moments to zero (as they would be at start)
    // mlp.vW1.fill(0.0f);
    // mlp.vW2.fill(0.0f);
    // mlp.vb1.fill(0.0f);
    // mlp.vb2.fill(0.0f);
    // mlp.sW1.fill(0.0f);
    // mlp.sW2.fill(0.0f);
    // mlp.sb1.fill(0.0f);
    // mlp.sb2.fill(0.0f);

    float learning_rate = 0.001f;
    mlp.update_params(learning_rate);
Matrix W1 = mlp.get_W1();
    // After first update, parameters should decrease due to positive gradients
    // Exact values depend on Adam formulas, but we can check for change
    for (size_t i = 0; i < W1.size(); ++i) {
        EXPECT_LT(W1.no_bounds_check(i), 2.0f); // Should be less than initial
    }
    // Similarly for W2, b1, b2
}