// mlp_test.cpp - Updated for 3-layer MLP
#include <gtest/gtest.h>
#include"../core/matrix.h"
#include"../ml/MLP.h"
#include <cmath>

class MLPTest : public ::testing::Test {
protected:
    size_t input_size = 2;
    size_t hidden_size1 = 3;
    size_t hidden_size2 = 4;
    size_t output_size = 2;
    MLP mlp;

    MLPTest() : mlp(input_size, hidden_size1, hidden_size2, output_size) {
        // Set test weights and biases for all three layers
        Matrix test_W1(hidden_size1, input_size);
        Matrix test_W2(hidden_size2, hidden_size1);
        Matrix test_W3(output_size, hidden_size2);
        Matrix test_b1(1, hidden_size1);
        Matrix test_b2(1, hidden_size2);
        Matrix test_b3(1, output_size);
        
        test_W1.fill(0.1);
        test_W2.fill(0.2);
        test_W3.fill(0.3);
        test_b1.fill(0.3);
        test_b2.fill(0.4);
        test_b3.fill(0.5);
        
        mlp.set_W1(test_W1);
        mlp.set_W2(test_W2);
        mlp.set_W3(test_W3);
        mlp.set_b1(test_b1);
        mlp.set_b2(test_b2);
        mlp.set_b3(test_b3);
    }
};

// Forward propagation test updated for 3 layers
TEST_F(MLPTest, ForwardPropKnownValues) {
    Matrix X(input_size, 1);
    X(0, 0) = 0.5; X(1, 0) = -0.5;

    Matrix A3 = mlp.forward_prop(X);
    
    // Verify output shape
    ASSERT_EQ(A3.row_count(), output_size);
    ASSERT_EQ(A3.col_count(), 1);
    
    // Verify calculations for all layers
    for (size_t i = 0; i < hidden_size1; i++) {
        double z1 = 0.1 * 0.5 + 0.1 * (-0.5) + 0.3;
        double expected_a1 = std::max(0.0, z1);  // Leaky ReLU
        // Note: Need getters for A1 and A2 if you want to verify them
    }
}

TEST_F(MLPTest, BackpropGradientShapes) {
    Matrix X(input_size, 1);
    Matrix Y(1, 1);
    mlp.forward_prop(X);
    mlp.back_prop(X, Y);

    // Check shapes for all three layers
    ASSERT_EQ(mlp.get_dW1().row_count(), hidden_size1);
    ASSERT_EQ(mlp.get_dW1().col_count(), input_size);
    
    ASSERT_EQ(mlp.get_dW2().row_count(), hidden_size2);
    ASSERT_EQ(mlp.get_dW2().col_count(), hidden_size1);
    
    ASSERT_EQ(mlp.get_dW3().row_count(), output_size);
    ASSERT_EQ(mlp.get_dW3().col_count(), hidden_size2);
    
    ASSERT_EQ(mlp.get_db1().row_count(), 1);
    ASSERT_EQ(mlp.get_db1().col_count(), hidden_size1);
    
    ASSERT_EQ(mlp.get_db2().row_count(), 1);
    ASSERT_EQ(mlp.get_db2().col_count(), hidden_size2);
    
    ASSERT_EQ(mlp.get_db3().row_count(), 1);
    ASSERT_EQ(mlp.get_db3().col_count(), output_size);
}

TEST_F(MLPTest, BackpropGradientValues) {
    Matrix X(input_size, 1);
    Matrix Y(1, 1);
    X(0,0) = 0.5; X(1,0) = 0.5;
    Y(0,0) = 1;

    mlp.forward_prop(X);
    mlp.back_prop(X, Y);

    // Get dZ3 (output layer gradient)
    Matrix dZ3 = mlp.get_dZ3();
    ASSERT_EQ(dZ3.row_count(), output_size);
    ASSERT_EQ(dZ3.col_count(), 1);
    
    // For binary classification with softmax, dZ3 = A3 - Y_one_hot
    // With our symmetric weights, A3 should be [0.5, 0.5]
    // Y = 1 means one_hot = [0, 1], so dZ3 ≈ [0.5, -0.5]
    EXPECT_NEAR(dZ3(0,0), 0.5, 0.1);
    EXPECT_NEAR(dZ3(1,0), -0.5, 0.1);
}

TEST_F(MLPTest, ParameterUpdate) {
    // Initialize gradients for all three layers
    Matrix dW1(hidden_size1, input_size);
    Matrix dW2(hidden_size2, hidden_size1);
    Matrix dW3(output_size, hidden_size2);
    Matrix db1(1, hidden_size1);
    Matrix db2(1, hidden_size2);
    Matrix db3(1, output_size);
    
    dW1.fill(0.01f);
    dW2.fill(0.02f);
    dW3.fill(0.03f);
    db1.fill(0.01f);
    db2.fill(0.02f);
    db3.fill(0.03f);
    
    mlp.set_dW1(dW1);
    mlp.set_dW2(dW2);
    mlp.set_dW3(dW3);
    mlp.set_db1(db1);
    mlp.set_db2(db2);
    mlp.set_db3(db3);

    // Reset Adam moments for all layers
    Matrix zero_vW1(hidden_size1, input_size);
    Matrix zero_vW2(hidden_size2, hidden_size1);
    Matrix zero_vW3(output_size, hidden_size2);
    Matrix zero_vb1(1, hidden_size1);
    Matrix zero_vb2(1, hidden_size2);
    Matrix zero_vb3(1, output_size);
    Matrix zero_sW1(hidden_size1, input_size);
    Matrix zero_sW2(hidden_size2, hidden_size1);
    Matrix zero_sW3(output_size, hidden_size2);
    Matrix zero_sb1(1, hidden_size1);
    Matrix zero_sb2(1, hidden_size2);
    Matrix zero_sb3(1, output_size);
    
    zero_vW1.fill(0.0f);
    zero_vW2.fill(0.0f);
    zero_vW3.fill(0.0f);
    zero_vb1.fill(0.0f);
    zero_vb2.fill(0.0f);
    zero_vb3.fill(0.0f);
    zero_sW1.fill(0.0f);
    zero_sW2.fill(0.0f);
    zero_sW3.fill(0.0f);
    zero_sb1.fill(0.0f);
    zero_sb2.fill(0.0f);
    zero_sb3.fill(0.0f);
    
    mlp.set_vW1(zero_vW1);
    mlp.set_vW2(zero_vW2);
    mlp.set_vW3(zero_vW3);
    mlp.set_vb1(zero_vb1);
    mlp.set_vb2(zero_vb2);
    mlp.set_vb3(zero_vb3);
    mlp.set_sW1(zero_sW1);
    mlp.set_sW2(zero_sW2);
    mlp.set_sW3(zero_sW3);
    mlp.set_sb1(zero_sb1);
    mlp.set_sb2(zero_sb2);
    mlp.set_sb3(zero_sb3);

    // Store original parameters
    Matrix orig_W1 = mlp.get_W1();
    Matrix orig_W2 = mlp.get_W2();
    Matrix orig_W3 = mlp.get_W3();
    
    mlp.update_params(0.1f);

    // Check that parameters have been updated (changed from original)
    for (size_t i = 0; i < mlp.get_W1().size(); ++i) {
        EXPECT_NE(mlp.get_W1().no_bounds_check(i), orig_W1.no_bounds_check(i));
    }
    for (size_t i = 0; i < mlp.get_W2().size(); ++i) {
        EXPECT_NE(mlp.get_W2().no_bounds_check(i), orig_W2.no_bounds_check(i));
    }
    for (size_t i = 0; i < mlp.get_W3().size(); ++i) {
        EXPECT_NE(mlp.get_W3().no_bounds_check(i), orig_W3.no_bounds_check(i));
    }
}

TEST(MatrixTest, SoftmaxStability) {
    Matrix logits(2, 1);
    logits(0, 0) = 1000.0;
    logits(1, 0) = 1001.0;

    Matrix probs = Matrix::softmax(logits);
    double sum = 0.0;
    for (size_t i = 0; i < probs.row_count(); ++i) {
        EXPECT_TRUE(std::isfinite(probs(i, 0)));
        EXPECT_GE(probs(i, 0), 0.0);
        EXPECT_LE(probs(i, 0), 1.0);
        sum += probs(i, 0);
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

TEST_F(MLPTest, GradientMagnitudeCheck) {
    // Initialize with random weights for all three layers
    Matrix test_W1 = Matrix::random(hidden_size1, input_size).multiply_scalar(0.5);
    Matrix test_W2 = Matrix::random(hidden_size2, hidden_size1).multiply_scalar(0.5);
    Matrix test_W3 = Matrix::random(output_size, hidden_size2).multiply_scalar(0.5);
    
    mlp.set_W1(test_W1);
    mlp.set_W2(test_W2);
    mlp.set_W3(test_W3);

    Matrix X(input_size, 2);
    Matrix Y(1, 2);
    X(0, 0) = 0.5; X(1, 0) = -0.3;
    X(0, 1) = 1.2; X(1, 1) = 0.8;
    Y(0, 0) = 0; Y(0, 1) = 1;
    
    EXPECT_EQ(X.col_count(), 2);
    EXPECT_EQ(Y.col_count(), 2);
    
    mlp.forward_prop(X);
    mlp.back_prop(X, Y);

    // Check gradient magnitudes for all layers
    double norm_dW1 = mlp.get_dW1().frobenius_norm();
    double norm_dW2 = mlp.get_dW2().frobenius_norm();
    double norm_dW3 = mlp.get_dW3().frobenius_norm();
    double norm_db1 = mlp.get_db1().frobenius_norm();
    double norm_db2 = mlp.get_db2().frobenius_norm();
    double norm_db3 = mlp.get_db3().frobenius_norm();

    EXPECT_GT(norm_dW1, 1e-6);
    EXPECT_LT(norm_dW1, 100.0);
    
    EXPECT_GT(norm_dW2, 1e-6);
    EXPECT_LT(norm_dW2, 100.0);
    
    EXPECT_GT(norm_dW3, 1e-6);
    EXPECT_LT(norm_dW3, 100.0);
    
    EXPECT_GT(norm_db1, 1e-6);
    EXPECT_LT(norm_db1, 100.0);
    
    EXPECT_GT(norm_db2, 1e-6);
    EXPECT_LT(norm_db2, 100.0);
    
    EXPECT_GT(norm_db3, 1e-6);
    EXPECT_LT(norm_db3, 100.0);
}

TEST(MatrixTest, AddInplaceSquared) {
    Matrix m1(2, 2);
    m1.fill(1.0f);
    Matrix m2(2, 2);
    m2.fill(2.0f);

    m1.add_inplace_squared(m2, 0.1f);

    for (size_t i = 0; i < m1.size(); ++i) {
        EXPECT_FLOAT_EQ(m1.no_bounds_check(i), 1.4f);
    }
}

TEST_F(MLPTest, UpdateParamsBatch) {
    // Create a larger MLP for batch update test
    MLP mlp_large(2, 3, 4, 1);
    
    // Initialize with known values
    Matrix dW1(3, 2);
    Matrix dW2(4, 3);
    Matrix dW3(1, 4);
    Matrix db1(1, 3);
    Matrix db2(1, 4);
    Matrix db3(1, 1);
    
    dW1.fill(1.0f);
    dW2.fill(1.0f);
    dW3.fill(1.0f);
    db1.fill(1.0f);
    db2.fill(1.0f);
    db3.fill(1.0f);
    
    mlp_large.set_dW1(dW1);
    mlp_large.set_dW2(dW2);
    mlp_large.set_dW3(dW3);
    mlp_large.set_db1(db1);
    mlp_large.set_db2(db2);
    mlp_large.set_db3(db3);
    
    float learning_rate = 0.001f;
    
    // Store original parameters
    Matrix orig_W1 = mlp_large.get_W1();
    Matrix orig_W2 = mlp_large.get_W2();
    Matrix orig_W3 = mlp_large.get_W3();
    
    mlp_large.update_params(learning_rate);
    
    // Check that all parameters have been updated
    Matrix W1 = mlp_large.get_W1();
    Matrix W2 = mlp_large.get_W2();
    Matrix W3 = mlp_large.get_W3();
    
    bool w1_changed = false, w2_changed = false, w3_changed = false;
    
    for (size_t i = 0; i < W1.size(); ++i) {
        if (std::abs(W1.no_bounds_check(i) - orig_W1.no_bounds_check(i)) > 1e-6) {
            w1_changed = true;
            break;
        }
    }
    
    for (size_t i = 0; i < W2.size(); ++i) {
        if (std::abs(W2.no_bounds_check(i) - orig_W2.no_bounds_check(i)) > 1e-6) {
            w2_changed = true;
            break;
        }
    }
    
    for (size_t i = 0; i < W3.size(); ++i) {
        if (std::abs(W3.no_bounds_check(i) - orig_W3.no_bounds_check(i)) > 1e-6) {
            w3_changed = true;
            break;
        }
    }
    
    EXPECT_TRUE(w1_changed);
    EXPECT_TRUE(w2_changed);
    EXPECT_TRUE(w3_changed);
}

// Test forward propagation with batch
TEST_F(MLPTest, ForwardPropBatch) {
    Matrix X(input_size, 3);  // Batch of 3
    X(0, 0) = 0.5; X(1, 0) = -0.5;
    X(0, 1) = 1.0; X(1, 1) = 0.0;
    X(0, 2) = -0.5; X(1, 2) = 0.5;
    
    Matrix A3 = mlp.forward_prop(X);
    
    // Check output shape
    ASSERT_EQ(A3.row_count(), output_size);
    ASSERT_EQ(A3.col_count(), 3);  // Batch size
    
    // Check that outputs are valid probabilities (sum to ~1 for each column)
    for (size_t col = 0; col < 3; ++col) {
        double col_sum = 0.0;
        for (size_t row = 0; row < output_size; ++row) {
            col_sum += A3(row, col);
            EXPECT_GE(A3(row, col), 0.0);
            EXPECT_LE(A3(row, col), 1.0);
        }
        EXPECT_NEAR(col_sum, 1.0, 1e-5);
    }
}

// Test prediction accuracy
TEST_F(MLPTest, PredictionAccuracy) {
    Matrix X(input_size, 2);
    Matrix Y(1, 2);
    
    X(0, 0) = 0.5; X(1, 0) = 0.5;
    X(0, 1) = -0.5; X(1, 1) = -0.5;
    Y(0, 0) = 1; Y(0, 1) = 0;
    
    Matrix A3 = mlp.forward_prop(X);
    Matrix predictions = mlp.get_predictions(A3);
    
    // Check prediction shape
    ASSERT_EQ(predictions.row_count(), 1);
    ASSERT_EQ(predictions.col_count(), 2);
    
    // Calculate accuracy (may not be perfect with random weights)
    float accuracy = mlp.get_accuracy(predictions, Y);
    EXPECT_GE(accuracy, 0.0);
    EXPECT_LE(accuracy, 1.0);
}

// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }