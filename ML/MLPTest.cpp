// mlp_test.cpp
#include <gtest/gtest.h>
#include "MLP.h"
#include "matrix.h"

class MLPTest : public ::testing::Test {
protected:
    size_t input_size = 2;
    size_t hidden_size = 3;
    size_t output_size = 2;
    MLP mlp;

    MLPTest() : mlp(input_size, hidden_size, output_size) {
        mlp.W1.fill(0.1);
        mlp.W2.fill(0.2);
        mlp.b1.fill(0.3);
        mlp.b2.fill(0.4);
    }
};

TEST_F(MLPTest, ForwardPropSanityCheck) {
    Matrix X(input_size, 2);
    X.fill(1.0);

    Matrix output = mlp.forward_prop(X);

    for (size_t i = 0; i < output.col_count(); i++) {
        double sum = 0.0;
        for (size_t j = 0; j < output.row_count(); j++) {
            double val = output(j, i);
            sum += val;
            ASSERT_GE(val, 0.0);
            ASSERT_LE(val, 1.0);
        }
        ASSERT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST_F(MLPTest, ForwardPropKnownValues) {
    Matrix X(input_size, 1);
    X(0, 0) = 0.5; X(1, 0) = -0.5;

    mlp.forward_prop(X);

    for (size_t i = 0; i < hidden_size; i++) {
        double z = 0.1 * 0.5 + 0.1 * (-0.5) + 0.3;
        double expected = std::max(0.0, z);
        ASSERT_NEAR(mlp.A1(i, 0), expected, 1e-6);
    }
}

TEST_F(MLPTest, BackpropGradientShapes) {
    Matrix X(input_size, 1);
    Matrix Y(1, 1);
    mlp.forward_prop(X);
    mlp.back_prop(X, Y);

    ASSERT_EQ(mlp.dW1.row_count(), hidden_size);
    ASSERT_EQ(mlp.dW1.col_count(), input_size);
    ASSERT_EQ(mlp.db1.row_count(), 1);
    ASSERT_EQ(mlp.db1.col_count(), hidden_size);
    ASSERT_EQ(mlp.dW2.row_count(), output_size);
    ASSERT_EQ(mlp.dW2.col_count(), hidden_size);
    ASSERT_EQ(mlp.db2.row_count(), 1);
    ASSERT_EQ(mlp.db2.col_count(), output_size);
}

TEST_F(MLPTest, BackpropGradientValues) {
    Matrix X(input_size, 1);
    Matrix Y(1, 1);
    X.fill(0.5);
    Y(0, 0) = 1;

    mlp.forward_prop(X);
    mlp.back_prop(X, Y);

    Matrix probs = mlp.A2;
    Matrix expected_dZ2 = probs;
    expected_dZ2(Y(0, 0), 0) -= 1.0;

    for (size_t i = 0; i < output_size; i++) {
        ASSERT_NEAR(mlp.dZ2(i, 0), expected_dZ2(i, 0), 1e-6);
    }
}

TEST_F(MLPTest, ParameterUpdate) {
    mlp.dW1.fill(0.01);
    mlp.db1.fill(0.02);
    mlp.dW2.fill(0.03);
    mlp.db2.fill(0.04);

    Matrix orig_W1 = mlp.W1;

    mlp.update_params(0.1);

    for (size_t i = 0; i < mlp.W1.size(); ++i) {
        double expected = orig_W1.no_bounds_check(i) - 0.1 * 0.01;
        ASSERT_NEAR(mlp.W1.no_bounds_check(i), expected, 1e-6);
    }
}

TEST_F(MLPTest, CrossEntropyLoss) {
    Matrix Y(1, 3);
    Y(0, 0) = 0; Y(0, 1) = 1; Y(0, 2) = 0;

    Matrix A2(output_size, 3);
    A2(0, 0) = 0.9; A2(1, 0) = 0.1;
    A2(0, 1) = 0.4; A2(1, 1) = 0.6;
    A2(0, 2) = 0.1; A2(1, 2) = 0.9;

    double loss = mlp.compute_loss(Y, A2);
    double expected = -(log(0.9) + log(0.6) + log(0.9)) / 3.0;
    ASSERT_NEAR(loss, expected, 1e-6);
}

TEST_F(MLPTest, OneHotEncoding) {
    Matrix Y(1, 3);
    Y(0, 0) = 0; Y(0, 1) = 2; Y(0, 2) = 1;

    Matrix one_hot = mlp.one_hot(Y, 3);

    ASSERT_EQ(one_hot.row_count(), 3);
    ASSERT_EQ(one_hot.col_count(), 3);
    ASSERT_EQ(one_hot(0, 0), 1.0);
    ASSERT_EQ(one_hot(1, 1), 1.0);
    ASSERT_EQ(one_hot(2, 2), 0.0);
}

TEST_F(MLPTest, PredictionAccuracy) {
    Matrix pred(1, 4);
    Matrix truth(1, 4);
    pred(0, 0) = 0; truth(0, 0) = 0;
    pred(0, 1) = 1; truth(0, 1) = 1;
    pred(0, 2) = 0; truth(0, 2) = 1;
    pred(0, 3) = 1; truth(0, 3) = 0;

    double acc = mlp.get_accuracy(pred, truth);
    ASSERT_NEAR(acc, 0.5, 1e-6);
}
