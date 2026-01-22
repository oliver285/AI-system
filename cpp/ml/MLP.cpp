
#include "MLP.h"
#include <iostream>
#include <iomanip>
#include <cmath>

// float t = 0; // timestep
MLP::MLP(size_t input_size, size_t hidden_size, size_t hidden_size2, size_t output_size)
    : W1(hidden_size, input_size),
      vW1(hidden_size, input_size),
      sW1(hidden_size, input_size),
      dW1(hidden_size, input_size),

      W2(hidden_size2, hidden_size),
      vW2(hidden_size2, hidden_size),
      sW2(hidden_size2, hidden_size),
      dW2(hidden_size2, hidden_size),

      W3(output_size, hidden_size2),
      vW3(output_size, hidden_size2),
      sW3(output_size, hidden_size2),
      dW3(output_size, hidden_size2),

      b1(1, hidden_size),
      vb1(1, hidden_size),
      sb1(1, hidden_size),
      db1(1, hidden_size),

      b2(1, hidden_size2),
      vb2(1, hidden_size2),
      sb2(1, hidden_size2),
      db2(1, hidden_size2),

      b3(1, output_size),
      vb3(1, output_size),
      sb3(1, output_size),
      db3(1, output_size)

{

    float scale1 = std::sqrt(2.0f / input_size);
    float scale2 = std::sqrt(2.0f / hidden_size);
    float scale3 = std::sqrt(2.0f / hidden_size2);

    W1 = Matrix::random(hidden_size, input_size).multiply_scalar(scale1);
    W2 = Matrix::random(hidden_size2, hidden_size).multiply_scalar(scale2);
    W3 = Matrix::random(output_size, hidden_size2).multiply_scalar(scale3);

    // Initialize biases to zero
    b1.fill(0.0f);
    b2.fill(0.0f);
    b3.fill(0.0f);

    // Initialize velocities and variances to zero
    vW1.fill(0.0f);
    vW2.fill(0.0f);
    vW3.fill(0.0f);
    sW1.fill(0.0f);
    sW2.fill(0.0f);
    sW3.fill(0.0f);
    vb1.fill(0.0f);
    vb2.fill(0.0f);
    vb3.fill(0.0f);
}

float MLP::compute_loss(const Matrix &Y, const Matrix &A2)
{
    Matrix one_hot_Y = one_hot(Y, A2.row_count());
    float loss = 0.0;
    for (size_t i = 0; i < A2.col_count(); ++i)
    {
        for (size_t j = 0; j < A2.row_count(); ++j)
        {
            float y = one_hot_Y(j, i);
            float a = A2(j, i);
            if (y > 0)
            {
                loss -= std::log(std::max(a, 1e-10f)); // avoid log(0)
            }
        }
    }
    return loss / A2.col_count(); // average over batch
}

bool checkError(Error err)
{
    if (err != NO_ERROR)
    {
        const char *error_msg = "Unknown error";
        switch (err)
        {
        case INDEX_OUT_OF_RANGE:
            error_msg = "Index out of range";
            break;
        case DIMENSION_MISMATCH:
            error_msg = "Dimension mismatch";
            break;
        case DIVIDE_BY_ZERO:
            error_msg = "Divide by zero";
            break;
        case NO_ERROR: // Not needed but for completeness
            return false;
        default:
            error_msg = "Unknown Error";
            break;
        }

        // For flight controllers, use proper logging instead of std::cerr
        // flight_log(LOG_CRITICAL, "Matrix error: %s", error_msg);

        std::cerr << "Matrix error: " << error_msg << "\n";
        return true;
    }
    return false;
}

Matrix MLP::forward_prop(const Matrix &X)
{
    size_t batch_size = X.col_count();
    if (batch_size == 0)
    {
        std::cerr << "Empty batch detected\n";
        return Matrix();
    }

    Error err = NO_ERROR;

    // ---- Layer 1 ----
    Z1 = Matrix::multiply(W1, X, &err);
    if (checkError(err))
        return Matrix();

    for (size_t j = 0; j < batch_size; ++j)
        for (size_t i = 0; i < Z1.row_count(); ++i)
            Z1(i, j) += b1(0, i, &err);

    A1 = Z1.leaky_RELU();

    // ---- Layer 2 ----
    Z2 = Matrix::multiply(W2, A1, &err);
    if (checkError(err))
        return Matrix();

    for (size_t j = 0; j < batch_size; ++j)
        for (size_t i = 0; i < Z2.row_count(); ++i)
            Z2(i, j) += b2(0, i, &err);

    A2 = Z2.leaky_RELU(); // FIX: use hidden activation, not softmax

    // ---- Layer 3 (Output) ----
    Z3 = Matrix::multiply(W3, A2, &err);
    if (checkError(err))
        return Matrix();

    for (size_t j = 0; j < batch_size; ++j)
        for (size_t i = 0; i < Z3.row_count(); ++i)
            Z3(i, j) += b3(0, i, &err);

    A3 = Matrix::softmax(Z3); // only final layer uses softmax
    return A3;
}

Matrix MLP::one_hot(const Matrix &Y, size_t num_classes)
{
    // Validate input dimensions
    if (Y.row_count() != 1)
    {
        return Matrix(); // Use consistent error handling
    }

    const size_t batch_size = Y.col_count();
    Matrix one_hot_Y(num_classes, batch_size);
    one_hot_Y.fill(0.0f);

    for (size_t i = 0; i < batch_size; ++i)
    {
        const float val = Y(0, i);

        // 1. Check if value is a valid integer
        if (std::abs(val - std::round(val)) > 1e-8)
        {
            return Matrix(); // Not an integer
        }

        const size_t class_idx = static_cast<size_t>(std::round(val));

        // 2. Validate class index range
        if (class_idx < 0.f || class_idx >= (num_classes))
        {
            return Matrix(); // Out of range
        }

        one_hot_Y(class_idx, i) = 1.0f;
    }

    return one_hot_Y;
}

// Matrix<2, 784>  deriv_ReLU() {
//     for (uint16_t i = 0; i < rows; ++i)
//         for (uint16_t j = 0; j < cols; ++j)
//             if (data[i][j] < 0)
//                 data[i][j] = 0;
// }

Matrix mean_over_batch(const Matrix &dZ)
{

    size_t rows = dZ.row_count();
    size_t batch_size = dZ.col_count();

    Matrix db(1, rows);

    for (size_t i = 0; i < rows; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < batch_size; ++j)
        {
            sum += dZ(i, j); // safe access
        }
        db(0, i) = sum / batch_size; // safe assignment
    }

    return db;
}

void MLP::back_prop(const Matrix &X, const Matrix &Y)
{
    Error err = NO_ERROR;
    size_t batch_size = X.col_count();

    // Validate input dimensions
    if (batch_size != Y.col_count())
    {
        std::cerr << "X and Y batch size mismatch" << std::endl;
        return;
    }

    // ---- Step 1: Output error ----
    Matrix one_hot_Y = one_hot(Y, A3.row_count()); // FIX: match output layer
    //  std::cout << "A3: " << A3.row_count() << "x" << A3.col_count()
    //           << ", one_hot_Y: " << one_hot_Y.row_count() << "x" << one_hot_Y.col_count() << "\n";
     dZ3 = A3.subtract(one_hot_Y, &err); // dZ3 = A3 - Y

    // ---- Step 2: Gradients for W3, b3 ----
    Matrix A2T = A2.transpose();
    dW3 = Matrix::multiply(dZ3, A2T, &err);
    dW3.scale_inplace(1.0f / batch_size);

    db3 = mean_over_batch(dZ3); // same loop you already wrote

    // ---- Step 3: Backprop to hidden layer 2 ----
    Matrix W3T = W3.transpose();
    Matrix dA2 = Matrix::multiply(W3T, dZ3, &err); // error signal
     dZ2 = dA2.hadamard_product(Z2.deriv_leaky_RELU(), &err);

    // ---- Step 4: Gradients for W2, b2 ----
    Matrix A1T = A1.transpose();
    dW2 = Matrix::multiply(dZ2, A1T, &err);
    dW2.scale_inplace(1.0f / batch_size);

    db2 = mean_over_batch(dZ2);

    // ---- Step 5: Backprop to hidden layer 1 ----
    Matrix W2T = W2.transpose();
    Matrix dA1 = Matrix::multiply(W2T, dZ2, &err);
     dZ1 = dA1.hadamard_product(Z1.deriv_leaky_RELU(), &err);

    // ---- Step 6: Gradients for W1, b1 ----
    Matrix XT = X.transpose();
    dW1 = Matrix::multiply(dZ1, XT, &err);
    dW1.scale_inplace(1.0f / batch_size);

    db1 = mean_over_batch(dZ1);

    // ---- Step 7: Clip to avoid exploding gradients ----
    float clip = 1.0f;
    dW1 = dW1.clip(-clip, clip);
    dW2 = dW2.clip(-clip, clip);
    dW3 = dW3.clip(-clip, clip);
    db1 = db1.clip(-clip, clip);
    db2 = db2.clip(-clip, clip);
    db3 = db3.clip(-clip, clip);
}
// Add gradient diagnostics
// std::cout << "dZ2 range: " << dZ2.min() << " to " << dZ2.max() << "\n";
// std::cout << "dZ1 range: " << dZ1.min() << " to " << dZ1.max() << "\n";
// std::cout << "ReLU' active: "
//   << dZ1_relu.mean() * 100.0 << "%\n";
//          // Add gradient clipping after calculations
// dW1 = dW1.clip(-.50, .50);
// dW2 = dW2.clip(-.50, .50);
// db1 = db1.clip(-.50, .50);
// db2 = db2.clip(-.50, .50);
// Scaling instead
// dW1 = dW1.multiply_scalar(100.0);  // Boost gradients
// dW2 = dW2.multiply_scalar(100.0);
// db1 = db1.multiply_scalar(100.0);
// db2 = db2.multiply_scalar(100.0);

// // Helper function (keep it inline for performance)
//  void update_layer_params(Matrix& param, Matrix& v, Matrix& s,
//                                float lr_corrected, float epsilon, Error* err) {
//     s.sqrt_inplace();
//     s.add_inplace_reg(epsilon);
//     v.multiply_scalar_inplace(lr_corrected); // In-place scaling
//     v.hadamard_division_inplace(s, err);     // v = lr_corrected * v / (sqrt(s) + epsilon)
//     if (checkError(*err)) return;
//     param.subtract_inplace_element(v);        // In-place subtraction
// }

// void MLP::update_params(float learning_rate) {
//     const float beta1 = 0.9f;
//     const float beta2 = 0.999f;
//     const float epsilon = 1e-8f;
//     static int t = 0;
//     t++;

//     Error err = NO_ERROR;

//     // Precompute factors once
//     const float v_correction = 1.0f / (1.0f - std::pow(beta1, t));
//     const float s_correction = 1.0f / (1.0f - std::pow(beta2, t));
//     const float lr_corrected = learning_rate * v_correction;
//     const float beta1_denom = 1.0f - beta1;
//     const float beta2_denom = 1.0f - beta2;

//     // Update first moment (momentum)
//     vW1.scale_inplace(beta1);
//     vW1.add_inplace(dW1, beta1_denom);

//     vW2.scale_inplace(beta1);
//     vW2.add_inplace(dW2, beta1_denom);

//     vb1.scale_inplace(beta1);
//     vb1.add_inplace(db1, beta1_denom);

//     vb2.scale_inplace(beta1);
//     vb2.add_inplace(db2, beta1_denom);

//     // Update second moment (variance) - using corrected add_inplace_squared
//     sW1.scale_inplace(beta2);
//     sW1.add_inplace_squared(dW1, beta2_denom);

//     sW2.scale_inplace(beta2);
//     sW2.add_inplace_squared(dW2, beta2_denom);

//     sb1.scale_inplace(beta2);
//     sb1.add_inplace_squared(db1, beta2_denom);

//     sb2.scale_inplace(beta2);
//     sb2.add_inplace_squared(db2, beta2_denom);

//     // Bias correction
//     vW1.scale_inplace(v_correction);
//     vW2.scale_inplace(v_correction);
//     vb1.scale_inplace(v_correction);
//     vb2.scale_inplace(v_correction);

//     sW1.scale_inplace(s_correction);
//     sW2.scale_inplace(s_correction);
//     sb1.scale_inplace(s_correction);
//     sb2.scale_inplace(s_correction);

//     // Parameter updates with error checking
//     update_layer_params(W1, vW1, sW1, lr_corrected, epsilon, &err);
//     if (checkError(err)) return;

//     update_layer_params(W2, vW2, sW2, lr_corrected, epsilon, &err);
//     if (checkError(err)) return;

//     update_layer_params(b1, vb1, sb1, lr_corrected, epsilon, &err);
//     if (checkError(err)) return;

//     update_layer_params(b2, vb2, sb2, lr_corrected, epsilon, &err);
//     if (checkError(err)) return;
// }

// Non-destructive helper: param -= lr * m_hat / (sqrt(v_hat)+eps)
void MLP::update_params(float learning_rate)
{
    static int t = 0; // timestep (persist across calls)
    t++;

    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;

    Error err;

    // ---- W1 ----
    vW1.multiply_scalar_inplace(beta1);
    vW1.add_inplaceMat(dW1.multiply_scalar(1.0f - beta1));

    sW1.multiply_scalar_inplace(beta2);
    sW1.add_inplaceMat(dW1.hadamard_product(dW1).multiply_scalar(1.0f - beta2));

    Matrix vW1_hat = vW1.multiply_scalar(1.0f / (1.0f - std::pow(beta1, t)));
    Matrix sW1_hat = sW1.multiply_scalar(1.0f / (1.0f - std::pow(beta2, t)));

    Matrix denomW1 = sW1_hat.sqrt();
    denomW1.add_inplace(epsilon);

    Matrix stepW1 = vW1_hat;
    stepW1.hadamard_division_inplace(denomW1, &err);
    stepW1.multiply_scalar_inplace(learning_rate);

    W1.subtract_inplace_element(stepW1);

    // ---- b1 ----
    vb1.multiply_scalar_inplace(beta1);
    vb1.add_inplaceMat(db1.multiply_scalar(1.0f - beta1));

    sb1.multiply_scalar_inplace(beta2);
    sb1.add_inplaceMat(db1.hadamard_product(db1).multiply_scalar(1.0f - beta2));

    Matrix vb1_hat = vb1.multiply_scalar(1.0f / (1.0f - std::pow(beta1, t)));
    Matrix sb1_hat = sb1.multiply_scalar(1.0f / (1.0f - std::pow(beta2, t)));

    Matrix denomB1 = sb1_hat.sqrt();
    denomB1.add_inplace(epsilon);

    Matrix stepB1 = vb1_hat;
    stepB1.hadamard_division_inplace(denomB1, &err);
    stepB1.multiply_scalar_inplace(learning_rate);

    b1.subtract_inplace_element(stepB1);

    // ---- W2 ----
    vW2.multiply_scalar_inplace(beta1);
    vW2.add_inplaceMat(dW2.multiply_scalar(1.0f - beta1));

    sW2.multiply_scalar_inplace(beta2);
    sW2.add_inplaceMat(dW2.hadamard_product(dW2).multiply_scalar(1.0f - beta2));

    Matrix vW2_hat = vW2.multiply_scalar(1.0f / (1.0f - std::pow(beta1, t)));
    Matrix sW2_hat = sW2.multiply_scalar(1.0f / (1.0f - std::pow(beta2, t)));

    Matrix denomW2 = sW2_hat.sqrt();
    denomW2.add_inplace(epsilon);

    Matrix stepW2 = vW2_hat;
    stepW2.hadamard_division_inplace(denomW2, &err);
    stepW2.multiply_scalar_inplace(learning_rate);

    W2.subtract_inplace_element(stepW2);

    // ---- b2 ----
    vb2.multiply_scalar_inplace(beta1);
    vb2.add_inplaceMat(db2.multiply_scalar(1.0f - beta1));

    sb2.multiply_scalar_inplace(beta2);
    sb2.add_inplaceMat(db2.hadamard_product(db2).multiply_scalar(1.0f - beta2));

    Matrix vb2_hat = vb2.multiply_scalar(1.0f / (1.0f - std::pow(beta1, t)));
    Matrix sb2_hat = sb2.multiply_scalar(1.0f / (1.0f - std::pow(beta2, t)));

    Matrix denomB2 = sb2_hat.sqrt();
    denomB2.add_inplace(epsilon);

    Matrix stepB2 = vb2_hat;
    stepB2.hadamard_division_inplace(denomB2, &err);
    stepB2.multiply_scalar_inplace(learning_rate);

    b2.subtract_inplace_element(stepB2);

    // ---- W3 ----
    vW3.multiply_scalar_inplace(beta1);
    vW3.add_inplaceMat(dW3.multiply_scalar(1.0f - beta1));

    sW3.multiply_scalar_inplace(beta2);
    sW3.add_inplaceMat(dW3.hadamard_product(dW3).multiply_scalar(1.0f - beta2));

    Matrix vW3_hat = vW3.multiply_scalar(1.0f / (1.0f - std::pow(beta1, t)));
    Matrix sW3_hat = sW3.multiply_scalar(1.0f / (1.0f - std::pow(beta2, t))); // ✅ fixed

    Matrix denomW3 = sW3_hat.sqrt(); // ✅ now correct
    denomW3.add_inplace(epsilon);

    Matrix stepW3 = vW3_hat;
    stepW3.hadamard_division_inplace(denomW3, &err);
    stepW3.multiply_scalar_inplace(learning_rate);

    W3.subtract_inplace_element(stepW3);

    // ---- b3 ----   // ✅ fixed comment
    vb3.multiply_scalar_inplace(beta1);
    vb3.add_inplaceMat(db3.multiply_scalar(1.0f - beta1));

    sb3.multiply_scalar_inplace(beta2);
    sb3.add_inplaceMat(db3.hadamard_product(db3).multiply_scalar(1.0f - beta2));

    Matrix vb3_hat = vb3.multiply_scalar(1.0f / (1.0f - std::pow(beta1, t)));
    Matrix sb3_hat = sb3.multiply_scalar(1.0f / (1.0f - std::pow(beta2, t)));

    Matrix denomB3 = sb3_hat.sqrt();
    denomB3.add_inplace(epsilon);

    Matrix stepB3 = vb3_hat; // ✅ fixed (was vb2_hat)
    stepB3.hadamard_division_inplace(denomB3, &err);
    stepB3.multiply_scalar_inplace(learning_rate);

    b3.subtract_inplace_element(stepB3);
}

// Matrix MLP::get_predictions(const Matrix& A) {
//     // A should be a (num_classes, batch_size) matrix of probabilities
//     size_t num_classes = A.row_count();
//     size_t batch_size = A.col_count();

//     Matrix predictions(1, batch_size);  // Will store class indices

//     for (size_t col = 0; col < batch_size; ++col) {
//         double max_val = A(0, col);
//         int max_idx = 0;

//         // Find class with highest probability
//         for (size_t row = 1; row < num_classes; ++row) {
//             if (A(row, col) > max_val) {
//                 max_val = A(row, col);
//                 max_idx = row;
//             }
//         }

//         predictions(0, col) = max_idx;
//     }

//     return predictions;
// }

Matrix MLP::get_predictions(const Matrix &A)
{
    // Validate input matrix
    if (A.row_count() == 0 || A.col_count() == 0)
    {
        std::cerr << "Empty probability matrix\n";
        return Matrix(1, 0);
    }

    // Add explicit binary classification check
    if (A.row_count() < 2)
    {
        std::cerr << "Output matrix needs >=2 rows for classification\n";
        return Matrix(1, 0);
    }

    size_t batch_size = A.col_count();
    Matrix predictions(1, batch_size);
    Error err = NO_ERROR;

    for (size_t col = 0; col < batch_size; ++col)
    {
        // Validate each column sum ≈ 1.0 (softmax output)
        float col_sum = 0.0f;
        for (size_t row = 0; row < A.row_count(); ++row)
        {
            col_sum += A(row, col, &err);
            if (checkError(err))
                return Matrix(1, 0);
        }

        if (std::abs(col_sum - 1.0) > 1e-5)
        {
            std::cerr << "Invalid probability distribution in column "
                      << col << " (sum=" << col_sum << ")\n";
            return Matrix(1, 0);
        }

        // Binary classification
        if (A.row_count() == 2)
        {
            const float prob0 = A(0, col, &err);
            const float prob1 = A(1, col, &err);
            if (checkError(err))
                return Matrix(1, 0);

            predictions(0, col, &err) = (prob1 > prob0) ? 1.0 : 0.0;
        }
        // Multi-class classification
        else
        {
            float max_val = A(0, col, &err);
            size_t max_idx = 0;

            for (size_t row = 1; row < A.row_count(); ++row)
            {
                const float val = A(row, col, &err);
                if (checkError(err))
                    return Matrix(1, 0);

                if (val > max_val)
                {
                    max_val = val;
                    max_idx = row;
                }
            }
            predictions(0, col, &err) = static_cast<float>(max_idx);
        }
        if (checkError(err))
            return Matrix(1, 0);
    }
    return predictions;
}
// float MLP::get_accuracy(const Matrix& predictions, const Matrix& Y_true) {
//     // Ensure predictions are row vectors
//     if (predictions.row_count() != 1) {
//         std::cerr << "get_accuracy: Predictions must be 1 x batch_size\n";
//         return 0.0;
//     }

//     // Convert Y_true to row vector if it's a column vector
//     Matrix Y_row;
//     if (Y_true.row_count() > 1 && Y_true.col_count() == 1) {
//         Y_row = Y_true.transpose();  // Convert column to row
//     } else {
//         Y_row = Y_true;  // Already in row format
//     }

//     if (Y_row.row_count() != 1) {
//         std::cerr << "Y_true must be convertible to 1 x batch_size. Got: "
//                   << Y_true.row_count() << " x " << Y_true.col_count() << "\n";
//         return 0.0;
//     }

//     // Proceed with accuracy calculation...
//     size_t correct = 0;
//     for (size_t i = 0; i < predictions.col_count(); ++i) {
//         if (predictions(0, i) == Y_row(0, i)) {
//             correct++;
//         }
//     }
//     return static_cast<float>(correct) / predictions.col_count();
// }

float MLP::get_accuracy(const Matrix &predictions, const Matrix &Y_true)
{
    if (predictions.col_count() != Y_true.col_count())
    {
        std::cerr << "Predictions and labels must have same number of columns\n";
        return 0.0;
    }

    size_t correct = 0;
    for (size_t i = 0; i < predictions.col_count(); ++i)
    {
        // Handle both row and column vectors
        float pred = predictions.row_count() == 1 ? predictions(0, i) : predictions(i, 0);
        float true_val = Y_true.row_count() == 1 ? Y_true(0, i) : Y_true(i, 0);

        if (pred == true_val)
        {
            correct++;
        }
    }
    return static_cast<float>(correct) / predictions.col_count();
}

void shuffle_data(Matrix &X, Matrix &Y)
{
    size_t batch_size = X.col_count();
    std::vector<size_t> indices(batch_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    Matrix X_shuffled(X.row_count(), batch_size);
    Matrix Y_shuffled(Y.row_count(), batch_size);

    for (size_t i = 0; i < batch_size; ++i)
    {
        size_t src_idx = indices[i];
        for (size_t r = 0; r < X.row_count(); ++r)
        {
            X_shuffled(r, i) = X(r, src_idx);
        }
        Y_shuffled(0, i) = Y(0, src_idx);
    }

    X = X_shuffled;
    Y = Y_shuffled;
}

float MLP::cross_entropy_loss(const Matrix &Y_pred, const Matrix &Y_true)
{
    float loss = 0.0;
    size_t batch_size = Y_pred.col_count();
    Error err = NO_ERROR;

    // Shape validation
    if (Y_true.row_count() != 1 || Y_true.col_count() != batch_size)
    {
        std::cerr << "Y_true must be 1 x batch_size. Got: "
                  << Y_true.row_count() << " x " << Y_true.col_count() << "\n";
        return 0.0;
    }

    for (size_t i = 0; i < batch_size; ++i)
    {
        int label = static_cast<int>(Y_true(0, i, &err));
        if (checkError(err))
            return 0.0;

        if (label < 0 || static_cast<size_t>(label) >= Y_pred.row_count())
        {
            std::cerr << "Invalid label: " << label << " at index " << i
                      << " (Y_pred has " << Y_pred.row_count() << " rows)\n";
            return 0.0;
        }

        float prob = Y_pred(label, i, &err);
        if (checkError(err))
            return 0.0;

        loss += -std::log(std::max(prob, 1e-8f));
    }

    return loss / batch_size;
}

void MLP::gradient_descent(Matrix &X, Matrix &Y, size_t epochs, float learning_rate)
{
    float decay_rate = 0.01f;
    float initial_lr = learning_rate;
    Error err = NO_ERROR;

    // Normalize dataset to [0,1]
    X = X.multiply_scalar(1.0f / 255.0f);

    float current_lr;
    float accuracy;
    float loss;
    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        // Shuffle dataset at the start of each epoch
        shuffle_data(X, Y);

        // Forward propagation on the whole dataset
        Matrix A3 = forward_prop(X);
        if (A3.row_count() == 0)
        {
            std::cerr << "Forward propagation failed at epoch " << epoch << "\n";
            continue;
        }

        // Backpropagation on the whole dataset
        back_prop(X, Y);

        // Adjust learning rate with decay
        current_lr = initial_lr * (1.0 / (1.0 + decay_rate * epoch));
        update_params(current_lr);

        // Evaluate epoch performance
        Matrix predictions = get_predictions(A3);
        accuracy = get_accuracy(predictions, Y);
        loss = cross_entropy_loss(A3, Y);

        // std::cout << "Epoch " << epoch
        //           << " | Loss: " << loss
        //           << " | Acc: " << accuracy
        //           << " | LR: " << current_lr
        //           << "\n";
    }
    std::cout << "Epoch " << epochs
              << " | Loss: " << loss
              << " | Acc: " << accuracy
              << " | LR: " << current_lr
              << "\n";
}
