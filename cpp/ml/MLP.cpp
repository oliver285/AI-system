
#include "MLP.h"
#include <iostream>
#include <iomanip>
#include <cmath>
MLP::MLP(size_t input_size, size_t hidden_size, size_t output_size)
: W1(hidden_size, input_size),
vW1(hidden_size, input_size),  // Matches declaration order
W2(output_size, hidden_size),
vW2(output_size, hidden_size),
dW1(hidden_size, input_size),
dW2(output_size, hidden_size),
b1(1, hidden_size),
vb1(1, hidden_size),
b2(1, output_size),
vb2(1, output_size),
db1(1, hidden_size),
db2(1, output_size)
{
    double scale1 = std::sqrt(2.0 / input_size);  // He initialization
    double scale2 = std::sqrt(2.0 / hidden_size);
// 
// double scale1 = std::sqrt(1.0 / (input_size + hidden_size));
// double scale2 = std::sqrt(1.0 / (hidden_size + output_size));
    
    W1 = Matrix::random(hidden_size, input_size).multiply_scalar(scale1);
    W2 = Matrix::random(output_size, hidden_size).multiply_scalar(scale2);
    
    // Initialize biases to zero
    b1.fill(0.0);
    b2.fill(0.0);
    
    // Initialize velocities to zero
    vW1.fill(0.0);
    vW2.fill(0.0);
    vb1.fill(0.0);
    vb2.fill(0.0);
}

double MLP::compute_loss(const Matrix& Y, const Matrix& A2) {
    Matrix one_hot_Y = one_hot(Y, A2.row_count());
    double loss = 0.0;
    for (size_t i = 0; i < A2.col_count(); ++i) {
        for (size_t j = 0; j < A2.row_count(); ++j) {
            double y = one_hot_Y(j, i);
            double a = A2(j, i);
            if (y > 0) {
                loss -= std::log(std::max(a, 1e-10));  // avoid log(0)
            }
        }
    }
    return loss / A2.col_count();  // average over batch
}


bool checkError(Error err) {
    if (err != NO_ERROR) {
        const char* error_msg = "Unknown error";
        switch(err) {
            case INDEX_OUT_OF_RANGE:
                error_msg = "Index out of range";
                break;
            case DIMENSION_MISMATCH:
                error_msg = "Dimension mismatch";
                break;
            case NO_ERROR: // Not needed but for completeness
                return false;
        }
        
        // For flight controllers, use proper logging instead of std::cerr
        // flight_log(LOG_CRITICAL, "Matrix error: %s", error_msg);
        
        std::cerr << "Matrix error: " << error_msg << "\n";
        return true;
    }
    return false;
}
Matrix MLP::forward_prop(const Matrix& X) {
    // Validate input dimensions
    // if (X.row_count() != input_size) {
    //     std::cerr << "Input dimension mismatch. Expected: " 
    //               << input_size << ", Got: " << X.row_count() << "\n";
    //     return Matrix();
    // }

    size_t batch_size = X.col_count();
    if (batch_size == 0) {
        std::cerr << "Empty batch detected\n";
        return Matrix();
    }

    Error err = NO_ERROR;
    
    // Layer 1: Z1 = W1 * X + b1
    Z1 = Matrix::multiply(W1, X, &err);
    if (checkError(err)) return Matrix();
    
    // Validate bias dimensions before addition
    if (Z1.row_count() != b1.col_count()) {
        std::cerr << "Bias dimension mismatch in layer 1. Z1 rows: "
                  << Z1.row_count() << ", b1 cols: " << b1.col_count() << "\n";
        return Matrix();
    }
    
    // Safe bias addition with bounds checking
    for (size_t j = 0; j < batch_size; ++j) {
        for (size_t i = 0; i < Z1.row_count(); ++i) {
            double bias_val = b1(0, i, &err);
            if (checkError(err)) return Matrix();
            
            double& z_val = Z1(i, j, &err);
            if (checkError(err)) return Matrix();
            
            z_val += bias_val;
        }
    }
    
    A1 = Z1.leaky_RELU();
    
    // Layer 2: Z2 = W2 * A1 + b2
    Z2 = Matrix::multiply(W2, A1, &err);
    if (checkError(err)) return Matrix();
    
    // Validate bias dimensions
    if (Z2.row_count() != b2.col_count()) {
        std::cerr << "Bias dimension mismatch in layer 2. Z2 rows: "
                  << Z2.row_count() << ", b2 cols: " << b2.col_count() << "\n";
        return Matrix();
    }
    
    // Safe bias addition
    for (size_t j = 0; j < batch_size; ++j) {
        for (size_t i = 0; i < Z2.row_count(); ++i) {
            double bias_val = b2(0, i, &err);
            if (checkError(err)) return Matrix();
            
            double& z_val = Z2(i, j, &err);
            if (checkError(err)) return Matrix();
            
            z_val += bias_val;
        }
    }
    
    A2 = Matrix::softmax(Z2);
    return A2;
}

Matrix MLP::one_hot(const Matrix& Y, size_t num_classes) {
    // Validate input dimensions
    if (Y.row_count() != 1) {
        return Matrix();  // Use consistent error handling
    }

    const size_t batch_size = Y.col_count();
    Matrix one_hot_Y(num_classes, batch_size);
    one_hot_Y.fill(0.0);

    for (size_t i = 0; i < batch_size; ++i) {
        const double val = Y(0, i);
        
        // 1. Check if value is a valid integer
        if (std::abs(val - std::round(val)) > 1e-8) {
            return Matrix();  // Not an integer
        }
        
        const int class_idx = static_cast<int>(std::round(val));
        
        // 2. Validate class index range
        if (class_idx < 0 || class_idx >= static_cast<int>(num_classes)) {
            return Matrix();  // Out of range
        }

        one_hot_Y(class_idx, i) = 1.0;
    }

    return one_hot_Y;
}

        // Matrix<2, 784>  deriv_ReLU() {
        //     for (uint16_t i = 0; i < rows; ++i)
        //         for (uint16_t j = 0; j < cols; ++j)
        //             if (data[i][j] < 0)
        //                 data[i][j] = 0;
        // }


        void MLP::back_prop(const Matrix& X, const Matrix& Y) {
            Error err = NO_ERROR;
            size_t batch_size = X.col_count();
        
            // Validate input dimensions
            if (batch_size != Y.col_count()) {
                std::cerr << "X and Y batch size mismatch" << std::endl;
                return;
            }
        
            // Convert Y to one-hot encoding with error handling
            Matrix one_hot_Y = one_hot(Y, A2.row_count());
            if (one_hot_Y.row_count() == 0 || one_hot_Y.col_count() == 0) {
                std::cerr << "one_hot encoding failed" << std::endl;
                return;
            }
        
            // Step 2: dZ2 = A2 - one_hot_Y (use safe subtraction)
            Matrix dZ2 = A2.subtract(one_hot_Y, &err);
            if (checkError(err)) return;
        
            // Step 3: dW2 = (dZ2 * A1^T) / batch_size
            Matrix A1T = A1.transpose();
            dW2 = Matrix::multiply(dZ2, A1T, &err);
            if (checkError(err)) return;
            dW2.scale_inplace(1.0 / batch_size);  // More efficient in-place scaling
        
            // Step 4: db2 = mean of dZ2 across batch
            db2 = Matrix(1, dZ2.row_count());
            for (size_t i = 0; i < dZ2.row_count(); ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < batch_size; ++j) {
                    sum += dZ2(i, j, &err);  // Safe access
                    if (checkError(err)) return;
                }
                db2(0, i, &err) = sum / batch_size;  // Safe assignment
                if (checkError(err)) return;
            }
            
            // Step 5: dZ1 = (W2^T * dZ2) ⊙ leaky_RELU'(Z1)
            Matrix W2T = W2.transpose();
            Matrix dZ1_linear = Matrix::multiply(W2T, dZ2, &err);
            if (checkError(err)) return;
            
            Matrix dZ1_relu = Z1.deriv_leaky_RELU();
            Matrix dZ1 = dZ1_linear.hadamard_product(dZ1_relu, &err);
            if (checkError(err)) return;
            
            // Step 6: dW1 = (dZ1 * X^T) / batch_size
            Matrix XT = X.transpose();
            dW1 = Matrix::multiply(dZ1, XT, &err);
            if (checkError(err)) return;
            dW1.scale_inplace(1.0 / batch_size);  // In-place scaling
            
            // Step 7: db1 = mean of dZ1 across batch
            db1 = Matrix(1, dZ1.row_count());
            for (size_t i = 0; i < dZ1.row_count(); ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < batch_size; ++j) {
                    sum += dZ1(i, j, &err);  // Safe access
                    if (checkError(err)) return;
                }
                db1(0, i, &err) = sum / batch_size;  // Safe assignment
                if (checkError(err)) return;
            }
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

        

        void MLP::update_params(double learning_rate) {
            const double momentum = 0.9;
            Error err = NO_ERROR;
            // Scale gradients in-place
            dW1.scale_inplace(learning_rate);
            dW2.scale_inplace(learning_rate);
            db1.scale_inplace(learning_rate);
            db2.scale_inplace(learning_rate);
            
            // Update velocities
            vW1 = vW1.multiply_scalar(momentum).subtract(dW1,&err);
            if (checkError(err)) return;
            vW2 = vW2.multiply_scalar(momentum).subtract(dW2,&err);
            if (checkError(err)) return;
            vb1 = vb1.multiply_scalar(momentum).subtract(db1,&err);
            if (checkError(err)) return;
            vb2 = vb2.multiply_scalar(momentum).subtract(db2,&err);
            if (checkError(err)) return;
            
            // Update parameters
            W1 = W1.add(vW1,&err);
            if (checkError(err)) return;
            W2 = W2.add(vW2,&err);
            if (checkError(err)) return;
            b1 = b1.add(vb1,&err);
            if (checkError(err)) return;
            b2 = b2.add(vb2,&err);
            if (checkError(err)) return;
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

        Matrix MLP::get_predictions(const Matrix& A) {
            // Validate input matrix
            if (A.row_count() == 0 || A.col_count() == 0) {
                std::cerr << "Empty probability matrix in get_predictions\n";
                return Matrix(1, 0);  // Return empty matrix
            }
        
            size_t batch_size = A.col_count();
            Matrix predictions(1, batch_size);
            Error err = NO_ERROR;
        
            for (size_t col = 0; col < batch_size; ++col) {
                // Validate each column sum ≈ 1.0 (softmax output)
                double col_sum = 0.0;
                for (size_t row = 0; row < A.row_count(); ++row) {
                    col_sum += A(row, col, &err);
                    if (checkError(err)) return Matrix(1, 0);
                }
                
                if (std::abs(col_sum - 1.0) > 1e-5) {
                    std::cerr << "Invalid probability distribution in column " 
                              << col << " (sum=" << col_sum << ")\n";
                    return Matrix(1, 0);
                }
        
                // Binary classification
                if (A.row_count() == 2) {
                    const double prob0 = A(0, col, &err);
                    const double prob1 = A(1, col, &err);
                    if (checkError(err)) return Matrix(1, 0);
                    
                    predictions(0, col, &err) = (prob1 > prob0) ? 1.0 : 0.0;
                } 
                // Multi-class classification
                else {
                    double max_val = A(0, col, &err);
                    size_t max_idx = 0;
                    
                    for (size_t row = 1; row < A.row_count(); ++row) {
                        const double val = A(row, col, &err);
                        if (checkError(err)) return Matrix(1, 0);
                        
                        if (val > max_val) {
                            max_val = val;
                            max_idx = row;
                        }
                    }
                    predictions(0, col, &err) = static_cast<double>(max_idx);
                }
                if (checkError(err)) return Matrix(1, 0);
            }
            return predictions;
        }
        double MLP::get_accuracy(const Matrix& predictions, const Matrix& labels) {
            // Validate input dimensions
            if (predictions.row_count() != 1 || labels.row_count() != 1) {
                std::cerr << "Inputs must be row vectors\n";
                return -1.0;  // Error indicator
            }
            
            const size_t batch_size = predictions.col_count();
            if (batch_size != labels.col_count()) {
                std::cerr << "Batch size mismatch: predictions " << batch_size
                          << " vs labels " << labels.col_count() << "\n";
                return -1.0;
            }
            
            if (batch_size == 0) {
                std::cerr << "Empty batch in accuracy calculation\n";
                return 0.0;  // Defined behavior for empty input
            }
        
            Error err = NO_ERROR;
            size_t correct_count = 0;
        
            for (size_t i = 0; i < batch_size; ++i) {
                const double pred = predictions(0, i, &err);
                const double label = labels(0, i, &err);
                if (checkError(err)) return -1.0;
                
                // Handle both integer and one-hot encoded labels
                if (std::abs(pred - label) < 1e-6 || 
                    std::abs(pred - static_cast<int>(label)) < 1e-6) {
                    correct_count++;
                }
            }
            
            return static_cast<double>(correct_count) / batch_size;
        }

        void shuffle_data(Matrix& X, Matrix& Y) {
            size_t batch_size = X.col_count();
            std::vector<size_t> indices(batch_size);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
        
            Matrix X_shuffled(X.row_count(), batch_size);
            Matrix Y_shuffled(Y.row_count(), batch_size);
        
            for (size_t i = 0; i < batch_size; ++i) {
                size_t src_idx = indices[i];
                for (size_t r = 0; r < X.row_count(); ++r) {
                    X_shuffled(r, i) = X(r, src_idx);
                }
                Y_shuffled(0, i) = Y(0, src_idx);
            }
        
            X = X_shuffled;
            Y = Y_shuffled;
        }


        double MLP::cross_entropy_loss(const Matrix& Y_pred, const Matrix& Y_true) {
            double loss = 0.0;
            size_t batch_size = Y_pred.col_count();
            
            for (size_t i = 0; i < batch_size; ++i) {
                int label = static_cast<int>(Y_true(0, i));
                double prob = Y_pred(label, i);
                loss += -std::log(std::max(prob, 1e-8));  // Avoid log(0)
            }
            return loss / batch_size;
        }


void MLP::gradient_descent(Matrix& X, Matrix& Y, size_t iterations, double learning_rate) {
    double decay_rate = 0;  // Changed to double, more standard value
    Matrix predictions;
    
    double initial_lr=learning_rate;
    double best_loss = std::numeric_limits<double>::max();
    size_t no_improve = 0;
    for (size_t i = 0; i < iterations; ++i) {  // Better loop structure
        shuffle_data(X, Y);
        // Forward propagation
        Matrix A2 = forward_prop(X);
        
        // Backpropagation
        back_prop(X, Y);
           // Learning rate decay
        //    double current_lr = initial_lr * (1.0 / (1.0 + decay_rate * i/10.0));
        // Adaptive LR based on loss improvement
        // double current_lr = initial_lr * std::exp(-decay_rate * i);
double current_lr = initial_lr * (1.0 / (1.0 + decay_rate * i/iterations));
        double current_loss = cross_entropy_loss(A2, Y);
      // Early stopping with patience
        if (current_loss < best_loss - 0.001) {
            best_loss = current_loss;
            no_improve = 0;
        } else {
            no_improve++;
            if (no_improve >= 20) {
                std::cout << "Early stopping at iteration " << i << "\n";
                break;
            }
            // Reduce LR when loss plateaus
            current_lr *= 0.5;
        }
        
        update_params(current_lr);
        // Print progress every 10 iterations
        if (i % 10 == 0) {
            Matrix predictions = get_predictions(A2);
            double accuracy = get_accuracy(predictions, Y);
 std::cout << "W1 range: " << W1.min() << " to " << W1.max() << "\n";
std::cout << "b1 range: " << b1.min() << " to " << b1.max() << "\n";
            std::cout << "Iter: " << i 
                      << " | Loss: " << current_loss
                      << " | Acc: " << accuracy
                      << " | LR: " << current_lr
                      << " | dW1: " << dW1.frobenius_norm()
                      << " | dW2: " << dW2.frobenius_norm() << "\n";
        }
    }
}



