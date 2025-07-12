
#include "matrix.h"
// #include <random>

class MLP {
    private:
        // Weights (unchanged)
        Matrix W1, W2, dW1, dW2;  // W1: (10, 784), W2: (2, 10)
        Matrix b1, b2, db1,db2;             // b1: (10, 1),   b2: (2, 1)
    
        // Activations (now batch matrices)
        Matrix Z1, A1, dZ1;  // Z1, A1: (10, batch_size)
        Matrix Z2, A2, dZ2;  // Z2, A2: (2, batch_size)
    
        int batch_size;
    
    public:
    MLP(size_t input_size, size_t hidden_size, size_t output_size)
    : W1(hidden_size, input_size),
      W2(output_size, hidden_size),
      b1(hidden_size, 1),
      b2(output_size, 1) {
    // Initialize weights (He initialization)
    double scale1 = std::sqrt(2.0 / input_size);
    double scale2 = std::sqrt(2.0 / hidden_size);
    
    W1 = Matrix::random(hidden_size, input_size).multiply_scalar(scale1);
    W2 = Matrix::random(output_size, hidden_size).multiply_scalar(scale2);
    b1.fill(0.1);  // Small constant bias initialization
    b2.fill(0.1);
}
// Forward pass for batch processing
Matrix forward_prop(const Matrix& X) {
    // X shape: (input_size, batch_size)
    // size_t batch_size = X.col_count();
    
    // Layer 1
    Z1 = Matrix::multiply(W1, X);  // (hidden_size, batch_size)
    // Broadcast bias addition
    for (size_t i = 0; i < Z1.row_count(); ++i) {
        for (size_t j = 0; j < Z1.col_count(); ++j) {
            Z1(i, j) += b1(i, 0);
        }
    }
    A1 = Z1.RELU();
    
    // Layer 2
    Z2 = Matrix::multiply(W2, A1);  // (output_size, batch_size)
    // Broadcast bias addition
    for (size_t i = 0; i < Z2.row_count(); ++i) {
        for (size_t j = 0; j < Z2.col_count(); ++j) {
            Z2(i, j) += b2(i, 0);
        }
    }
    A2 = Matrix::softmax(Z2);
    
    return A2;
}

Matrix one_hot(const Matrix& Y, size_t num_classes = 2) {
    // Validate input dimensions
    if (Y.row_count() != 1) {
        throw std::invalid_argument("Input Y must be a row vector of shape (1, batch_size)");
    }

    const size_t batch_size = Y.col_count();
    Matrix one_hot_Y(num_classes, batch_size);
    one_hot_Y.fill(0.0);  // Initialize all to 0

    // Convert each label to one-hot vector
    for (size_t i = 0; i < batch_size; ++i) {
        const int class_idx = static_cast<int>(Y(0, i));
        
        // Validate class index
        if (class_idx < 0 || class_idx >= num_classes) {
            throw std::out_of_range(
                "Class index " + std::to_string(class_idx) + 
                " is out of range for num_classes=" + std::to_string(num_classes)
            );
        }

        one_hot_Y(class_idx, i) = 1.0;  // Set the appropriate position to 1
    }

    return one_hot_Y;
} 

        // Matrix<2, 784>  deriv_ReLU() {
        //     for (uint16_t i = 0; i < rows; ++i)
        //         for (uint16_t j = 0; j < cols; ++j)
        //             if (data[i][j] < 0)
        //                 data[i][j] = 0;
        // }


        void back_prop(const Matrix& X, const Matrix& Y) {
            // Step 1: Convert Y to one-hot encoding
            Matrix one_hot_Y = one_hot(Y);  // Shape: (2, batch_size)
            
            // Step 2: dZ2 = A2 - one_hot_Y (output error)
            Matrix dZ2 = A2 - one_hot_Y;  // Element-wise subtraction
            
            // Step 3: dW2 = (dZ2 * A1^T) / batch_size
            dW2 = Matrix::multiply(dZ2, A1.transpose());
            dW2 = dW2.multiply_scalar(1.0 / batch_size);  // Average over batch
            
            // Step 4: db2 = mean of dZ2 across batch (column-wise sum)
            db2 = Matrix(dZ2.row_count(), 1);
            for (size_t i = 0; i < dZ2.row_count(); ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < dZ2.col_count(); ++j) {
                    sum += dZ2(i, j);
                }
                db2(i, 0) = sum / batch_size;
            }
            
            // Step 5: dZ1 = (W2^T * dZ2) ⊙ ReLU'(Z1)
            Matrix dZ1_linear = Matrix::multiply(W2.transpose(), dZ2);
            Matrix dZ1_relu = Z1.deriv_RELU();  // ReLU derivative
            Matrix dZ1 = dZ1_linear.hadamard_product(dZ1_relu);  // Element-wise multiply
            
            // Step 6: dW1 = (dZ1 * X^T) / batch_size
            dW1 = Matrix::multiply(dZ1, X.transpose());
            dW1 = dW1.multiply_scalar(1.0 / batch_size);  // Average over batch
            
            // Step 7: db1 = mean of dZ1 across batch
            db1 = Matrix(dZ1.row_count(), 1);
            for (size_t i = 0; i < dZ1.row_count(); ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < dZ1.col_count(); ++j) {
                    sum += dZ1(i, j);
                }
                db1(i, 0) = sum / batch_size;
            }
        }

        void update_params(double learning_rate) {  // Changed int to double
            // Update weights and biases using gradients
            W1 = W1 - dW1.multiply_scalar(learning_rate);
            b1 = b1 - db1.multiply_scalar(learning_rate);
            W2 = W2 - dW2.multiply_scalar(learning_rate);
            b2 = b2 - db2.multiply_scalar(learning_rate);
        }

        Matrix get_predictions(const Matrix& A) {
            // A should be a (num_classes, batch_size) matrix of probabilities
            size_t num_classes = A.row_count();
            size_t batch_size = A.col_count();
            
            Matrix predictions(1, batch_size);  // Will store class indices
        
            for (size_t col = 0; col < batch_size; ++col) {
                double max_val = A(0, col);
                int max_idx = 0;
        
                // Find class with highest probability
                for (size_t row = 1; row < num_classes; ++row) {
                    if (A(row, col) > max_val) {
                        max_val = A(row, col);
                        max_idx = row;
                    }
                }
        
                predictions(0, col) = max_idx;
            }
        
            return predictions;
        }

        double get_accuracy(const Matrix& predictions, const Matrix& labels) {
            // Validate input dimensions
            if (predictions.row_count() != 1 || labels.row_count() != 1 || 
                predictions.col_count() != labels.col_count()) {
                throw std::invalid_argument(
                    "Input matrices must be row vectors of same length. Got: " +
                    std::to_string(predictions.row_count()) + "x" + std::to_string(predictions.col_count()) +
                    " vs " + std::to_string(labels.row_count()) + "x" + std::to_string(labels.col_count())
                );
            }
        
            const size_t batch_size = predictions.col_count();
            size_t correct_count = 0;  // Fixed missing semicolon
        
            for (size_t i = 0; i < batch_size; ++i) {
                if (std::abs(predictions(0,i) - labels(0,i)) < 1e-6) {  // Proper spacing around operators
                    correct_count++;
                }
            }
        
            return static_cast<double>(correct_count) / batch_size;
        }

void gradient_descent(const Matrix& X, const Matrix& Y, size_t iterations, double learning_rate) {
    double decay_rate = 0.01;  // Changed to double, more standard value
    Matrix predictions;
    double accuracy;
    
    for (size_t i = 0; i < iterations; ++i) {  // Better loop structure
        // Forward propagation
        Matrix A2 = forward_prop(X);
        
        // Backpropagation
        back_prop(X, Y);
        
        // Update parameters
        update_params(learning_rate);
        
        // Learning rate decay
        learning_rate = learning_rate * (1.0 / (1.0 + decay_rate * i));
        
        // Print progress every 10 iterations
        if (i % 10 == 0) {
            predictions = get_predictions(A2);
            accuracy = get_accuracy(predictions, Y);
            std::cout << "Iteration: " << i 
                      << ", Accuracy: " << std::setprecision(4) << std::fixed << accuracy
                      << ", Learning Rate: " << learning_rate << std::endl;
        }
    }
}
/* potential alternative metric once code has proven functional*/


// struct TrainingMetrics {
//     double train_loss;
//     double train_accuracy;
//     double val_accuracy;
//     double learning_rate;
// };

// TrainingMetrics gradient_descent(Matrix& X_train, Matrix& Y_train, 
//                                const Matrix& X_val, const Matrix& Y_val,
//                                size_t iterations, double initial_lr,
//                                double min_lr = 1e-5, size_t patience = 20) {
//     double decay_rate = 0.01;
//     double best_val_accuracy = 0.0;
//     size_t no_improvement_count = 0;
//     std::vector<TrainingMetrics> history;

//     for (size_t i = 0; i < iterations; ++i) {
//         // 1. Forward pass
//         Matrix A2 = forward_prop(X_train);
        
//         // 2. Calculate metrics
//         double loss = compute_loss(A2, Y_train);
//         double accuracy = get_accuracy(get_predictions(A2), Y_train);
        
//         // 3. Backpropagation
//         back_prop(X_train, Y_train);
        
//         // 4. Gradient clipping
//         clip_gradients(5.0); // Prevent exploding gradients
        
//         // 5. Update parameters
//         update_params(initial_lr);
        
//         // 6. Learning rate decay with lower bound
//         initial_lr = std::max(min_lr, 
//                             initial_lr * (1.0 / (1.0 + decay_rate * i)));
        
//         // 7. Validation check
//         double val_acc = 0;
//         if (i % 10 == 0) {
//             Matrix val_pred = forward_prop(X_val);
//             val_acc = get_accuracy(get_predictions(val_pred), Y_val);
            
//             // Early stopping check
//             if (val_acc > best_val_accuracy) {
//                 best_val_accuracy = val_acc;
//                 no_improvement_count = 0;
//                 // save_best_weights();
//             } else {
//                 no_improvement_count++;
//             }
//         }

//         // 8. Store metrics
//         history.push_back({loss, accuracy, val_acc, initial_lr});

//         // 9. Progress reporting
//         if (i % 10 == 0) {
//             std::cout << fmt::format(
//                 "Iter {:4d} | Loss: {:.4f} | Train Acc: {:.2f}% | "
//                 "Val Acc: {:.2f}% | lr: {:.6f} | No imp: {}/{}",
//                 i, loss, accuracy*100, val_acc*100, initial_lr,
//                 no_improvement_count, patience
//             ) << std::endl;
//         }

//         // 10. Early stopping
//         if (no_improvement_count >= patience) {
//             std::cout << "Early stopping triggered" << std::endl;
//             break;
//         }
//     }

//     return history.back();
// }

// def predict_new_image(image_path, W1, b1, W2, b2):
//     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
//     img = cv2.resize(img, (28, 28))
//     img_flatten = img.flatten().reshape(-1, 1)
//     _, _, _, A2 = forward_prop(W1, b1, W2, b2, img_flatten)
//     prediction = get_predictions(A2)[0]
//     label = "Cracked" if prediction == 1 else "Not Cracked"
//     print(f"{image_path}: {label}")
//     return label

// # =======================
// # Preprocessing Dataset
// # =======================
// def load_images_from_folder(folder, label, image_size=(28, 28)):
//     data = []
//     for filename in os.listdir(folder):
//         img_path = os.path.join(folder, filename)
//         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
//         if img is not None:
//             img = cv2.resize(img, image_size)
//             img_flatten = img.flatten()
//             data.append(np.insert(img_flatten, 0, label))
//     return data

// cracked = load_images_from_folder("datasets/testing/data/Cracked", label=1)
// not_cracked = load_images_from_folder("datasets/testing/data/NonCracked", label=0)
// all_data = np.array(cracked + not_cracked)
// np.random.shuffle(all_data)
// pd.DataFrame(all_data).to_csv("crack_dataset.csv", index=False)

// # =======================
// # Load + Split Dataset
// # =======================
// data = pd.read_csv('crack_dataset.csv').to_numpy()
// np.random.shuffle(data)
// m, n = data.shape

// if m < 200:
//     split_idx = m // 5  # 20% dev set if small
// else:
//     split_idx = 100

// data_dev = data[:split_idx].T
// Y_dev = data_dev[0].astype(int)
// X_dev = data_dev[1:n]

// data_train = data[split_idx:].T
// Y_train = data_train[0].astype(int)
// X_train = data_train[1:n]

// # =======================
// # Train Model
// # =======================
// W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=100, alpha=0.1)

// # Evaluate on Dev Set
// _, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
// accuracy_dev = get_accuracy(get_predictions(A2_dev), Y_dev)
// print(f"Dev Set Accuracy: {accuracy_dev:.4f}")


        

    };


    




// BLA::Matrix<3, 3, float> jacobian(BLA::Matrix<3,1, float> p, 
//                                   BLA::Matrix<3,1, float> tether_lengths, 
//                                   BLA::Matrix<3, 3, float> teth_anchor, 
//                                   BLA::Matrix<3, 3, float> offset) {
//     BLA::Matrix<3, 3, float> J;
//     double h = 1e-5;
//     BLA::Matrix<3,1, float> p1, f1, f2;

//     for (int i = 0; i < 3; i++) {
//         p1 = p;
//         p1(i) += h;
//         f1 = equations(p1, teth_anchor, offset, tether_lengths);
        
//         p1(i) -= 2*h;
//         f2 = equations(p1, teth_anchor, offset, tether_lengths);

//         for (int j = 0; j < 3; j++) {
//             J(j, i) = (f1(j) - f2(j)) / (2 * h);
//         }
//     }
//     return J;
// }