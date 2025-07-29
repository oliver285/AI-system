#include "../core/matrix.h"
#include "../ml/MLP.h"
#include "../vision/image_processor.h"

int main() {
    try {
        image_processor processor;
        const uint8_t IMG_WIDTH = 28, IMG_HEIGHT = 28;
        const size_t input_size = IMG_WIDTH * IMG_HEIGHT;  // Correct input size

        // Load datasets
        Matrix cracked_images = processor.load_dataset("../datasets/testing/data/Cracked");
        Matrix cracked_labels = processor.load_labels("../datasets/testing/data/Cracked", 1);
        Matrix noncracked_images = processor.load_dataset("../datasets/testing/data/NonCracked");
        Matrix noncracked_labels = processor.load_labels("../datasets/testing/data/NonCracked", 0);

        // Validate dataset sizes
        if (cracked_images.col_count() != input_size || 
            noncracked_images.col_count() != input_size) {
            std::cerr << "Error: Image dimensions do not match expected input size\n";
            return 1;
        }

        // Combine images and labels
        size_t total_samples = cracked_images.row_count() + noncracked_images.row_count();
        Matrix all_images(total_samples, input_size);
        Matrix all_labels(total_samples, 1);

        // Copy cracked data
        for (size_t i = 0; i < cracked_images.row_count(); ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                all_images(i, j) = cracked_images(i, j);
            }
            all_labels(i, 0) = cracked_labels(i, 0);
        }

        // Copy non-cracked data
        size_t offset = cracked_images.row_count();
        for (size_t i = 0; i < noncracked_images.row_count(); ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                all_images(i + offset, j) = noncracked_images(i, j);
            }
            all_labels(i + offset, 0) = noncracked_labels(i, 0);
        }

        // Shuffle the combined dataset
        processor.shuffle_dataset(all_images, all_labels);
        
        // Split into training and testing sets (80/20)
        // After combining and shuffling:
        // size_t total_samples = all_images.row_count();
        Matrix all_labels_row(1, total_samples);  // Create row vector for labels

        // Convert labels to row vector
        for (size_t i = 0; i < total_samples; ++i) {
            all_labels_row(0, i) = all_labels(i, 0);
        }

        // Split into training and testing sets (80/20)
        size_t train_size = total_samples * 0.8;
        size_t test_size = total_samples - train_size;

        Matrix train_images(train_size, input_size);
        Matrix train_labels(1, train_size);  // Row vector
        Matrix test_images(test_size, input_size);
        Matrix test_labels(1, test_size);    // Row vector

        // Copy training data
        for (size_t i = 0; i < train_size; ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                train_images(i, j) = all_images(i, j);
            }
            train_labels(0, i) = all_labels_row(0, i);  // Assign to row vector
        }

        // Copy testing data
        for (size_t i = 0; i < test_size; ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                test_images(i, j) = all_images(i + train_size, j);
            }
            test_labels(0, i) = all_labels_row(0, i + train_size);  // Assign to row vector
        }

        // Initialize MLP with correct dimensions
        const size_t hidden_size = 64;  // Reasonable hidden layer size
        const size_t output_size = 2;   // Binary classification
        MLP mlp(input_size, hidden_size, output_size);
        
     // After splitting into train/test sets:
Matrix train_images_T = train_images.transpose();
Matrix test_images_T = test_images.transpose();

// Pass transposed matrices to MLP functions
mlp.gradient_descent(train_images_T, train_labels, 1000, 0.1);
Matrix output = mlp.forward_prop(test_images_T);
        if (output.row_count() == 0 || output.col_count() == 0) {
            std::cerr << "Forward propagation failed\n";
            return 1;
        }
        
        Matrix predictions = mlp.get_predictions(output);
        if (predictions.row_count() == 0 || predictions.col_count() == 0) {
            std::cerr << "Prediction failed\n";
            return 1;
        }
        
        std::cout << "\nTest set predictions:\n";
        predictions.print();
        
        double accuracy = mlp.get_accuracy(predictions, test_labels);
        std::cout << "Test accuracy: " << accuracy << "\n";
    
        std::cout << "\nPrediction confidence:\n";
        // size_t batch_size = test_images.row_count();
        // for (size_t i = 0; i < batch_size; i++) {
        //     int pred = static_cast<int>(predictions(0, i));
        //     double conf = output(pred, i);
        //     std::cout << "Sample " << i << ": " << conf << "\n";
        // }
        
        // Model evaluation
        double loss = mlp.compute_loss(test_labels, output);
        std::cout << "Test loss: " << loss << "\n";
        
        if (accuracy == 1.0 && loss < 0.1) {
            std::cout << "PERFECT SYSTEM\n";
        } else if (accuracy >= 0.9) {
            std::cout << "GOOD PERFORMANCE\n";
        } else {
            std::cout << "UNDERPERFORMING SYSTEM\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
