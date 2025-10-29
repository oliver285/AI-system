#include "../core/matrix.h"
#include "../ml/MLP.h"
#include "../vision/image_processor.h"
#include <fstream>

int main() {
    try {
        image_processor processor;
        const uint8_t IMG_WIDTH = 28, IMG_HEIGHT = 28;
        const size_t input_size = IMG_WIDTH * IMG_HEIGHT;

        // Load datasets
        Matrix cracked_images = processor.load_dataset("../datasets/testing/data/Cracked");
        Matrix cracked_labels = processor.load_labels("../datasets/testing/data/Cracked", 1);
        Matrix noncracked_images = processor.load_dataset("../datasets/testing/data/NonCracked");
        Matrix noncracked_labels = processor.load_labels("../datasets/testing/data/NonCracked", 0);
         std::vector<float> train_accs, test_accs;
    std::vector<float> train_losses, test_losses;


        // Validate dataset sizes
        if (cracked_images.col_count() != input_size || noncracked_images.col_count() != input_size) {
            std::cerr << "Error: Image dimensions do not match expected input size\n";
            return 1;
        }

        // Combine images and labels
        size_t total_samples = cracked_images.row_count() + noncracked_images.row_count();
        Matrix all_images(total_samples, input_size);
        Matrix all_labels(total_samples, 1);

        // Copy cracked data
        for (size_t i = 0; i < cracked_images.row_count(); ++i) {
            for (size_t j = 0; j < input_size; ++j)
                all_images(i, j) = cracked_images(i, j);
            all_labels(i, 0) = cracked_labels(i, 0);
        }

        // Copy non-cracked data
        size_t offset = cracked_images.row_count();
        for (size_t i = 0; i < noncracked_images.row_count(); ++i) {
            for (size_t j = 0; j < input_size; ++j)
                all_images(i + offset, j) = noncracked_images(i, j);
            all_labels(i + offset, 0) = noncracked_labels(i, 0);
        }

        // Shuffle combined dataset
        processor.shuffle_dataset(all_images, all_labels);

        // Split into train/test sets
        size_t train_size = total_samples * 0.8;
        size_t test_size = total_samples - train_size;

        Matrix train_images(train_size, input_size);
        Matrix train_labels(train_size, 1);
        Matrix test_images(test_size, input_size);
        Matrix test_labels(test_size, 1);

        // Copy training data
        for (size_t i = 0; i < train_size; ++i) {
            for (size_t j = 0; j < input_size; ++j)
                train_images(i, j) = all_images(i, j);
            train_labels(i, 0) = all_labels(i, 0);
        }

        // Copy test data
        for (size_t i = 0; i < test_size; ++i) {
            for (size_t j = 0; j < input_size; ++j)
                test_images(i, j) = all_images(i + train_size, j);
            test_labels(i, 0) = all_labels(i + train_size, 0);
        }

        // Create MLP
         const size_t hidden_size2 = 128;
        const size_t hidden_size = 256;
        const size_t output_size = 2;
        MLP mlp(input_size, hidden_size,hidden_size2, output_size);

        const size_t batch_size = 64;
        size_t num_batches = train_size / batch_size + (train_size % batch_size != 0 ? 1 : 0);

        // Training loop
        for (size_t epoch = 0; epoch < 100; ++epoch) {
            std::cout << "Epoch " << epoch + 1 << "/" << 100 << "\n";

            // Shuffle training data at the start of each epoch
            processor.shuffle_dataset(train_images, train_labels);


     for (size_t start = 0; start < train_size; start += batch_size) {
    size_t current_batch_size = std::min(batch_size, train_size - start);

    Matrix batch_images(current_batch_size, input_size);
    Matrix batch_labels(current_batch_size, 1);

    for (size_t i = 0; i < current_batch_size; ++i) {
        for (size_t j = 0; j < input_size; ++j)
            batch_images(i, j) = train_images(start + i, j);
        batch_labels(i, 0) = train_labels(start + i, 0);
    }

    // Train step
    Matrix batch_images_T = batch_images.transpose();
    Matrix batch_output = mlp.forward_prop(batch_images_T);
    mlp.back_prop(batch_images_T, batch_labels.transpose()); // depends on your implementation
    mlp.update_params(0.01f);
}


            // Evaluate training performance
            Matrix train_output = mlp.forward_prop(train_images.transpose());
            Matrix train_preds = mlp.get_predictions(train_output);
            
            // Convert train_labels to row vector for accuracy calculation
            Matrix train_labels_row = train_labels.transpose();
            float train_acc = mlp.get_accuracy(train_preds, train_labels_row);
            float train_loss = mlp.cross_entropy_loss(train_output, train_labels_row);

        train_accs.push_back(train_acc);
        train_losses.push_back(train_loss);

            std::cout << "Epoch " << epoch + 1
                      << " | Train Loss: " << train_loss
                      << " | Train Acc: " << train_acc << "\n";

            // Optional: Evaluate on test set periodically
            // if ((epoch + 1) % 10 == 0) {
                Matrix test_output = mlp.forward_prop(test_images.transpose());
                Matrix test_preds = mlp.get_predictions(test_output);
                Matrix test_labels_row = test_labels.transpose();
                float test_acc = mlp.get_accuracy(test_preds, test_labels_row);
                float test_loss = mlp.cross_entropy_loss(test_output, test_labels_row);
         test_accs.push_back(test_acc);
        test_losses.push_back(test_loss);
                
                std::cout << "Epoch " << epoch + 1
                          << " | Test Loss: " << test_loss
                          << " | Test Acc: " << test_acc << "\n";
            // }
        }
    // --- Save to CSV ---
    std::ofstream file("training_log.csv");
    file << "epoch,train_acc,test_acc,train_loss,test_loss\n";
    for (size_t i = 0; i < train_accs.size(); ++i) {
        file << i+1 << ","
             << train_accs[i] << ","
             << test_accs[i] << ","
             << train_losses[i] << ","
             << test_losses[i] << "\n";
    }
    file.close();

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
