#include"../include/MLP.h"


int main() {

    //Matrix unit tests
        //     test_relu();
        //     test_leaky_relu();
        //     test_relu_derivative();
        //     test_leaky_relu_derivative();
        //     test_softmax();
        //     test_matrix_multiplication();
        //     test_transpose();
            
        //     std::cout << "All tests passed!\n";
    
    
        // Test with small dataset
        const size_t input_size = 2;
        const size_t hidden_size = 3;
        const size_t output_size = 2;
        const size_t batch_size = 4;
    
        // Create MLP
        MLP mlp(input_size, hidden_size, output_size);
    
    //     Create sample input (2 features, 4 samples)
    //    Create sample input with variation
    Matrix X(input_size, batch_size);
    X(0,0) = 0.1; X(1,0) = 0.2;
    X(0,1) = 0.9; X(1,1) = 0.8;
    X(0,2) = 0.1; X(1,2) = 0.9;
    X(0,3) = 0.9; X(1,3) = 0.1;  // Fill with sample data
    double mean = X.mean();
    double std = 0.0;
    for (size_t i = 0; i < X.size(); i++) {
        std += pow(X.no_bounds_check(i) - mean, 2);
    }
    std = sqrt(std / X.size());
    X = X.subtract_scalar(mean).multiply_scalar(1.0/std);
        // Create sample labels
        Matrix Y(1, batch_size);
        Y(0,0) = 0; Y(0,1) = 1; Y(0,2) = 0; Y(0,3) = 1;
    
        // Train the network
        mlp.gradient_descent(X, Y, 1000, .1);
    
        // Test prediction
        Matrix output = mlp.forward_prop(X);
        Matrix predictions = mlp.get_predictions(output);
        
        std::cout << "\nFinal predictions:\n";
        predictions.print();
        
        double accuracy = mlp.get_accuracy(predictions, Y);
        std::cout << "Final accuracy: " << accuracy << "\n";
    
        std::cout << "\nPrediction confidence:\n";
    for(int i=0; i<batch_size; i++) {
        int pred = static_cast<int>(predictions(0, i));
        double conf = output(pred, i);
        std::cout << "Sample " << i << ": " << conf << "\n";
    }
    
    // Should add
    if(accuracy == 1.0 && mlp.compute_loss(Y, output) < 0.1) {
        std::cout << "PERFECT SYSTEM\n";
    } else if(accuracy == 1.0) {
        std::cout << "Correct predictions but LOW CONFIDENCE\n";
    } else {
        std::cout << "UNDERPERFORMING SYSTEM\n";
    }
    
        return 0;
    }
    