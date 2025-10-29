// cnn.h
#ifndef CNN_H
#define CNN_H

#include "../core/matrix.h"
#include <vector>

class CNN {
private:
    size_t input_channels;
    std::vector<size_t> conv_filter_sizes;
    std::vector<size_t> conv_filter_counts;
    std::vector<size_t> pool_sizes;
    std::vector<size_t> strides;
    
    // Weight storage: [layer][output_ch][input_ch][filter_matrix]
    std::vector<std::vector<std::vector<Matrix>>> conv_weights;
    std::vector<std::vector<Matrix>> conv_biases;
    
    // For backpropagation
    std::vector<std::vector<Matrix>> layer_inputs;
    std::vector<std::vector<Matrix>> layer_outputs;
    std::vector<std::vector<Matrix>> layer_conv_outputs; // Before activation

public:
    CNN(size_t input_channels, 
         const std::vector<size_t>& conv_filter_sizes,
         const std::vector<size_t>& conv_filter_counts,
         const std::vector<size_t>& pool_sizes,
         const std::vector<size_t>& strides);
    
    void initialize_weights();
    std::vector<Matrix> forward(const std::vector<Matrix>& input, Error* err = nullptr);
    
    // Pooling operations

std::vector<Matrix> apply_max_pooling_backward(const std::vector<Matrix>& dL_dOutput,
                                                   const std::vector<Matrix>& pool_inputs,
                                                   const std::vector<Matrix>& pool_outputs,
                                                   size_t pool_size, size_t stride, Error* err);
   std::vector<Matrix> apply_avg_pooling_backward(const std::vector<Matrix>& dL_dOutput, 
                                                   size_t pool_size, size_t stride, Error* err); 
   std::vector<Matrix> apply_max_pooling_forward(const std::vector<Matrix>& input, 
                                         size_t pool_size, size_t stride, Error* err);

 std::vector<Matrix> apply_avg_pooling_forward(const std::vector<Matrix>& input,
                                         size_t pool_size, size_t stride, Error* err);
    
    // Backward pass
    std::vector<std::vector<Matrix>> backward(const std::vector<Matrix>& input,
                                             const std::vector<Matrix>& dL_doutput,
                                             float learning_rate, Error* err);
    
    // Utility functions
    Matrix flatten_features(const std::vector<Matrix>& features);
    std::vector<Matrix> unflatten_features(const Matrix& flattened, 
                                          size_t channels, size_t height, size_t width);
    
std::vector<Matrix> backward_conv_input(const std::vector<Matrix>& dL_dOutput,
                                           size_t layer_idx, Error* err);
    

std::vector<std::vector<Matrix>> backward_conv_weights(const std::vector<Matrix>& dL_dOutput,
                                                          const std::vector<Matrix>& layer_input,
                                                          size_t layer_idx, Error* err);
    // Getters
    const auto& get_weights() const { return conv_weights; }
    const auto& get_biases() const { return conv_biases; }
};

#endif