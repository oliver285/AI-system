#include "cnn.h"
#include <random>
#include <cmath>

CNN::CNN(size_t input_channels,
         const std::vector<size_t> &conv_filter_sizes,
         const std::vector<size_t> &conv_filter_counts,
         const std::vector<size_t> &pool_sizes,
         const std::vector<size_t> &strides)
    : input_channels(input_channels),
      conv_filter_sizes(conv_filter_sizes),
      conv_filter_counts(conv_filter_counts),
      pool_sizes(pool_sizes),
      strides(strides)
{

    initialize_weights();
}

void CNN::initialize_weights()
{
    conv_weights.clear();
    conv_biases.clear();

    for (size_t layer = 0; layer < conv_filter_sizes.size(); ++layer)
    {
        size_t input_chs = (layer == 0) ? input_channels : conv_filter_counts[layer - 1];
        size_t output_chs = conv_filter_counts[layer];
        size_t filter_size = conv_filter_sizes[layer];

        std::vector<std::vector<Matrix>> layer_weights;
        std::vector<Matrix> layer_biases;

        for (size_t out_ch = 0; out_ch < output_chs; ++out_ch)
        {
            std::vector<Matrix> channel_weights;
            for (size_t in_ch = 0; in_ch < input_chs; ++in_ch)
            {
                // Use YOUR random initialization
                Matrix filter = Matrix::random(filter_size, filter_size);
                // He initialization scaling
                float scale = std::sqrt(2.0f / (input_chs * filter_size * filter_size));
                filter.scale_inplace(scale);
                channel_weights.push_back(filter);
            }
            layer_weights.push_back(channel_weights);

            // Initialize bias
            Matrix bias(1, 1);
            bias.fill(0.1f); // Small positive bias
            layer_biases.push_back(bias);
        }

        conv_weights.push_back(layer_weights);
        conv_biases.push_back(layer_biases);
    }
}

std::vector<Matrix> CNN::forward(const std::vector<Matrix> &input, Error *err)
{
    std::vector<Matrix> current = input;
    layer_inputs.clear();
    layer_outputs.clear();

    // Process through each convolutional layer
    for (size_t layer = 0; layer < conv_filter_sizes.size(); ++layer)
    {
        layer_inputs.push_back(current);

        std::vector<Matrix> layer_output;
        size_t output_chs = conv_filter_counts[layer];
        size_t input_chs = current.size();

        // For each output channel
        for (size_t out_ch = 0; out_ch < output_chs; ++out_ch)
        {
            Matrix feature_map;
            bool first_channel = true;

            // Convolve across input channels
            for (size_t in_ch = 0; in_ch < input_chs; ++in_ch)
            {
                Matrix conv_result = Matrix::Convolve2D(
                    current[in_ch],
                    conv_weights[layer][out_ch][in_ch],
                    strides[layer],
                    err);

                if (err && *err != NO_ERROR)
                    return {};

                if (first_channel)
                {
                    feature_map = conv_result;
                    first_channel = false;
                }
                else
                {
                    feature_map = feature_map.add(conv_result, err);
                    if (err && *err != NO_ERROR)
                        return {};
                }
            }

            // Add bias and apply activation
            feature_map.add_inplace(conv_biases[layer][out_ch](0, 0));
            feature_map = feature_map.RELU();

            layer_output.push_back(feature_map);
        }

        current = layer_output;
        layer_outputs.push_back(current);

        // Apply pooling if specified for this layer
        if (layer < pool_sizes.size() && pool_sizes[layer] > 1)
        {
            // You would add your pooling implementation here
            current = apply_avg_pooling_forward(current,
                                                pool_sizes[layer], strides[layer], err); // apply_pooling(current, pool_sizes[layer], strides[layer], err);
        }
    }

    return current;
}

Matrix CNN::flatten_features(const std::vector<Matrix> &features)
{
    size_t total_size = 0;
    for (const auto &fm : features)
    {
        total_size += fm.row_count() * fm.col_count();
    }

    Matrix flattened(1, total_size);
    size_t current_idx = 0;

    for (const auto &fm : features)
    {
        for (size_t i = 0; i < fm.row_count(); ++i)
        {
            for (size_t j = 0; j < fm.col_count(); ++j)
            {
                flattened(0, current_idx++) = fm(i, j);
            }
        }
    }

    return flattened;
}

std::vector<Matrix> CNN::unflatten_features(const Matrix &flattened,
                                            size_t channels, size_t height, size_t width)
{
    std::vector<Matrix> unflattened;

    // Validate input size
    size_t total_elements = channels * height * width;
    if (flattened.size() != total_elements)
    {
        std::cerr << "Error: Flattened size " << flattened.size()
                  << " doesn't match expected " << total_elements << std::endl;
        return unflattened; // Return empty vector on error
    }

    size_t current_idx = 0;

    for (size_t channel = 0; channel < channels; ++channel)
    {
        Matrix temp(height, width); // Note: height rows, width columns

        for (size_t i = 0; i < height; ++i)
        {
            for (size_t j = 0; j < width; ++j)
            {
                // Correct indexing: row-major order within each channel
                temp(i, j) = flattened(0, current_idx);
                current_idx++;
            }
        }

        unflattened.push_back(temp);
    }

    return unflattened;
}

std::vector<Matrix> CNN::apply_max_pooling_backward(const std::vector<Matrix> &dL_dOutput,
                                                    const std::vector<Matrix> &pool_inputs,
                                                    const std::vector<Matrix> &pool_outputs,
                                                    size_t pool_size, size_t stride, Error *err)
{
    std::vector<Matrix> dL_dInput;

    for (size_t ch = 0; ch < dL_dOutput.size(); ++ch)
    {
        const Matrix &input_map = pool_inputs[ch];
        const Matrix &dOutput_map = dL_dOutput[ch];

        Matrix dInput_map(input_map.row_count(), input_map.col_count());
        dInput_map.fill(0.0f);

        // float scale = 1.0f / (pool_size * pool_size);
        // Iterate through output positions
        for (size_t i = 0; i < dOutput_map.row_count(); ++i)
        {
            for (size_t j = 0; j < dOutput_map.col_count(); ++j)
            {
                // float grad_value = dOutput_map(i, j) * scale;
                size_t max_i = i * stride;
                size_t max_j = j * stride;
                float maxval = input_map(max_i, max_j);

                // Distribute gradient to all positions in the pooling region
                for (size_t m = 0; m < pool_size; ++m)
                {
                    for (size_t n = 0; n < pool_size; ++n)
                    {

                        size_t current_i = i * stride + m;
                        size_t current_j = j * stride + n;
                        float current_val = input_map(current_i, current_j);
                        if (current_val > maxval)
                        {
                            // dInput_map(input_i, input_j) += grad_value;
                            max_i = current_i;
                            max_j = current_j;
                            maxval = current_val;
                        }
                    }
                }

                dInput_map(max_i, max_j) += dOutput_map(i, j);
            }
        }

        dL_dInput.push_back(dInput_map);
    }

    return dL_dInput;
}

std::vector<Matrix> CNN::apply_avg_pooling_backward(const std::vector<Matrix> &dL_dOutput,
                                                    size_t pool_size, size_t stride, Error *err)
{
    std::vector<Matrix> dL_dInput;

    // For each feature map in the gradient
    for (const auto &dOutput_map : dL_dOutput)
    {
        // Calculate input dimensions (reverse of pooling)
        size_t input_height = (dOutput_map.row_count() - 1) * stride + pool_size;
        size_t input_width = (dOutput_map.col_count() - 1) * stride + pool_size;

        Matrix dInput_map(input_height, input_width);
        dInput_map.fill(0.0f); // Initialize with zeros

        float scale = 1.0f / (pool_size * pool_size);

        // Iterate through output positions
        for (size_t i = 0; i < dOutput_map.row_count(); ++i)
        {
            for (size_t j = 0; j < dOutput_map.col_count(); ++j)
            {
                float grad_value = dOutput_map(i, j) * scale;

                // Distribute gradient to all positions in the pooling region
                for (size_t m = 0; m < pool_size; ++m)
                {
                    for (size_t n = 0; n < pool_size; ++n)
                    {
                        size_t input_i = i * stride + m;
                        size_t input_j = j * stride + n;

                        if (input_i < input_height && input_j < input_width)
                        {
                            dInput_map(input_i, input_j) += grad_value;
                        }
                    }
                }
            }
        }
        dL_dInput.push_back(dInput_map);
    }

    return dL_dInput;
}

// Output(i,j) = max(Input(m,n)) for m ∈ [i×stride, i×stride + pool_size)
//                           for n ∈ [j×stride, j×stride + pool_size)
std::vector<Matrix> CNN::apply_max_pooling_forward(const std::vector<Matrix> &input,
                                                   size_t pool_size, size_t stride, Error *err)
{
    std::vector<Matrix> output;

    for (size_t ch = 0; ch < input.size(); ++ch)
    {
        const Matrix &input_map = input[ch];
        size_t input_height = input_map.row_count();
        size_t input_width = input_map.col_count();

        // Calculate output dimensions
        size_t output_height = (input_height - pool_size) / stride + 1;
        size_t output_width = (input_width - pool_size) / stride + 1;

        Matrix output_map(output_height, output_width);

        for (size_t i = 0; i < output_height; ++i)
        {
            for (size_t j = 0; j < output_width; ++j)
            {
                float max_val = -999999999999.f;

                // Find max in pooling region
                for (size_t m = 0; m < pool_size; ++m)
                {
                    for (size_t n = 0; n < pool_size; ++n)
                    {
                        size_t input_i = i * stride + m;
                        size_t input_j = j * stride + n;
                        float current_val = input_map(input_i, input_j);
                        if (current_val > max_val)
                        {
                            max_val = current_val;
                        }
                    }
                }

                output_map(i, j) = max_val;
            }
        }

        output.push_back(output_map);
    }

    return output;
}

// Output(i,j) = (1/(pool_size²)) × ∑∑ Input(m,n)
//               for m ∈ [i×stride, i×stride + pool_size)
//               for n ∈ [j×stride, j×stride + pool_size)

std::vector<Matrix> CNN::apply_avg_pooling_forward(const std::vector<Matrix> &input,
                                                   size_t pool_size, size_t stride, Error *err)
{
    std::vector<Matrix> output;

    for (size_t ch = 0; ch < input.size(); ++ch)
    {
        const Matrix &input_map = input[ch];
        size_t input_height = input_map.row_count();
        size_t input_width = input_map.col_count();

        // Calculate output dimensions
        size_t output_height = (input_height - pool_size) / stride + 1;
        size_t output_width = (input_width - pool_size) / stride + 1;

        Matrix output_map(output_height, output_width);
        float scale = 1.0f / (pool_size * pool_size);

        for (size_t i = 0; i < output_height; ++i)
        {
            for (size_t j = 0; j < output_width; ++j)
            {
                float sum = 0.0f;

                // Sum over pooling region
                for (size_t m = 0; m < pool_size; ++m)
                {
                    for (size_t n = 0; n < pool_size; ++n)
                    {
                        size_t input_i = i * stride + m;
                        size_t input_j = j * stride + n;
                        sum += input_map(input_i, input_j);
                    }
                }

                output_map(i, j) = sum * scale;
            }
        }

        output.push_back(output_map);
    }

    return output;
}

// std::vector<Matrix> layer_output;
//     size_t output_chs = conv_filter_counts[layer];
//     size_t input_chs = current.size();

//     // For each output channel
//     for (size_t out_ch = 0; out_ch < output_chs; ++out_ch)
//     {
//         Matrix feature_map;
//         bool first_channel = true;

//         // Convolve across input channels
//         for (size_t in_ch = 0; in_ch < input_chs; ++in_ch)
//         {
//             Matrix conv_result = Matrix::Convolve2D(
//                 current[in_ch],
//                 conv_weights[layer][out_ch][in_ch],
//                 strides[layer],
//                 err);

std::vector<Matrix> CNN::backward_conv_input(const std::vector<Matrix> &dL_dOutput,
                                             size_t layer_idx, Error *err)
{
    size_t output_chs = conv_filter_counts[layer_idx];
    size_t input_chs = (layer_idx == 0) ? input_channels : conv_filter_counts[layer_idx - 1];

    std::vector<Matrix> dL_dInput(input_chs);

    // Initialize each input channel gradient
    for (size_t in_ch = 0; in_ch < input_chs; ++in_ch)
    {
        dL_dInput[in_ch] = Matrix(layer_inputs[layer_idx][in_ch].row_count(),
                                  layer_inputs[layer_idx][in_ch].col_count());
        dL_dInput[in_ch].fill(0.0f);
    }

    // Accumulate gradients
    for (size_t in_ch = 0; in_ch < input_chs; ++in_ch)
    {
        Matrix &input_grad = dL_dInput[in_ch];

        for (size_t out_ch = 0; out_ch < output_chs; ++out_ch)
        {
            Matrix rotated_filter = rot180(conv_weights[layer_idx][out_ch][in_ch]);
            Matrix conv_grad = Matrix::Convolve2D(dL_dOutput[out_ch], rotated_filter, 1, err);

            if (err && *err != NO_ERROR)
                return {};

            // Element-wise addition
            for (size_t i = 0; i < input_grad.row_count(); ++i)
            {
                for (size_t j = 0; j < input_grad.col_count(); ++j)
                {
                    input_grad(i, j) += conv_grad(i, j);
                }
            }
        }
    }

    return dL_dInput;
}

// ∂L/∂W = X ∗ ∂L/∂Y  (valid convolution)
// ∂L/∂b = ∑∑ ∂L/∂Y  (sum over all spatial dimensions)

// std::vector<std::vector<Matrix>> CNN::backward_conv_weights(const std::vector<Matrix> &dL_dOutput,
//                                                             const std::vector<Matrix> &layer_input,
//                                                             size_t layer_idx, Error *err)
// {

//     size_t output_chs = conv_filter_counts[layer_idx];
//     size_t input_chs = (layer_idx == 0) ? input_channels : conv_filter_counts[layer_idx - 1];
//     std::vector<std::vector<Matrix>> BONNIE(input_chs);
//     for (size_t in_ch1 = 0; in_ch1 < input_chs; ++in_ch1)
//         for (size_t in_ch = 0; in_ch < input_chs; ++in_ch)
//         {
//             BONNIE[in_ch1][in_ch] = Matrix(layer_inputs[layer_idx][in_ch].row_count(),
//                                            layer_inputs[layer_idx][in_ch].col_count());
//             BONNIE[in_ch1][in_ch].fill(0.0f);
//         }
//     // Compute ∂L/∂W and ∂L/∂b for current layer
//     // Returns {weight_gradients, bias_gradients}
//     for (size_t in_ch = 0; in_ch < input_chs; ++in_ch)

//         for (size_t out_ch = 0; out_ch < output_chs; ++out_ch)
//         {

//             // Matrix rotated_filter = rot180(conv_weights[layer_idx][out_ch][in_ch]);
//             Matrix weight_grad = Matrix::Convolve2D(dL_dOutput[out_ch], layer_input[in_ch], 1, err);

//             if (err && *err != NO_ERROR)
//                 return {};

//         }
// }
ConvGradients CNN::backward_conv_weights(const std::vector<Matrix> &dL_dOutput,
                                                            const std::vector<Matrix> &layer_input,
                                                            size_t layer_idx, Error *err)
{

    size_t output_chs = conv_filter_counts[layer_idx];
    size_t input_chs = layer_input.size();

    // Structure: result[0] = weight_grads, result[1] = bias_grads
    ConvGradients result;

    // 1. WEIGHT GRADIENTS: Same structure as conv_weights[layer_idx]
    //    weight_grads[out_ch][in_ch] = gradient for filter connecting in_ch to out_ch
  std::vector<std::vector<Matrix>> weight_grads(output_chs);
for (size_t out_ch = 0; out_ch < output_chs; ++out_ch) {
    weight_grads[out_ch].resize(input_chs);
    for (size_t in_ch = 0; in_ch < input_chs; ++in_ch) {
        weight_grads[out_ch][in_ch] = Matrix::Convolve2D(
            layer_input[in_ch],
            dL_dOutput[out_ch],
            1,  // optional if your function requires it
            err
        );

        if (err && *err != NO_ERROR)
            return {};
    }
}

result.weight_grads = weight_grads;

// result.push_back(weight_grads);


    // 2. BIAS GRADIENTS: One 1x1 matrix per output channel
    std::vector<Matrix> bias_grads;
    bias_grads.resize(output_chs);

    for (size_t out_ch = 0; out_ch < output_chs; ++out_ch)
    {
        bias_grads[out_ch] = Matrix(1, 1);
        float sum = 0.0f;
        // Sum all elements in this output channel's gradient
        const Matrix &grad = dL_dOutput[out_ch];
        for (size_t i = 0; i < grad.row_count(); ++i)
        {
            for (size_t j = 0; j < grad.col_count(); ++j)
            {
                sum += grad(i, j);
            }
        }

        bias_grads[out_ch](0, 0) = sum;
    }

    result.bias_grads = bias_grads;

    return result;
}

std::vector<std::vector<Matrix>> CNN::backward(const std::vector<Matrix> &input,
                                          const std::vector<Matrix> &dL_dOutput,
                                          float learning_rate, Error *err)
{
 // Store gradients for all layers
    std::vector<std::vector<Matrix>> layer_dInputs;

    // Gradient coming from the output (initially dL/dOutput)
    std::vector<Matrix> dL_dY = dL_dOutput;

 // Process through each convolutional layer
    for (size_t layer = 0; layer < conv_filter_sizes.size(); ++layer)
    {
       // Get stored activations (from forward)
        const auto &layer_input = layer_inputs[layer];
        const auto &layer_output = layer_outputs[layer];

        // === Step 1: Backprop through activation ===
        // ReLU’(x) = 1 if x > 0 else 0
        for (size_t ch = 0; ch < dL_dY.size(); ++ch){
        
  dL_dY[ch] = layer_output[ch].deriv_RELU();
}


      // === Step 2: Compute weight and bias gradients ===
        ConvGradients grads = backward_conv_weights(dL_dY, layer_input, layer, err);
        if (err && *err != NO_ERROR)
            return {};


     // === Step 3: Compute input gradients for previous layer ===
        std::vector<Matrix> dL_dX = backward_conv_input(dL_dY, layer, err);
        if (err && *err != NO_ERROR)
            return {};
// === Step 4: Update weights ===
        for (size_t out_ch = 0; out_ch < conv_weights[layer].size(); ++out_ch)
        {
            for (size_t in_ch = 0; in_ch < conv_weights[layer][out_ch].size(); ++in_ch)
            {

                Matrix temp1 = grads.weight_grads[out_ch][in_ch].scale(-learning_rate);
                conv_weights[layer][out_ch][in_ch].add_inplace(
                    grads.weight_grads[out_ch][in_ch],-learning_rate);
            }
            conv_biases[layer][out_ch].add_inplace(
                grads.bias_grads[out_ch],-learning_rate);
        }

        // === Step 5: Pooling backprop if applicable ===
        if (layer < pool_sizes.size() && pool_sizes[layer] > 1)
        {
            dL_dX = apply_avg_pooling_backward(
                dL_dX, pool_sizes[layer], strides[layer], err);
        }

        // === Step 6: Pass gradient to next iteration ===
        dL_dY = dL_dX;

        layer_dInputs.push_back(dL_dX);
    }

    std::reverse(layer_dInputs.begin(), layer_dInputs.end());
    return layer_dInputs;
}

    

