#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>

using namespace std;

// Activation function: Sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of Sigmoid function for backpropagation
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

// A class representing a basic feedforward neural network
class NeuralNetwork {
public:
    vector<vector<double>> weights1; // Weights between input layer and hidden layer
    vector<vector<double>> weights2; // Weights between hidden layer and output layer
    vector<double> hiddenLayer;      // Hidden layer values
    vector<double> outputLayer;      // Output layer values

    NeuralNetwork(int inputSize, int hiddenSize, int outputSize);
    void initializeWeights();
    vector<double> forward(vector<double> &input);
    void train(vector<vector<double>> &trainingData, vector<vector<double>> &trainingLabels, int epochs, double learningRate);

private:
    double loss(double output, double target);
    void backpropagation(vector<double> &input, vector<double> &target, double learningRate);
};

// Constructor to initialize the network structure
NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
    hiddenLayer.resize(hiddenSize);
    outputLayer.resize(outputSize);
    
    // Initialize weights randomly
    weights1.resize(inputSize, vector<double>(hiddenSize));
    weights2.resize(hiddenSize, vector<double>(outputSize));
    initializeWeights();
}

// Function to initialize weights randomly using normal distribution
void NeuralNetwork::initializeWeights() {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0, 1); // Mean = 0, Std Dev = 1

    for (auto &row : weights1)
        for (auto &w : row)
            w = dist(gen);

    for (auto &row : weights2)
        for (auto &w : row)
            w = dist(gen);
}

// Forward pass: Computes output by passing data through the network
vector<double> NeuralNetwork::forward(vector<double> &input) {
    // Calculate hidden layer values
    for (int i = 0; i < hiddenLayer.size(); ++i) {
        hiddenLayer[i] = 0.0;
        for (int j = 0; j < input.size(); ++j) {
            hiddenLayer[i] += input[j] * weights1[j][i];
        }
        hiddenLayer[i] = sigmoid(hiddenLayer[i]);
    }

    // Calculate output layer values
    for (int i = 0; i < outputLayer.size(); ++i) {
        outputLayer[i] = 0.0;
        for (int j = 0; j < hiddenLayer.size(); ++j) {
            outputLayer[i] += hiddenLayer[j] * weights2[j][i];
        }
        outputLayer[i] = sigmoid(outputLayer[i]);
    }
    return outputLayer;
}

// Loss function: Mean Squared Error (MSE)
double NeuralNetwork::loss(double output, double target) {
    return 0.5 * pow((output - target), 2);
}

// Backpropagation function: Adjusts weights based on error
void NeuralNetwork::backpropagation(vector<double> &input, vector<double> &target, double learningRate) {
    // Calculate output layer error
    vector<double> outputErrors(outputLayer.size());
    for (int i = 0; i < outputLayer.size(); ++i) {
        outputErrors[i] = (target[i] - outputLayer[i]) * sigmoid_derivative(outputLayer[i]);
    }

    // Calculate hidden layer error
    vector<double> hiddenErrors(hiddenLayer.size(), 0.0);
    for (int i = 0; i < hiddenLayer.size(); ++i) {
        for (int j = 0; j < outputErrors.size(); ++j) {
            hiddenErrors[i] += outputErrors[j] * weights2[i][j];
        }
        hiddenErrors[i] *= sigmoid_derivative(hiddenLayer[i]);
    }

    // Update weights between hidden and output layers
    for (int i = 0; i < weights2.size(); ++i) {
        for (int j = 0; j < weights2[i].size(); ++j) {
            weights2[i][j] += learningRate * outputErrors[j] * hiddenLayer[i];
        }
    }

    // Update weights between input and hidden layers
    for (int i = 0; i < weights1.size(); ++i) {
        for (int j = 0; j < weights1[i].size(); ++j) {
            weights1[i][j] += learningRate * hiddenErrors[j] * input[i];
        }
    }
}

// Training function: Trains the network using backpropagation
void NeuralNetwork::train(vector<vector<double>> &trainingData, vector<vector<double>> &trainingLabels, int epochs, double learningRate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < trainingData.size(); ++i) {
            forward(trainingData[i]);
            backpropagation(trainingData[i], trainingLabels[i], learningRate);
        }
        cout << "Epoch " << epoch + 1 << " completed." << endl;
    }
}

int main() {
    // Example: 2 input neurons, 3 hidden neurons, and 1 output neuron
    NeuralNetwork nn(2, 3, 1);

    // Dummy data: Training inputs and expected outputs
    vector<vector<double>> trainingData = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> trainingLabels = {{0}, {1}, {1}, {0}}; // XOR problem

    nn.train(trainingData, trainingLabels, 1000, 0.1);

    // Test forward pass
    vector<double> output = nn.forward(trainingData[0]);
    cout << "Output: " << output[0] << endl;

    return 0;
}
