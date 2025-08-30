🧠 Crack Detection via Custom MLP Neural Network
This project implements a lightweight machine learning pipeline from scratch (no ML frameworks) to classify concrete cracks in grayscale images using a custom-built Multilayer Perceptron (MLP) and Matrix library. It is optimized for educational and embedded use cases where fine control, low overhead, and high transparency are crucial.

🤖 MLP Class
Two-layer MLP (input → hidden → output) with:

He initialization

Leaky ReLU activation

Softmax output layer

Cross-entropy loss

Backpropagation with momentum

Adaptive learning rate & early stopping

Supports binary & multi-class classification

🖼️ Image Processor
Loads images in .jpg or .png format

Preprocesses to 28x28 grayscale, normalizes pixels to [0,1]

Converts to flattened Matrix format

Can label folders as classes (e.g., Cracked, NonCracked)

Can shuffle datasets and save to CSV

🚀 Getting Started
Install dependencies:

Requires OpenCV

C++17 (for filesystem, chrono, etc.)

Compile:

Make

./build/bin/basicai

Make run_tests


🧠 Training
Uses forward pass, loss computation, backpropagation, and momentum update

Includes:

Adaptive learning rate with decay

Early stopping based on no improvement

Periodic diagnostic printing

📈 Outputs
crack_images.csv: flattened, preprocessed image data

crack_labels.csv: corresponding class labels

Console output shows:

Iteration, loss, accuracy

Gradient norms (dW1, dW2)

Parameter ranges (W1, b1, etc.)

✅ Default (build all)

Builds all executables:

basicai – primary executable

tests – test suite using GoogleTest

🚀 Run Executables
./build/bin/basicai


🧪 Running Tests
Run All Tests


make test
or
make run_tests

🛡️ Error Handling
All matrix operations return meaningful messages or fallback states on:

Index out-of-bounds

Dimension mismatch

Invalid probability distributions

📌 Notes
No external ML libraries are used.

All math, activation, and training logic is implemented from scratch.

Designed for transparency, modifiability, and low-level understanding.

📤 Contact
For questions or suggestions, feel free to reach out to the developer
