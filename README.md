🧠 Crack Detection via Custom MLP Neural Network
This project implements a lightweight machine learning pipeline from scratch (no ML frameworks) to classify concrete cracks in grayscale images using a custom-built Multilayer Perceptron (MLP) and Matrix library. It is optimized for educational and embedded use cases where fine control, low overhead, and high transparency are crucial.

📁 Project Structure
graphql
Copy
Edit
.
├── matrix.h / matrix.cpp         # Custom matrix class with safe access, activation, math ops
├── MLP.h / MLP.cpp               # Custom MLP implementation with forward/backward propagation
├── image_processor.h / .cpp     # Image loading, preprocessing, and dataset handling
├── main.cpp                      # Sample entrypoint (commented/test code)
├── crack_images.csv              # (Generated) flattened dataset
├── crack_labels.csv              # (Generated) corresponding labels
🏗️ Key Features
🧮 Matrix Library
Safe element access with error checking (operator() and no_bounds_check)

Matrix operations: multiply, add, subtract, scalar ops, transpose, Hadamard product

Activation functions: ReLU, Leaky ReLU, Softmax

Statistics: mean, sum, min, max, Frobenius norm

Gradient support: row-wise mean/std, element-wise subtraction, clipping

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

🧪 Example Dataset
To train the model:

bash
Copy
Edit
📂 ../datasets/
   ├── Cracked/
   │    ├── img1.jpg
   │    ├── img2.jpg
   └── NonCracked/
        ├── imgA.jpg
        ├── imgB.jpg
Cracked → label 1

NonCracked → label 0

🚀 Getting Started
Install dependencies:

Requires OpenCV

C++17 (for filesystem, chrono, etc.)

Compile:
Make
./build/bin/basicai
Make tests


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
bash
Copy
Edit
make
Builds all executables:

main – primary executable

basicai – secondary application

tests – test suite using GoogleTest

🚀 Run Executables
Run Main App
bash
Copy
Edit
./build/bin/main
Run BasicAI
bash
Copy
Edit
./build/bin/basicai
🧪 Running Tests
Run All Tests
bash
Copy
Edit
make test
or

bash
Copy
Edit
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
