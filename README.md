# Crack Detection via Custom MLP Neural Network

Lightweight, **from-scratch** machine learning pipeline for classifying concrete surface cracks in grayscale images.  
No external ML frameworks (PyTorch/TensorFlow) — everything implemented manually for transparency, low overhead, and embedded readiness.

![Training Accuracy]<img width="1280" height="960" alt="accuracy_vs_epoch" src="https://github.com/user-attachments/assets/3f53c159-2ec5-41cb-bcdb-0617c746c952" />
  
*Train (blue) vs Test (orange) accuracy over 100 epochs — reaching ~87% test accuracy*

## Highlights of Custom Implementation

This project is built entirely from scratch for maximum control and embedded readiness:

## Highlights of Custom Implementation

This project is built entirely from scratch for maximum control and embedded readiness:

- **Custom Convolutional Neural Network** — Full forward/backward pass with convolution, pooling, ReLU, and gradient computation.  
  → [View cnn.h / cnn.cpp](cpp/CNN/cnn.h) (key file — see Convolve2D, pooling, backprop)

- **Custom Matrix Class** — Core engine: manual operator overloading, in-place ops, memory-efficient.  
  → [View matrix.h / matrix.cpp](cpp/core/matrix.h)

- **MLP Classifier** — Three-layer perceptron with **Adam optimizer** (bias-corrected first/second moments), Leaky ReLU activation, momentum, and adaptive learning rate decay.  
  → [View mlp.h / mlp.cpp](cpp/ml/MLP.h) (key file — see full Adam implementation in update_params, backprop, softmax/cross-entropy)

- **Image Pipeline** — OpenCV loading → preprocess → flatten to custom matrix format.  
  → [View image_processor.h/cpp](cpp/vision/image_processor.h)
## Key Metrics
- **Test Accuracy**: 87.2% (peak)
- **Train Accuracy**: 88.5% (peak)
- **Inference Time**: < 10 ms on desktop (28×28 image, single forward pass)
- **Model Size**: ~150 k parameters (fully controllable)
- **No external dependencies** beyond OpenCV for image loading

## Features
- Custom **Matrix** class with operator overloading (addition, multiplication, transpose)
- Three-layer MLP:
  - He initialization
  - Leaky ReLU hidden activation
  - Softmax output
  - Cross-entropy loss
- Training loop with:
  - Momentum optimizer
  - Adaptive learning rate decay
  - Early stopping
- Image pipeline:
  - Load .jpg/.png → resize to 28×28 → normalize [0,1] → flatten to column vector
  - Automatic folder-based dataset loading and labeling
  - Shuffle + CSV export

## Why From Scratch?
- Full control over every operation — ideal for understanding and future embedded deployment
- Zero runtime overhead from frameworks
- Easily portable to microcontrollers (planned STM32/Jetsons target)

## Quick Start
```bash
# Build
make

# Run training/classification
./build/bin/basicai

# Run tests
make test
