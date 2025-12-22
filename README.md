# Crack Detection via Custom MLP Neural Network

Lightweight, **from-scratch** machine learning pipeline for classifying concrete surface cracks in grayscale images.  
No external ML frameworks (PyTorch/TensorFlow) — everything implemented manually for transparency, low overhead, and embedded readiness.

![Training Accuracy]<img width="1280" height="960" alt="accuracy_vs_epoch" src="https://github.com/user-attachments/assets/3f53c159-2ec5-41cb-bcdb-0617c746c952" />
  
*Train (blue) vs Test (orange) accuracy over 100 epochs — reaching ~87% test accuracy*

## Highlights of Custom Implementation

This project is built entirely from scratch for maximum control and embedded readiness:

- **Custom Matrix Class** — Core of the system: manual operator overloading, memory-efficient operations, no external dependencies.  
  → [View matrix.h / matrix.cpp](cpp/core/matrix.h) (key file — see in-place ops, transpose, multiplication)

- **MLP from Scratch** — Forward/backward pass, momentum optimizer, adaptive LR.  
  → [View mlp.h / mlp.cpp](cpp/ml/MLP.h)

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
