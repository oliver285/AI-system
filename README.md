# AI-system

AI-System is a lightweight neural network trained on the Kaggle Cracked/Non-Cracked surface image dataset. The current version is implemented in pure Python using NumPy and OpenCV, and achieves up to 70% accuracy on the validation set.

This project is a proof of concept for an embedded, vision-based crack detection system that will eventually be ported to C/C++ for deployment on microcontrollers.

🚀 Features
Trained using custom NumPy-based neural network (no TensorFlow or PyTorch)

Input: grayscale 28×28 surface images

Output: Binary classification — Cracked or Not Cracked

Uses softmax and one-hot encoding for output layer

Preprocessing pipeline built with OpenCV

Real-time single-image prediction function

Clean training/inference separation
