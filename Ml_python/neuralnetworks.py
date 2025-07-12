import os
import numpy as np
import cv2
import pandas as pd

# =======================
# Neural Network Functions
# =======================
# simple feedforward neural network (also known as a multilayer perceptron or MLP)
def init_params():
    W1 = np.random.randn(10, 784) * np.sqrt(2 / 784)
    W2 = np.random.randn(2, 10) * np.sqrt(2 / 10)
    b1 = np.random.randn(10, 1) - 0.5 
    b2 = np.random.randn(2, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z): return np.maximum(0, Z)
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    num_classes = 2  # Since we're doing binary classification
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y


def deriv_ReLU(Z): return Z > 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    Y = Y - Y.min()
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2): return np.argmax(A2, axis=0)
def get_accuracy(predictions, Y): return np.mean(predictions == Y)

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    decay_rate = 0.01
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        alpha = alpha * (1 / (1 + decay_rate * i))
        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration: {i}, Accuracy: {accuracy:.4f}")
    return W1, b1, W2, b2

def predict_new_image(image_path, W1, b1, W2, b2):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img_flatten = img.flatten().reshape(-1, 1)
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, img_flatten)
    prediction = get_predictions(A2)[0]
    label = "Cracked" if prediction == 1 else "Not Cracked"
    print(f"{image_path}: {label}")
    return label

# =======================
# Preprocessing Dataset
# =======================
def load_images_from_folder(folder, label, image_size=(28, 28)):
    data = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            img_flatten = img.flatten()
            data.append(np.insert(img_flatten, 0, label))
    return data

cracked = load_images_from_folder("datasets/testing/data/Cracked", label=1)
not_cracked = load_images_from_folder("datasets/testing/data/NonCracked", label=0)
all_data = np.array(cracked + not_cracked)
np.random.shuffle(all_data)
pd.DataFrame(all_data).to_csv("crack_dataset.csv", index=False)

# =======================
# Load + Split Dataset
# =======================
data = pd.read_csv('crack_dataset.csv').to_numpy()
np.random.shuffle(data)
m, n = data.shape

if m < 200:
    split_idx = m // 5  # 20% dev set if small
else:
    split_idx = 100

data_dev = data[:split_idx].T
Y_dev = data_dev[0].astype(int)
X_dev = data_dev[1:n]

data_train = data[split_idx:].T
Y_train = data_train[0].astype(int)
X_train = data_train[1:n]

# =======================
# Train Model
# =======================
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=100, alpha=0.1)

# Evaluate on Dev Set
_, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
accuracy_dev = get_accuracy(get_predictions(A2_dev), Y_dev)
print(f"Dev Set Accuracy: {accuracy_dev:.4f}")

# =======================
# Predict on One Image
# =======================
predict_new_image("datasets/testing/data/Cracked/sample1.jpg", W1, b1, W2, b2)
