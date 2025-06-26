
#include "matrix.h"
#include <random>
#include <cmath>
class MLP {
    private:
        // Fixed size matrices (weights)
        Matrix<10, 784> W1, dW1;
        Matrix<2, 10> W2, dW2;
        Matrix<10, 1> b1;
        Matrix<2, 1> b2;
        
        // Dynamic size matrices (activations - batch size determined at runtime)
        std::vector<Matrix<10>> Z1, A1, dZ1;  // For multiple batches
        std::vector<Matrix<2>> Z2, A2, dZ2;
        
        uint8_t num_classes;
        int current_batch_size;
    
    public:
        // Initialize with maximum expected batch size
        void init_activations(int max_batch_size) {
            current_batch_size = max_batch_size;
            Z1.resize(max_batch_size);
            A1.resize(max_batch_size);
            dZ1.resize(max_batch_size);
            Z2.resize(max_batch_size);
            A2.resize(max_batch_size);
            dZ2.resize(max_batch_size);
        }
        void init_params() {
            double scale1 = std::sqrt(2.0 / 784);
            double scale2 = std::sqrt(2.0 / 10);
    
            W1 = Matrix<10, 784>::random().multiply_scalar(scale1);
            W2 = Matrix<2, 10>::random().multiply_scalar(scale2);
            b1 = Matrix<10, 1>::random().subtract_scalar(0.5);
            b2 = Matrix<2, 1>::random().subtract_scalar(0.5);
        }
    
        void forward_prop(const Matrix<784, m>& X) {
            Z1 = Matrix<10, 784>::multiply(W1, X) + b1;
            A1 = Z1.RELU();
            Z2 = Matrix<2, 10>::multiply(W2, A1) + b2;
            A2 = Matrix<2, 1>::softmax(Z2);

        }

        Matrix<2, 784> one_hot(const Matrix<784, 2>& Y){
        num_classes=2;

       Matrix<2, 784> one_hot_Y;

       for(int i=0;i<one_hot_Y.col;i+=2){

        one_hot_Y[0][i]=1;
        one_hot_Y[1][i]=1;

       }

       return one_hot_Y;



        }

        Matrix<2, 784>  deriv_ReLU() {
            for (uint16_t i = 0; i < rows; ++i)
                for (uint16_t j = 0; j < cols; ++j)
                    if (data[i][j] < 0)
                        data[i][j] = 0;
        }


        void back_prop(const Matrix<784, 1>& X, const Matrix<2, 1>& Y) {
            // Step 1: Convert Y to one-hot encoding (if needed)
            Matrix<2, 1> one_hot_Y;
            one_hot_Y.fill(0);
            one_hot_Y.data[static_cast<int>(Y.data[0][0])][0] = 1;  // assumes Y has a single class label (0 or 1)
        
            // Step 2: dZ2 = A2 - one_hot_Y
            DZ2 = Matrix<2, 1>::Sub_Matrix(A2, one_hot_Y);
        
            // Step 3: dW2 = dZ2 * A1^T
            dW2 = Matrix<2, 10>::multiply(DZ2, A1.transpose());
        
            // Step 4: db2 = sum of dZ2 (per neuron)
            db2 = DZ2; // for batch size 1, db2 is just dZ2
        
            // Step 5: dZ1 = (W2^T * dZ2) * ReLU'(Z1)
            Matrix<10, 1> dZ1_linear = Matrix<10, 2>::multiply(W2.transpose(), DZ2);
            Matrix<10, 1> dZ1_relu = Z1;
            dZ1_relu.deriv_ReLU();  // in-place ReLU'
            DZ1 = Matrix<10, 1>::multiply(dZ1_linear, dZ1_relu); // element-wise multiply
        
            // Step 6: dW1 = dZ1 * X^T
            dW1 = Matrix<10, 784>::multiply(DZ1, X.transpose());
        
            // Step 7: db1 = sum of dZ1 (per neuron)
            db1 = DZ1; // batch size 1
        
            // Optional: scale by learning rate and batch size if needed
        }

void update_params(int alpha){
W1.subtract_scalar(dW1.multiply_scalar(alpha));
b1.subtract_scalar(db1.multiply_scalar(alpha));
W2.subtract_scalar(dW2.multiply_scalar(alpha));
b2.subtract_scalar(db2.multiply_scalar(alpha));


}

Matrix<1, 784> get_predictions(const Matrix<2, 784>& A) {
    Matrix<1, 784> argmax;

    for (uint16_t col = 0; col < 784; ++col) {
        double max_val = A.data[0][col];
        int max_idx = 0;

        for (uint16_t row = 1; row < 2; ++row) {
            if (A.data[row][col] > max_val) {
                max_val = A.data[row][col];
                max_idx = row;
            }
        }

        argmax.data[0][col] = max_idx;
    }

    return argmax;
}

double get_accuracy(const Matrix<1, 784>& A, const Matrix<1, 784>& B) {
    double accuracy = 0;

    for (uint16_t i = 0; i < 784; ++i) {
        if (A.data[0][i] == B.data[0][i]) {
            accuracy++;
        }
    }

    return accuracy / 784.0;
}

void gradient_descent(){



}
// def gradient_descent(X, Y, iterations, alpha):
//     W1, b1, W2, b2 = init_params()
//     decay_rate = 0.01
//     for i in range(iterations):
//         Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
//         dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
//         W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
//         alpha = alpha * (1 / (1 + decay_rate * i))
//         if i % 10 == 0:
//             predictions = get_predictions(A2)
//             accuracy = get_accuracy(predictions, Y)
//             print(f"Iteration: {i}, Accuracy: {accuracy:.4f}")
//     return W1, b1, W2, b2

// def predict_new_image(image_path, W1, b1, W2, b2):
//     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
//     img = cv2.resize(img, (28, 28))
//     img_flatten = img.flatten().reshape(-1, 1)
//     _, _, _, A2 = forward_prop(W1, b1, W2, b2, img_flatten)
//     prediction = get_predictions(A2)[0]
//     label = "Cracked" if prediction == 1 else "Not Cracked"
//     print(f"{image_path}: {label}")
//     return label

// # =======================
// # Preprocessing Dataset
// # =======================
// def load_images_from_folder(folder, label, image_size=(28, 28)):
//     data = []
//     for filename in os.listdir(folder):
//         img_path = os.path.join(folder, filename)
//         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
//         if img is not None:
//             img = cv2.resize(img, image_size)
//             img_flatten = img.flatten()
//             data.append(np.insert(img_flatten, 0, label))
//     return data

// cracked = load_images_from_folder("datasets/testing/data/Cracked", label=1)
// not_cracked = load_images_from_folder("datasets/testing/data/NonCracked", label=0)
// all_data = np.array(cracked + not_cracked)
// np.random.shuffle(all_data)
// pd.DataFrame(all_data).to_csv("crack_dataset.csv", index=False)

// # =======================
// # Load + Split Dataset
// # =======================
// data = pd.read_csv('crack_dataset.csv').to_numpy()
// np.random.shuffle(data)
// m, n = data.shape

// if m < 200:
//     split_idx = m // 5  # 20% dev set if small
// else:
//     split_idx = 100

// data_dev = data[:split_idx].T
// Y_dev = data_dev[0].astype(int)
// X_dev = data_dev[1:n]

// data_train = data[split_idx:].T
// Y_train = data_train[0].astype(int)
// X_train = data_train[1:n]

// # =======================
// # Train Model
// # =======================
// W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=100, alpha=0.1)

// # Evaluate on Dev Set
// _, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
// accuracy_dev = get_accuracy(get_predictions(A2_dev), Y_dev)
// print(f"Dev Set Accuracy: {accuracy_dev:.4f}")


        

    };


    




// BLA::Matrix<3, 3, float> jacobian(BLA::Matrix<3,1, float> p, 
//                                   BLA::Matrix<3,1, float> tether_lengths, 
//                                   BLA::Matrix<3, 3, float> teth_anchor, 
//                                   BLA::Matrix<3, 3, float> offset) {
//     BLA::Matrix<3, 3, float> J;
//     double h = 1e-5;
//     BLA::Matrix<3,1, float> p1, f1, f2;

//     for (int i = 0; i < 3; i++) {
//         p1 = p;
//         p1(i) += h;
//         f1 = equations(p1, teth_anchor, offset, tether_lengths);
        
//         p1(i) -= 2*h;
//         f2 = equations(p1, teth_anchor, offset, tether_lengths);

//         for (int j = 0; j < 3; j++) {
//             J(j, i) = (f1(j) - f2(j)) / (2 * h);
//         }
//     }
//     return J;
// }