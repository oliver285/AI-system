
#include "matrix.h"
#include <random>
#include <cmath>
class MLP {
    private:
        Matrix<10, 784> W1, dW1;
        Matrix<2, 10> W2, DW2;
        Matrix<10, 1> b1, db1;
        Matrix<2, 1> b2, db2;
        Matrix<10, 784> Z1, A1,DZ1;
        Matrix<2, 784> Z2, A2, DZ2;
    
        double alpha;
        uint8_t num_classes;
    
    public:
        void init_params() {
            double scale1 = std::sqrt(2.0 / 784);
            double scale2 = std::sqrt(2.0 / 10);
    
            W1 = Matrix<10, 784>::random().multiply_scalar(scale1);
            W2 = Matrix<2, 10>::random().multiply_scalar(scale2);
            b1 = Matrix<10, 1>::random().subtract_scalar(0.5);
            b2 = Matrix<2, 1>::random().subtract_scalar(0.5);
        }
    
        void forward_prop(const Matrix<784, 1>& X) {
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