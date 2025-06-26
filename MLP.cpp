
#include "matrix.h"
#include <random>
#include <cmath>

class MLP{
private:
Matrix W1<10,784>, W2<2,10> b1<10,1>, b2<2,1>, Z1<10,784>, Z2<2,10>, A1<10,784>,A2<2,10>,DZ1<2,10>,DZ2<2,10>;
double alpha,X,Y;
uint8_t num_classes;
public:




void init_params(){
    int scale1 = sqrt(2/784);
    int scale2 = sqrt(2/10);
W1.random()*scale1;
W2.random()*scale2;
b1.random().subtract(.5);
b2.random().subtract(.5);


}


}



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