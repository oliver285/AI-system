#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#include "matrix.h"
#include <cmath>
#include <limits>

namespace MathUtils {

// Numerically stable softmax
inline Matrix softmax(const Matrix& m) {
    Matrix result(m.rows(), m.cols());
    for(size_t j = 0; j < m.cols(); j++) {
        double max_val = -std::numeric_limits<double>::max();
        for(size_t i = 0; i < m.rows(); i++) {
            if(m(i, j) > max_val) max_val = m(i, j);
        }
        
        double sum = 0.0;
        for(size_t i = 0; i < m.rows(); i++) {
            result(i, j) = std::exp(m(i, j) - max_val);
            sum += result(i, j);
        }
        
        for(size_t i = 0; i < m.rows(); i++) {
            result(i, j) /= sum;
        }
    }
    return result;
}

// Fletcher-16 checksum (useful for data validation)
inline uint16_t fletcher16(const uint8_t* data, size_t len) {
    uint16_t sum1 = 0, sum2 = 0;
    for(size_t i = 0; i < len; i++) {
        sum1 = (sum1 + data[i]) % 255;
        sum2 = (sum2 + sum1) % 255;
    }
    return (sum2 << 8) | sum1;
}

} // namespace MathUtils

#endif