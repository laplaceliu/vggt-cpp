#include "swiglu_ffn.h"
#include <cmath>
#include <stdexcept>

using namespace Eigen;

SwiGLUFFN::SwiGLUFFN(int in_features, int hidden_features, int out_features, bool bias)
    : in_features_(in_features),
      hidden_features_(hidden_features == -1 ? in_features : hidden_features),
      out_features_(out_features == -1 ? in_features : out_features),
      bias_(bias) {

    // Initialize weights
    w12_ = MatrixXf::Random(in_features_, 2 * hidden_features_);
    w3_ = MatrixXf::Random(hidden_features_, out_features_);

    // Initialize biases if needed
    if (bias_) {
        b12_ = VectorXf::Random(2 * hidden_features_);
        b3_ = VectorXf::Random(out_features_);
    }
}

MatrixXf SwiGLUFFN::silu(const MatrixXf& x) {
    // SiLU activation: x * sigmoid(x)
    return x.array() * (1.0f / (1.0f + (-x.array()).exp()));
}

MatrixXf SwiGLUFFN::forward(const MatrixXf& x) {
    if (x.cols() != in_features_) {
        throw std::invalid_argument("Input dimension mismatch");
    }

    // First linear transformation
    MatrixXf x12 = x * w12_;
    if (bias_) {
        x12.rowwise() += b12_.transpose();
    }

    // Split into two parts and apply SwiGLU
    int batch_size = x.rows();
    MatrixXf x1 = x12.block(0, 0, batch_size, hidden_features_);
    MatrixXf x2 = x12.block(0, hidden_features_, batch_size, hidden_features_);

    MatrixXf hidden = silu(x1).array() * x2.array();

    // Second linear transformation
    MatrixXf output = hidden * w3_;
    if (bias_) {
        output.rowwise() += b3_.transpose();
    }

    return output;
}
