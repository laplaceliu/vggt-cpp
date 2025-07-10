// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

#pragma once

#include <Eigen/Dense>

class SwiGLUFFN {
public:
    SwiGLUFFN(int in_features,
              int hidden_features = -1,
              int out_features = -1,
              bool bias = true);

    // Forward pass
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);

private:
    // SiLU activation function
    static Eigen::MatrixXf silu(const Eigen::MatrixXf& x);

    int in_features_;
    int hidden_features_;
    int out_features_;
    bool bias_;

    // Weight matrices
    Eigen::MatrixXf w12_;  // Input to hidden (2*hidden_features)
    Eigen::MatrixXf w3_;   // Hidden to output
    Eigen::VectorXf b12_;  // Bias for w12
    Eigen::VectorXf b3_;   // Bias for w3
};
