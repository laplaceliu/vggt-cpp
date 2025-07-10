// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

#include "mlp.h"

namespace vggt {
namespace layers {

MlpImpl::MlpImpl(
    int64_t in_features,
    int64_t hidden_features,
    int64_t out_features,
    const torch::nn::Functional& act_layer,
    double drop,
    bool bias)
    : in_features_(in_features),
      hidden_features_(hidden_features == -1 ? in_features : hidden_features),
      out_features_(out_features == -1 ? in_features : out_features),
      act(act_layer),
      drop(torch::nn::DropoutOptions(drop)) {
    // Register layers
    fc1 = register_module("fc1",
        torch::nn::Linear(torch::nn::LinearOptions(in_features_, hidden_features_).bias(bias)));
    fc2 = register_module("fc2",
        torch::nn::Linear(torch::nn::LinearOptions(hidden_features_, out_features_).bias(bias)));
}

torch::Tensor MlpImpl::forward(const torch::Tensor& x) {
    x = fc1->forward(x);
    x = act->forward(x);
    x = drop->forward(x);
    x = fc2->forward(x);
    x = drop->forward(x);
    return x;
}

} // namespace layers
} // namespace vggt
