/**
 * @file mlp.cpp
 * @brief Implementation of Multi-Layer Perceptron for vision transformers
 */

#include "mlp.h"

namespace vggt {
namespace layers {

MlpImpl::MlpImpl(
    int64_t in_features,
    int64_t hidden_features,
    bool bias,
    double drop
) {
    // Set default values if not provided
    int64_t out_features = in_features;
    hidden_features = hidden_features > 0 ? hidden_features : in_features;
    
    // Initialize layers
    fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(in_features, hidden_features).bias(bias)));
    act = register_module("act", torch::nn::GELU());
    fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(hidden_features, out_features).bias(bias)));
    drop_ = register_module("drop", torch::nn::Dropout(torch::nn::DropoutOptions(drop)));
}

MlpImpl::MlpImpl(const MlpOptions& options) {
    int64_t in_features = options.in_features;
    int64_t hidden_features = options.hidden_features;
    int64_t out_features = options.out_features;
    bool bias = options.bias_;
    double drop = options.drop_;
    
    // Initialize layers
    fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(in_features, hidden_features).bias(bias)));
    
    // Use custom activation layer if provided, otherwise use GELU
    if (options.act_layer_.has_value()) {
        act = register_module("act", options.act_layer_.value());
    } else {
        act = register_module("act", torch::nn::GELU());
    }
    
    fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(hidden_features, out_features).bias(bias)));
    drop_ = register_module("drop", torch::nn::Dropout(torch::nn::DropoutOptions(drop)));
}

torch::Tensor MlpImpl::forward(const torch::Tensor& x) {
    auto out = fc1->forward(x);
    out = act->forward(out);
    out = drop_->forward(out);
    out = fc2->forward(out);
    out = drop_->forward(out);
    return out;
}

} // namespace layers
} // namespace vggt