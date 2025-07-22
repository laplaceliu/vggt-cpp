/**
 * @file mlp.h
 * @brief Multi-Layer Perceptron implementation for vision transformers
 *
 * This file defines the MLP module used in vision transformers.
 * References:
 *   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
 *   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py
 */

#pragma once

#include <torch/torch.h>

namespace vggt {
namespace layers {

/**
 * @brief Options for the MLP module
 */
struct MlpOptions {
    MlpOptions(int64_t in_features, int64_t hidden_features = 0, int64_t out_features = 0) {
        this->in_features = in_features;
        this->hidden_features = hidden_features > 0 ? hidden_features : in_features;
        this->out_features = out_features > 0 ? out_features : in_features;
    }

    // Fluent API for setting options
    MlpOptions& bias(bool bias_) {
        this->bias_ = bias_;
        return *this;
    }

    MlpOptions& drop(double drop_) {
        this->drop_ = drop_;
        return *this;
    }

    MlpOptions& act_layer(torch::nn::Module act_layer_) {
        this->act_layer_ = act_layer_;
        return *this;
    }

    int64_t in_features;
    int64_t hidden_features;
    int64_t out_features;
    bool bias_ = true;
    double drop_ = 0.0;
    c10::optional<torch::nn::Module> act_layer_ = c10::nullopt;
};

/**
 * @brief Multi-Layer Perceptron module for vision transformers
 *
 * This class implements a standard MLP with configurable hidden dimensions,
 * activation function, and dropout rate.
 */
class MlpImpl : public torch::nn::Module {
public:
    /**
     * @brief Constructor for MLP module
     *
     * @param in_features Input feature dimension
     * @param hidden_features Hidden layer feature dimension (default: same as in_features)
     * @param out_features Output feature dimension (default: same as in_features)
     * @param drop Dropout rate (default: 0.0)
     * @param bias Whether to use bias in linear layers (default: true)
     */
    MlpImpl(
        int64_t in_features,
        int64_t hidden_features = 0,
        bool bias = true,
        double drop = 0.0
    );

    /**
     * @brief Constructor for MLP module with options
     *
     * @param options MLP configuration options
     */
    explicit MlpImpl(const MlpOptions& options);

    /**
     * @brief Forward pass of the MLP module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(const torch::Tensor& x);

private:
    torch::nn::Linear fc1;
    torch::nn::GELU act;
    torch::nn::Linear fc2;
    torch::nn::Dropout drop_;
};

TORCH_MODULE(Mlp);

} // namespace layers
} // namespace vggt
