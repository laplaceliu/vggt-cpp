/**
 * @file rope.h
 * @brief 2D Rotary Position Embeddings (RoPE) implementation
 *
 * This file defines classes for implementing 2D Rotary Position Embeddings,
 * which extends the original RoPE concept to handle 2D spatial positions.
 */

#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <tuple>

namespace vggt {
namespace layers {

/**
 * @brief Generates and caches 2D spatial positions for patches in a grid
 * 
 * This class efficiently manages the generation of spatial coordinates for patches
 * in a 2D grid, caching results to avoid redundant computations.
 */
class PositionGetter {
public:
    /**
     * @brief Initializes the position generator with an empty cache
     */
    PositionGetter();

    /**
     * @brief Generates spatial positions for a batch of patches
     * 
     * @param batch_size Number of samples in the batch
     * @param height Height of the grid in patches
     * @param width Width of the grid in patches
     * @param device Target device for the position tensor
     * @return Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
     */
    torch::Tensor operator()(int64_t batch_size, int64_t height, int64_t width, torch::Device device);

private:
    // Cache for position tensors, keyed by (height, width)
    std::unordered_map<std::string, torch::Tensor> position_cache_;
};

/**
 * @brief 2D Rotary Position Embedding implementation
 * 
 * This module applies rotary position embeddings to input tokens based on their
 * 2D spatial positions. It handles the position-dependent rotation of features
 * separately for vertical and horizontal dimensions.
 */
class RotaryPositionEmbedding2DImpl : public torch::nn::Module {
public:
    /**
     * @brief Initializes the 2D RoPE module
     * 
     * @param frequency Base frequency for the position embeddings (default: 100.0)
     * @param scaling_factor Scaling factor for frequency computation (default: 1.0)
     */
    RotaryPositionEmbedding2DImpl(double frequency = 100.0, double scaling_factor = 1.0);

    /**
     * @brief Applies 2D rotary position embeddings to input tokens
     * 
     * @param tokens Input tensor of shape (batch_size, n_heads, n_tokens, dim)
     * @param positions Position tensor of shape (batch_size, n_tokens, 2)
     * @return Tensor with applied 2D rotary position embeddings
     */
    torch::Tensor forward(const torch::Tensor& tokens, const torch::Tensor& positions);

private:
    /**
     * @brief Computes frequency components for rotary embeddings
     * 
     * @param dim Feature dimension (must be even)
     * @param seq_len Maximum sequence length
     * @param device Target device for computations
     * @param dtype Data type for the computed tensors
     * @return Tuple of (cosine, sine) tensors for frequency components
     */
    std::tuple<torch::Tensor, torch::Tensor> compute_frequency_components(
        int64_t dim, int64_t seq_len, torch::Device device, torch::ScalarType dtype);

    /**
     * @brief Performs feature rotation by splitting and recombining feature dimensions
     * 
     * @param x Input tensor to rotate
     * @return Rotated feature tensor
     */
    torch::Tensor rotate_features(const torch::Tensor& x);

    /**
     * @brief Applies 1D rotary position embeddings along one dimension
     * 
     * @param tokens Input token features
     * @param positions Position indices
     * @param cos_comp Cosine components for rotation
     * @param sin_comp Sine components for rotation
     * @return Tokens with applied rotary position embeddings
     */
    torch::Tensor apply_1d_rope(
        const torch::Tensor& tokens,
        const torch::Tensor& positions,
        const torch::Tensor& cos_comp,
        const torch::Tensor& sin_comp);

    double base_frequency_;
    double scaling_factor_;
    
    // Cache for frequency components, keyed by (dim, seq_len, device, dtype)
    std::unordered_map<std::string, std::tuple<torch::Tensor, torch::Tensor>> frequency_cache_;
};

TORCH_MODULE(RotaryPositionEmbedding2D);

} // namespace layers
} // namespace vggt