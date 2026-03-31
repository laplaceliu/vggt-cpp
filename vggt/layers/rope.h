#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <utility>
#include <functional>
#include <tuple>

namespace vggt {
namespace layers {

/**
 * @struct PairHash
 * @brief Hash function for std::pair<int64_t, int64_t>
 * 
 * Used for caching position embeddings with a hash map.
 */
struct PairHash {
    /**
     * @brief Hash a pair of int64_t values
     * @param key The pair to hash
     * @return Hash value
     */
    size_t operator()(const std::pair<int64_t, int64_t>& key) const {
        auto hash_combine = [](size_t seed, size_t value) {
            return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
        };

        size_t seed = std::hash<int64_t>{}(key.first);
        seed = hash_combine(seed, std::hash<int64_t>{}(key.second));
        return seed;
    }
};

/**
 * @struct TupleHash
 * @brief Hash function for tuple of (int64_t, int64_t, torch::Device, torch::Dtype)
 * 
 * Used for caching frequency components with a hash map.
 */
struct TupleHash {
    /**
     * @brief Hash a tuple of dimension, sequence length, device, and dtype
     * @param key The tuple to hash
     * @return Hash value
     */
    size_t operator()(const std::tuple<int64_t, int64_t, torch::Device, torch::Dtype>& key) const {
        auto hash_combine = [](size_t seed, size_t value) {
            return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
        };

        size_t seed = 0;
        seed = hash_combine(seed, std::hash<int64_t>{}(std::get<0>(key)));
        seed = hash_combine(seed, std::hash<int64_t>{}(std::get<1>(key)));
        seed = hash_combine(seed, std::hash<torch::Device>{}(std::get<2>(key)));
        seed = hash_combine(seed, std::hash<torch::Dtype>{}(std::get<3>(key)));
        return seed;
    }
};

/**
 * @class PositionGetter
 * @brief Generates 2D position coordinates for tokens
 * 
 * Creates a grid of (y, x) coordinates for 2D position encoding.
 * Results are cached to avoid recomputation.
 */
class PositionGetter {
public:
    PositionGetter() = default;

    /**
     * @brief Get 2D positions for a batch of tokens
     * @param batch_size Number of sequences in batch
     * @param height Grid height
     * @param width Grid width
     * @param device Device to create tensor on
     * @return Positions tensor of shape [batch_size, height*width, 2]
     */
    torch::Tensor operator()(int64_t batch_size, int64_t height, int64_t width, const torch::Device& device);

private:
    std::unordered_map<std::pair<int64_t, int64_t>, torch::Tensor, PairHash> position_cache_;
};

/**
 * @class RotaryPositionEmbedding2DImpl
 * @brief 2D Rotary Position Embedding (RoPE) implementation
 * 
 * Implements rotary position embedding for 2D features as described in
 * "RoFormer: Enhanced Transformer with Rotary Position Embedding".
 * 
 * The key idea is to rotate the query and key vectors in 2D space based on
 * their position, encoding relative position information without explicit
 * distance biases.
 * 
 * The 2D version splits features into vertical (y) and horizontal (x)
 * components, applies 1D RoPE to each, then concatenates the result.
 * 
 * @param base_frequency Base frequency for the rotary angles (default: 100.0)
 * @param scaling_factor Scaling factor for positions (default: 1.0)
 */
class RotaryPositionEmbedding2DImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a 2D RoPE layer
     * @param frequency Base frequency for angles (default: 100.0)
     * @param scaling_factor Scaling factor for positions (default: 1.0)
     */
    RotaryPositionEmbedding2DImpl(double frequency = 100.0, double scaling_factor = 1.0);

    /**
     * @brief Apply 2D rotary position embedding
     * @param tokens Input tokens of shape [B, N, C] where C is even
     * @param positions Position indices of shape [B, N, 2] (y, x coordinates)
     * @return Embedded tokens with RoPE applied
     * @note Input feature dimension C must be even
     * @note positions values should be integers in [0, seq_len)
     */
    torch::Tensor forward(torch::Tensor tokens, torch::Tensor positions);

private:
    /**
     * @brief Compute frequency components for rotary embedding
     * @param dim Feature dimension (half of embed_dim)
     * @param seq_len Sequence length
     * @param device Device for tensor
     * @param dtype Data type
     * @return Pair of (cos, sin) components cached by (dim, seq_len, device, dtype)
     */
    std::pair<torch::Tensor, torch::Tensor> compute_frequency_components(
        int64_t dim, int64_t seq_len, const torch::Device& device, torch::Dtype dtype);

    /**
     * @brief Rotate feature dimensions
     * @param x Input tensor [..., C]
     * @return Rotated tensor [..., C] where C/2 elements are swapped with negated C/2 elements
     */
    static torch::Tensor rotate_features(torch::Tensor x);

    /**
     * @brief Apply 1D RoPE to tokens
     * @param tokens Token embeddings [..., D]
     * @param positions Position indices [..., 1]
     * @param cos_comp Cosine components [max_pos, D]
     * @param sin_comp Sine components [max_pos, D]
     * @return RoPE-transformed tokens
     */
    torch::Tensor apply_1d_rope(
        torch::Tensor tokens, torch::Tensor positions, torch::Tensor cos_comp, torch::Tensor sin_comp);

    double base_frequency_;  ///< Base frequency for angle computation
    double scaling_factor_;  ///< Scaling factor for positions
    std::unordered_map<std::tuple<int64_t, int64_t, torch::Device, torch::Dtype>,
                       std::pair<torch::Tensor, torch::Tensor>, TupleHash> frequency_cache_;
};

TORCH_MODULE(RotaryPositionEmbedding2D);

} // namespace layers
} // namespace vggt
