/**
 * @file rope.cpp
 * @brief Implementation of 2D Rotary Position Embeddings (RoPE)
 *
 * This file implements the PositionGetter and RotaryPositionEmbedding2D classes
 * defined in rope.h, providing functionality for 2D rotary position embeddings.
 */

#include "rope.h"
#include <sstream>

namespace vggt {
namespace layers {

PositionGetter::PositionGetter() {}

torch::Tensor PositionGetter::operator()(int64_t batch_size, int64_t height, int64_t width, torch::Device device) {
    // Create a cache key from height and width
    std::stringstream key_stream;
    key_stream << height << "_" << width;
    std::string cache_key = key_stream.str();

    // Check if positions for this grid size are already cached
    if (position_cache_.find(cache_key) == position_cache_.end()) {
        // Generate y and x coordinates
        auto y_coords = torch::arange(height, device);
        auto x_coords = torch::arange(width, device);
        
        // Create cartesian product of coordinates
        std::vector<torch::Tensor> positions_list;
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                positions_list.push_back(torch::tensor({y, x}, device));
            }
        }
        
        // Stack tensors to create positions tensor
        auto positions = torch::stack(positions_list);
        position_cache_[cache_key] = positions;
    }

    // Get cached positions and expand for batch
    auto cached_positions = position_cache_[cache_key];
    return cached_positions.view({1, height * width, 2}).expand({batch_size, -1, -1}).clone();
}

RotaryPositionEmbedding2DImpl::RotaryPositionEmbedding2DImpl(double frequency, double scaling_factor)
    : base_frequency_(frequency), scaling_factor_(scaling_factor) {}

std::tuple<torch::Tensor, torch::Tensor> RotaryPositionEmbedding2DImpl::compute_frequency_components(
    int64_t dim, int64_t seq_len, torch::Device device, torch::ScalarType dtype) {
    
    // Create a cache key
    std::stringstream key_stream;
    key_stream << dim << "_" << seq_len << "_" << device << "_" << dtype;
    std::string cache_key = key_stream.str();

    // Check if frequency components are already cached
    if (frequency_cache_.find(cache_key) == frequency_cache_.end()) {
        // Compute frequency bands
        auto exponents = torch::arange(0, dim, 2, device).to(torch::kFloat32) / dim;
        auto inv_freq = 1.0 / torch::pow(torch::tensor(base_frequency_, device), exponents);

        // Generate position-dependent frequencies
        auto positions = torch::arange(seq_len, device).to(inv_freq.dtype());
        auto angles = torch::einsum("i,j->ij", {positions, inv_freq});

        // Compute and cache frequency components
        angles = angles.to(dtype);
        angles = torch::cat({angles, angles}, -1);
        auto cos_components = angles.cos().to(dtype);
        auto sin_components = angles.sin().to(dtype);
        
        frequency_cache_[cache_key] = std::make_tuple(cos_components, sin_components);
    }

    return frequency_cache_[cache_key];
}

torch::Tensor RotaryPositionEmbedding2DImpl::rotate_features(const torch::Tensor& x) {
    int64_t feature_dim = x.size(-1);
    auto x1 = x.index({"...", torch::indexing::Slice(0, feature_dim / 2)});
    auto x2 = x.index({"...", torch::indexing::Slice(feature_dim / 2, torch::indexing::None)});
    return torch::cat({-x2, x1}, -1);
}

torch::Tensor RotaryPositionEmbedding2DImpl::apply_1d_rope(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const torch::Tensor& cos_comp,
    const torch::Tensor& sin_comp) {
    
    // Embed positions with frequency components
    auto cos = torch::embedding(positions, cos_comp).unsqueeze(1);
    auto sin = torch::embedding(positions, sin_comp).unsqueeze(1);

    // Apply rotation
    return (tokens * cos) + (rotate_features(tokens) * sin);
}

torch::Tensor RotaryPositionEmbedding2DImpl::forward(const torch::Tensor& tokens, const torch::Tensor& positions) {
    // Validate inputs
    TORCH_CHECK(tokens.size(-1) % 2 == 0, "Feature dimension must be even");
    TORCH_CHECK(positions.dim() == 3 && positions.size(-1) == 2, 
                "Positions must have shape (batch_size, n_tokens, 2)");

    // Compute feature dimension for each spatial direction
    int64_t feature_dim = tokens.size(-1) / 2;

    // Get frequency components
    int64_t max_position = static_cast<int64_t>(positions.max().item<float>()) + 1;
    auto [cos_comp, sin_comp] = compute_frequency_components(
        feature_dim, max_position, tokens.device(), tokens.scalar_type());

    // Split features for vertical and horizontal processing
    auto chunks = tokens.chunk(2, -1);
    auto vertical_features = chunks[0];
    auto horizontal_features = chunks[1];

    // Apply RoPE separately for each dimension
    auto y_positions = positions.index({"...", 0});
    auto x_positions = positions.index({"...", 1});
    
    vertical_features = apply_1d_rope(vertical_features, y_positions, cos_comp, sin_comp);
    horizontal_features = apply_1d_rope(horizontal_features, x_positions, cos_comp, sin_comp);

    // Combine processed features
    return torch::cat({vertical_features, horizontal_features}, -1);
}

} // namespace layers
} // namespace vggt