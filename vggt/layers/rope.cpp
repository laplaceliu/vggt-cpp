#include "rope.h"
#include <tuple>
#include <unordered_map>
#include <functional>

namespace vggt {
namespace layers {

torch::Tensor PositionGetter::operator()(int64_t batch_size, int64_t height, int64_t width, const torch::Device& device) {
    auto key = std::make_pair(height, width);
    if (position_cache_.find(key) == position_cache_.end()) {
        // Create 1D coordinate arrays for each dimension
        auto y_coords = torch::arange(height, torch::TensorOptions().device(device));
        auto x_coords = torch::arange(width, torch::TensorOptions().device(device));
        // Cartesian product gives all (y, x) combinations
        auto positions = torch::cartesian_prod({y_coords, x_coords});
        position_cache_[key] = positions;
    }

    // Return cached positions expanded to batch size
    // Shape: [height*width, 2] -> [batch_size, height*width, 2]
    auto cached_positions = position_cache_[key];
    return cached_positions.view({1, height * width, 2}).expand({batch_size, -1, -1}).clone();
}

RotaryPositionEmbedding2DImpl::RotaryPositionEmbedding2DImpl(double frequency, double scaling_factor)
    : base_frequency_(frequency), scaling_factor_(scaling_factor),
      frequency_cache_{} {}

std::pair<torch::Tensor, torch::Tensor> RotaryPositionEmbedding2DImpl::compute_frequency_components(
    int64_t dim, int64_t seq_len, const torch::Device& device, torch::Dtype dtype) {
    auto key = std::make_tuple(dim, seq_len, device, dtype);
    if (frequency_cache_.find(key) == frequency_cache_.end()) {
        // Compute exponents: arange(0, dim, 2) / dim gives [0, 2/dim, 4/dim, ...]
        auto exponents = torch::arange(0, dim, 2, torch::TensorOptions().device(device)).to(torch::kFloat32) / dim;
        // Inverse frequency: 1 / base^exponent
        auto inv_freq = 1.0 / torch::pow(base_frequency_, exponents);

        // Compute angles for each position: positions * inv_freq
        auto positions = torch::arange(seq_len, torch::TensorOptions().device(device).dtype(inv_freq.dtype()));
        auto angles = torch::einsum("i,j->ij", {positions, inv_freq});

        // Convert to requested dtype and duplicate for both sin and cos
        angles = angles.to(dtype);
        angles = torch::cat({angles, angles}, -1);  // [seq_len, dim]
        
        // Compute sin and cos components
        auto cos_components = angles.cos().to(dtype);
        auto sin_components = angles.sin().to(dtype);
        frequency_cache_[key] = std::make_pair(cos_components, sin_components);
    }

    return frequency_cache_[key];
}

torch::Tensor RotaryPositionEmbedding2DImpl::rotate_features(torch::Tensor x) {
    // Rotate feature dimensions: split in half and negate second half
    // [x1, x2, x3, x4, x5, x6] -> [-x4, -x5, -x6, x1, x2, x3]
    auto feature_dim = x.size(-1);
    auto x1 = x.slice(-1, 0, feature_dim / 2);
    auto x2 = x.slice(-1, feature_dim / 2);
    return torch::cat({-x2, x1}, -1);
}

torch::Tensor RotaryPositionEmbedding2DImpl::apply_1d_rope(
    torch::Tensor tokens, torch::Tensor positions, torch::Tensor cos_comp, torch::Tensor sin_comp) {
    // Look up cos and sin values for given positions
    // cos_comp: [max_pos, dim], positions: [B, N], result: [B, N, dim]
    auto cos = torch::embedding(cos_comp, positions).unsqueeze(1);
    auto sin = torch::embedding(sin_comp, positions).unsqueeze(1);
    
    // Apply rotary transformation: x * cos + rotate(x) * sin
    return tokens * cos + rotate_features(tokens) * sin;
}

torch::Tensor RotaryPositionEmbedding2DImpl::forward(torch::Tensor tokens, torch::Tensor positions) {
    TORCH_CHECK(tokens.size(-1) % 2 == 0, "Feature dimension must be even");
    TORCH_CHECK(positions.dim() == 3 && positions.size(-1) == 2,
                "Positions must have shape (batch_size, n_tokens, 2)");

    // Split features into vertical (y) and horizontal (x) components
    // Each has half the feature dimension
    auto feature_dim = tokens.size(-1) / 2;
    
    // Get max position for frequency computation
    auto max_position = positions.max().item<int64_t>() + 1;
    auto [cos_comp, sin_comp] = compute_frequency_components(
        feature_dim, max_position, tokens.device(), static_cast<torch::Dtype>(tokens.scalar_type()));

    // Split tokens along feature dimension
    auto chunks = torch::chunk(tokens, 2, -1);
    auto vertical_features = chunks[0];      // [B, N, dim]
    auto horizontal_features = chunks[1];   // [B, N, dim]
    
    // Apply 1D RoPE separately to y and x components
    // positions.select(-1, 0) extracts y-coordinates, select(-1, 1) extracts x-coordinates
    vertical_features = apply_1d_rope(vertical_features, positions.select(-1, 0), cos_comp, sin_comp);
    horizontal_features = apply_1d_rope(horizontal_features, positions.select(-1, 1), cos_comp, sin_comp);

    // Concatenate back along feature dimension
    return torch::cat({vertical_features, horizontal_features}, -1);
}

} // namespace layers
} // namespace vggt
