#pragma once

#include <torch/torch.h>

namespace vggt {
namespace layers {

/**
 * @brief Helper function to convert single value to 2-tuple
 * @param x Input value
 * @return Tuple of (x, x)
 */
inline std::tuple<int64_t, int64_t> make_2tuple(int64_t x) {
    return std::make_tuple(x, x);
}

/**
 * @brief Helper function to pass through 2-tuple
 * @param x Input tuple
 * @return Same tuple
 */
inline std::tuple<int64_t, int64_t> make_2tuple(const std::tuple<int64_t, int64_t>& x) {
    return x;
}

/**
 * @class PatchEmbedImpl
 * @brief Patch Embedding module for Vision Transformer
 * 
 * Converts an input image into a sequence of flattened patches using a
 * convolutional projection. Each patch is embedded into a vector of
 * dimension embed_dim.
 * 
 * The forward pass:
 * 1. Apply Conv2d to extract patches [B, C, H, W] -> [B, embed_dim, H', W']
 * 2. Flatten spatial dimensions and transpose [B, embed_dim, H', W'] -> [B, H'*W', embed_dim]
 * 3. Optionally apply layer normalization
 * 4. Optionally reshape to spatial format [B, H', W', embed_dim] if flatten_embedding=false
 * 
 * @param img_size Input image size (single int or tuple, default: 224)
 * @param patch_size Size of each patch (single int or tuple, default: 16)
 * @param in_chans Number of input channels (default: 3)
 * @param embed_dim Embedding dimension per patch (default: 768)
 * @param norm_layer Optional LayerNorm applied after projection
 * @param flatten_embedding Whether to flatten spatial dimensions (default: true)
 */
class PatchEmbedImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a PatchEmbed layer
     * @param img_size Input image size (default: 224)
     * @param patch_size Size of each patch (default: 16)
     * @param in_chans Number of input channels (default: 3)
     * @param embed_dim Embedding dimension (default: 768)
     * @param norm_layer Optional normalization layer
     * @param flatten_embedding Whether to flatten spatial dims (default: true)
     */
    PatchEmbedImpl(
        int64_t img_size = 224,
        int64_t patch_size = 16,
        int64_t in_chans = 3,
        int64_t embed_dim = 768,
        torch::nn::AnyModule norm_layer = torch::nn::AnyModule(),
        bool flatten_embedding = true
    );

    /**
     * @brief Forward pass of patch embedding
     * @param x Input tensor of shape [B, C, H, W]
     * @return Embedded patches:
     *         - If flatten_embedding=true: [B, num_patches, embed_dim]
     *         - If flatten_embedding=false: [B, H', W', embed_dim]
     * @note H and W must be divisible by patch_size
     */
    torch::Tensor forward(torch::Tensor x);

private:
    std::tuple<int64_t, int64_t> img_size_;          ///< Input image size (H, W)
    std::tuple<int64_t, int64_t> patch_size_;         ///< Patch size (H, W)
    std::tuple<int64_t, int64_t> patches_resolution_; ///< Number of patches (H', W')
    int64_t num_patches_;                             ///< Total number of patches H'*W'
    int64_t in_chans_;                                ///< Number of input channels
    int64_t embed_dim_;                               ///< Embedding dimension
    bool flatten_embedding_;                          ///< Whether to flatten patches
    torch::nn::Conv2d proj_{nullptr};                 ///< Convolutional projection
    torch::nn::LayerNorm norm_{nullptr};              ///< Optional normalization
    bool use_norm_;                                   ///< Whether norm is applied
};

TORCH_MODULE(PatchEmbed);

} // namespace layers
} // namespace vggt
