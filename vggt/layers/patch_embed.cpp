#include "patch_embed.h"

namespace vggt {
namespace layers {

PatchEmbedImpl::PatchEmbedImpl(
    int64_t img_size,
    int64_t patch_size,
    int64_t in_chans,
    int64_t embed_dim,
    torch::nn::AnyModule norm_layer,
    bool flatten_embedding
) : PatchEmbedImpl(make_2tuple(img_size), make_2tuple(patch_size), in_chans, embed_dim, norm_layer, flatten_embedding) {}

PatchEmbedImpl::PatchEmbedImpl(
    std::tuple<int64_t, int64_t> img_size,
    std::tuple<int64_t, int64_t> patch_size,
    int64_t in_chans,
    int64_t embed_dim,
    torch::nn::AnyModule norm_layer,
    bool flatten_embedding
) : use_norm_(!norm_layer.is_empty()) {
    // Store the tuple values directly
    img_size_ = img_size;
    patch_size_ = patch_size;
    
    // Compute number of patches in each spatial dimension
    patches_resolution_ = std::make_tuple(
        std::get<0>(img_size_) / std::get<0>(patch_size_),
        std::get<1>(img_size_) / std::get<1>(patch_size_)
    );
    
    // Total number of patches
    num_patches_ = std::get<0>(patches_resolution_) * std::get<1>(patches_resolution_);

    in_chans_ = in_chans;
    embed_dim_ = embed_dim;
    flatten_embedding_ = flatten_embedding;

    // Convolutional projection: each patch becomes a vector of size embed_dim
    // Uses Conv2d with kernel_size=stride to implement non-overlapping patch extraction
    proj_ = register_module("proj", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_chans, embed_dim, {std::get<0>(patch_size_), std::get<1>(patch_size_)})
            .stride({std::get<0>(patch_size_), std::get<1>(patch_size_)})
    ));
    
    // Optional layer normalization applied after patch embedding
    if (use_norm_) {
        norm_ = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
    }
}

torch::Tensor PatchEmbedImpl::forward(torch::Tensor x) {
    auto sizes = x.sizes();
    auto H = sizes[2];
    auto W = sizes[3];
    auto patch_H = std::get<0>(patch_size_);
    auto patch_W = std::get<1>(patch_size_);

    // Verify input dimensions are divisible by patch size
    TORCH_CHECK(H % patch_H == 0, "Input image height ", H, " is not a multiple of patch height ", patch_H);
    TORCH_CHECK(W % patch_W == 0, "Input image width ", W, " is not a multiple of patch width ", W);

    // Apply convolutional projection to extract patches
    x = proj_->forward(x); // B C H W
    
    // Get spatial dimensions after projection
    H = x.size(2);
    W = x.size(3);
    
    // Flatten spatial dimensions and transpose: B C HW -> B HW C
    x = x.flatten(2).transpose(1, 2); // B HW C
    
    // Apply layer normalization if enabled
    if (use_norm_) {
        x = norm_->forward(x);
    }
    
    // Optionally reshape to spatial format instead of flat sequence
    if (!flatten_embedding_) {
        x = x.reshape(torch::IntArrayRef({-1, H, W, embed_dim_})); // B H W C
    }
    
    return x;
}

} // namespace layers
} // namespace vggt
