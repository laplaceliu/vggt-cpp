/**
 * @brief Implementation of patch embedding module for vision transformers
 *
 * This file implements the PatchEmbedImpl class methods defined in patch_embed.h.
 * It provides the core functionality for converting images into sequences of
 * embedded patches for processing by vision transformer architectures.
 *
 * The implementation includes:
 * 1. Constructor that initializes the patch embedding parameters, including
 *    image size, patch size, input channels, and embedding dimension
 * 2. A convolutional projection layer that maps each patch to the embedding space
 * 3. Forward method that processes input images, validates dimensions, and
 *    produces embedded patch sequences
 * 4. Optional flattening of the embedded patches into a sequence format
 * 5. Optional normalization of the embedded patches
 *
 * The implementation performs validation to ensure input dimensions match
 * the expected model configuration, throwing exceptions for mismatches.
 */

#include "patch_embed.h"

namespace vggt {
namespace layers {

PatchEmbedImpl::PatchEmbedImpl(
    int64_t img_size,
    int64_t patch_size,
    int64_t in_chans,
    int64_t embed_dim,
    torch::nn::AnyModule norm,
    bool flatten_embedding)
    : img_size_({img_size, img_size}),
      patch_size_({patch_size, patch_size}),
      patches_resolution_({
          img_size / patch_size,
          img_size / patch_size
      }),
      num_patches_(patches_resolution_.first * patches_resolution_.second),
      in_chans_(in_chans),
      embed_dim_(embed_dim),
      flatten_embedding_(flatten_embedding),
      proj(torch::nn::Conv2dOptions(in_chans, embed_dim, patch_size)
          .stride(patch_size)),
      norm(norm) {

    if (img_size % patch_size != 0) {
        throw std::runtime_error(
            "Image size must be divisible by patch size. Got: " +
            std::to_string(img_size) + " and " + std::to_string(patch_size));
    }

    register_module("proj", proj);
    if (!norm.is_empty()) {
        register_module("norm", norm);
    }
}

torch::Tensor PatchEmbedImpl::forward(const torch::Tensor& x) {
    auto B = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    if (H != img_size_.first || W != img_size_.second) {
        throw std::runtime_error(
            "Input image size (" + std::to_string(H) + "*" + std::to_string(W) +
            ") doesn't match model (" + std::to_string(img_size_.first) + "*" +
            std::to_string(img_size_.second) + ").");
    }

    if (C != in_chans_) {
        throw std::runtime_error(
            "Input channels (" + std::to_string(C) + ") doesn't match model (" +
            std::to_string(in_chans_) + ").");
    auto x_proj = proj->forward(x); // [B, embed_dim, H/patch_size, W/patch_size]

    if (flatten_embedding_) {
        x_proj = x_proj.flatten(2).transpose(1, 2); // [B, num_patches, embed_dim]
    }

    if (!norm.is_empty()) {
        x_proj = norm.forward<torch::Tensor>(x_proj);
    }

    return x_proj;
}

} // namespace layers
} // namespace vggt
