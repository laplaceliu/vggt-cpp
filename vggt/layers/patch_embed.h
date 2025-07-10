/**
 * @brief Patch embedding module for vision transformers
 *
 * This file defines the PatchEmbed module which implements the patch embedding
 * technique used in Vision Transformers (ViT). Patch embedding is the first stage
 * in a vision transformer that converts an image into a sequence of embedded patches.
 *
 * The implementation includes:
 * 1. A PatchEmbedImpl class that splits images into fixed-size patches
 * 2. Projection of each patch to an embedding dimension using a convolutional layer
 * 3. Optional normalization of the embedded patches
 * 4. Support for different image sizes, patch sizes, and embedding dimensions
 *
 * Patch embedding is a critical component that allows transformers, which were
 * originally designed for sequence data, to process image data by converting
 * 2D image patches into a sequence of token embeddings.
 */

#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/normalization.h>

namespace vggt {
namespace layers {

class PatchEmbedImpl : public torch::nn::Module {
public:
    PatchEmbedImpl(
        int64_t img_size = 224,
        int64_t patch_size = 16,
        int64_t in_chans = 3,
        int64_t embed_dim = 768,
        torch::nn::AnyModule norm = nullptr,
    torch::Tensor forward(const torch::Tensor& x);

private:
    std::pair<int64_t, int64_t> img_size_;
    std::pair<int64_t, int64_t> patch_size_;
    std::pair<int64_t, int64_t> patches_resolution_;
    int64_t num_patches_;
    int64_t in_chans_;
    int64_t embed_dim_;
    bool flatten_embedding_;
    torch::nn::Conv2d proj{nullptr};
    torch::nn::AnyModule norm{nullptr};
};

TORCH_MODULE(PatchEmbed);

} // namespace layers
} // namespace vggt
