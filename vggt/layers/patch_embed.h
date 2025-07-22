/**
 * @file patch_embed.h
 * @brief Patch embedding implementation for vision transformers
 *
 * This file defines the PatchEmbed module which converts 2D images into sequences
 * of patch embeddings used in vision transformers. It splits the input image into
 * fixed-size patches and projects each patch to an embedding dimension.
 *
 * The module takes an input tensor of shape (B, C, H, W) and outputs:
 * 1. Either a sequence of patch embeddings (B, N, D) where N is the number of patches
 * 2. Or a spatial representation (B, H', W', D) if flatten_embedding is false
 */

#pragma once

#include <torch/torch.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/modules/container/any.h>
#include <tuple>

namespace vggt {
namespace layers {

/**
 * Helper function to convert a single integer or a pair of integers to a std::pair
 */
std::pair<int64_t, int64_t> make_2tuple(int64_t x);
std::pair<int64_t, int64_t> make_2tuple(const std::pair<int64_t, int64_t>& x);

class PatchEmbedImpl : public torch::nn::Module {
public:
    /**
     * 2D image to patch embedding: (B,C,H,W) -> (B,N,D) or (B,H',W',D)
     *
     * @param img_size Image size (single integer for square images or pair for rectangular)
     * @param patch_size Patch token size (single integer for square patches or pair for rectangular)
     * @param in_chans Number of input image channels
     * @param embed_dim Number of linear projection output channels
     * @param norm_layer Optional normalization layer
     * @param flatten_embedding Whether to flatten the spatial dimensions into a sequence
     */
    PatchEmbedImpl(
        int64_t img_size = 224,
        int64_t patch_size = 16,
        int64_t in_chans = 3,
        int64_t embed_dim = 768,
        torch::nn::AnyModule norm_layer = torch::nn::AnyModule(),
        bool flatten_embedding = true);

    PatchEmbedImpl(
        const std::pair<int64_t, int64_t>& img_size,
        const std::pair<int64_t, int64_t>& patch_size,
        int64_t in_chans = 3,
        int64_t embed_dim = 768,
        torch::nn::AnyModule norm_layer = torch::nn::AnyModule(),
        bool flatten_embedding = true);

    torch::Tensor forward(const torch::Tensor& x);

    /**
     * Calculate FLOPs for the module
     */
    double flops() const;

private:
    std::pair<int64_t, int64_t> img_size_;
    std::pair<int64_t, int64_t> patch_size_;
    std::pair<int64_t, int64_t> patches_resolution_;
    int64_t num_patches_;
    int64_t in_chans_;
    int64_t embed_dim_;
    bool flatten_embedding_;

    torch::nn::Conv2d proj;
    torch::nn::AnyModule norm;
};

TORCH_MODULE(PatchEmbed);

} // namespace layers
} // namespace vggt
