/**
 * @file patch_embed.cpp
 * @brief Implementation of the PatchEmbed module for vision transformers
 *
 * This file implements the PatchEmbedImpl class methods defined in patch_embed.h.
 * It provides the core functionality for converting 2D images into sequences of
 * patch embeddings used in vision transformers, including:
 *
 * 1. Constructor for initializing the patch embedding module with specified parameters
 * 2. Forward method that implements the patch embedding computation logic
 * 3. Support for both flattened sequence output and spatial output formats
 * 4. Helper functions for handling dimension specifications
 * 5. FLOPs calculation for computational complexity estimation
 *
 * The implementation uses PyTorch's tensor operations and modules to ensure
 * compatibility with the PyTorch ecosystem and efficient execution on both
 * CPU and GPU devices.
 */

#include "patch_embed.h"
#include <stdexcept>

namespace vggt {
namespace layers {

std::pair<int64_t, int64_t> make_2tuple(int64_t x) {
    return std::make_pair(x, x);
}

std::pair<int64_t, int64_t> make_2tuple(const std::pair<int64_t, int64_t>& x) {
    return x;
}

PatchEmbedImpl::PatchEmbedImpl(
    int64_t img_size,
    int64_t patch_size,
    int64_t in_chans,
    int64_t embed_dim,
    torch::nn::AnyModule norm_layer,
    bool flatten_embedding)
    : PatchEmbedImpl(
          make_2tuple(img_size),
          make_2tuple(patch_size),
          in_chans,
          embed_dim,
          norm_layer,
          flatten_embedding) {
}

PatchEmbedImpl::PatchEmbedImpl(
    const std::pair<int64_t, int64_t>& img_size,
    const std::pair<int64_t, int64_t>& patch_size,
    int64_t in_chans,
    int64_t embed_dim,
    torch::nn::AnyModule norm_layer,
    bool flatten_embedding)
    : img_size_(img_size),
      patch_size_(patch_size),
      patches_resolution_({img_size.first / patch_size.first, img_size.second / patch_size.second}),
      num_patches_(patches_resolution_.first * patches_resolution_.second),
      in_chans_(in_chans),
      embed_dim_(embed_dim),
      flatten_embedding_(flatten_embedding) {
    
    // Create projection from image patches to embedding dimension
    proj = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_chans, embed_dim, patch_size)
            .stride(patch_size)
    );
    
    // Register the projection module
    register_module("proj", proj);
    
    // Set up normalization layer if provided, otherwise use Identity
    if (norm_layer.ptr() != nullptr) {
        norm = norm_layer;
        register_module("norm", norm);
    } else {
        register_module("norm", torch::nn::Identity());
    }
}

torch::Tensor PatchEmbedImpl::forward(const torch::Tensor& x) {
    auto B = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    
    auto patch_H = patch_size_.first;
    auto patch_W = patch_size_.second;
    
    // Check that input dimensions are divisible by patch size
    if (H % patch_H != 0) {
        throw std::runtime_error(
            "Input image height " + std::to_string(H) + 
            " is not a multiple of patch height " + std::to_string(patch_H));
    }
    
    if (W % patch_W != 0) {
        throw std::runtime_error(
            "Input image width " + std::to_string(W) + 
            " is not a multiple of patch width " + std::to_string(patch_W));
    }
    
    // Project patches
    auto x_projected = proj->forward(x);  // B, embed_dim, H', W'
    
    auto H_out = x_projected.size(2);
    auto W_out = x_projected.size(3);
    
    // Flatten spatial dimensions and transpose to get sequence of embeddings
    auto x_flattened = x_projected.flatten(2).transpose(1, 2);  // B, H'*W', embed_dim
    
    // Apply normalization
    torch::Tensor output;
    if (norm.ptr() != nullptr) {
        output = norm.forward(x_flattened);
    } else {
        output = x_flattened;
    }
    
    // Reshape back to spatial format if needed
    if (!flatten_embedding_) {
        output = output.reshape({B, H_out, W_out, embed_dim_});  // B, H', W', embed_dim
    }
    
    return output;
}

double PatchEmbedImpl::flops() const {
    // Calculate FLOPs for the convolution operation
    double flops = static_cast<double>(patches_resolution_.first) * 
                  static_cast<double>(patches_resolution_.second) * 
                  static_cast<double>(embed_dim_) * 
                  static_cast<double>(in_chans_) * 
                  static_cast<double>(patch_size_.first * patch_size_.second);
    
    // Add FLOPs for normalization if present
    if (norm.ptr() != nullptr) {
        flops += static_cast<double>(patches_resolution_.first * 
                                    patches_resolution_.second * 
                                    embed_dim_);
    }
    
    return flops;
}

} // namespace layers
} // namespace vggt