#include "patch_embed.h"

namespace vggt {

PatchEmbedImpl::PatchEmbedImpl(
    int64_t img_size,
    int64_t patch_size,
    int64_t in_chans,
    int64_t embed_dim,
    torch::nn::AnyModule norm_layer,
    bool flatten_embedding
) : norm_(std::move(norm_layer)) {
    img_size_ = make_2tuple(img_size);
    patch_size_ = make_2tuple(patch_size);
    patches_resolution_ = std::make_tuple(
        std::get<0>(img_size_) / std::get<0>(patch_size_),
        std::get<1>(img_size_) / std::get<1>(patch_size_)
    );
    num_patches_ = std::get<0>(patches_resolution_) * std::get<1>(patches_resolution_);

    in_chans_ = in_chans;
    embed_dim_ = embed_dim;
    flatten_embedding_ = flatten_embedding;

    proj_ = register_module("proj", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_chans, embed_dim, {std::get<0>(patch_size_), std::get<1>(patch_size_)}).stride({std::get<0>(patch_size_), std::get<1>(patch_size_)})
    ));
    if (norm_.ptr()) {
        register_module("norm", norm_.ptr());
    }
}

torch::Tensor PatchEmbedImpl::forward(torch::Tensor x) {
    auto sizes = x.sizes();
    auto H = sizes[2];
    auto W = sizes[3];
    auto patch_H = std::get<0>(patch_size_);
    auto patch_W = std::get<1>(patch_size_);

    TORCH_CHECK(H % patch_H == 0, "Input image height ", H, " is not a multiple of patch height ", patch_H);
    TORCH_CHECK(W % patch_W == 0, "Input image width ", W, " is not a multiple of patch width ", patch_W);

    x = proj_->forward(x); // B C H W
    H = x.size(2);
    W = x.size(3);
    x = x.flatten(2).transpose(1, 2); // B HW C
    x = norm_.forward(x);
    if (!flatten_embedding_) {
        x = x.reshape(torch::IntArrayRef({-1, H, W, embed_dim_})); // B H W C
    }
    return x;
}

} // namespace vggt