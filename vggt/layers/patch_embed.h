#pragma once

#include <torch/torch.h>

namespace vggt {
namespace layers {
inline std::tuple<int64_t, int64_t> make_2tuple(int64_t x) {
    return std::make_tuple(x, x);
}

inline std::tuple<int64_t, int64_t> make_2tuple(const std::tuple<int64_t, int64_t>& x) {
    return x;
}

class PatchEmbedImpl : public torch::nn::Module {
public:
    PatchEmbedImpl(
        int64_t img_size = 224,
        int64_t patch_size = 16,
        int64_t in_chans = 3,
        int64_t embed_dim = 768,
        torch::nn::AnyModule norm_layer = torch::nn::AnyModule(),
        bool flatten_embedding = true
    );

    torch::Tensor forward(torch::Tensor x);

private:
    std::tuple<int64_t, int64_t> img_size_;
    std::tuple<int64_t, int64_t> patch_size_;
    std::tuple<int64_t, int64_t> patches_resolution_;
    int64_t num_patches_;
    int64_t in_chans_;
    int64_t embed_dim_;
    bool flatten_embedding_;
    torch::nn::Conv2d proj_{nullptr};
    torch::nn::AnyModule norm_;
};

TORCH_MODULE(PatchEmbed);
}
} // namespace vggt
