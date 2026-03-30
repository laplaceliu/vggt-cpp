#pragma once

#include <torch/torch.h>
#include <vector>
#include "block.h"
#include "patch_embed.h"
#include "mlp.h"
#include "layer_scale.h"

namespace vggt {
namespace layers {

class DinoVisionTransformerImpl : public torch::nn::Module {
public:
    DinoVisionTransformerImpl(
        int64_t img_size = 224,
        int64_t patch_size = 16,
        int64_t in_chans = 3,
        int64_t embed_dim = 768,
        int64_t depth = 12,
        int64_t num_heads = 12,
        double mlp_ratio = 4.0,
        bool qkv_bias = true,
        bool ffn_bias = true,
        bool proj_bias = true,
        double drop_path_rate = 0.0,
        bool drop_path_uniform = false,
        torch::Tensor init_values = {},
        torch::nn::AnyModule embed_layer = torch::nn::AnyModule(),
        torch::nn::AnyModule act_layer = torch::nn::AnyModule(),
        torch::nn::AnyModule block_fn = torch::nn::AnyModule(),
        std::string ffn_layer = "mlp",
        int64_t block_chunks = 1,
        int64_t num_register_tokens = 0,
        bool interpolate_antialias = false,
        double interpolate_offset = 0.1,
        bool qk_norm = false
    );

    torch::Tensor interpolate_pos_encoding(torch::Tensor x, int64_t w, int64_t h);

    torch::Tensor prepare_tokens_with_masks(torch::Tensor x, torch::Tensor masks);

    std::vector<torch::Tensor> forward_features_list(
        std::vector<torch::Tensor> x_list,
        std::vector<torch::Tensor> masks_list
    );

    torch::Tensor forward_features(torch::Tensor x, torch::Tensor masks);

    torch::Tensor forward(torch::Tensor x, torch::Tensor masks);

    std::vector<torch::Tensor> get_intermediate_layers(
        torch::Tensor x,
        int64_t n = 1,
        bool reshape = false,
        bool return_class_token = false,
        bool norm = true
    );

    void init_weights();

    // Properties
    int64_t num_features() const { return embed_dim_; }
    int64_t embed_dim() const { return embed_dim_; }
    int64_t num_patches() const { return num_patches_; }
    int64_t patch_size_val() const { return patch_size_; }

private:
    int64_t img_size_;
    int64_t patch_size_;
    int64_t in_chans_;
    int64_t embed_dim_;
    int64_t depth_;
    int64_t num_heads_;
    int64_t num_patches_;
    int64_t num_register_tokens_;
    bool interpolate_antialias_;
    double interpolate_offset_;
    bool use_reentrant_;
    bool chunked_blocks_;

    torch::nn::AnyModule patch_embed_;
    torch::Tensor cls_token;
    torch::Tensor pos_embed;
    torch::Tensor register_tokens;
    torch::Tensor mask_token;
    std::vector<torch::nn::AnyModule> blocks_;
    torch::nn::LayerNorm norm_{nullptr};
    torch::nn::Linear head_{nullptr};
};

TORCH_MODULE(DinoVisionTransformer);

// Factory functions
DinoVisionTransformer vit_small(
    int64_t patch_size = 16,
    int64_t num_register_tokens = 0,
    int64_t img_size = 224
);

DinoVisionTransformer vit_base(
    int64_t patch_size = 16,
    int64_t num_register_tokens = 0,
    int64_t img_size = 224
);

DinoVisionTransformer vit_large(
    int64_t patch_size = 16,
    int64_t num_register_tokens = 0,
    int64_t img_size = 224
);

DinoVisionTransformer vit_giant2(
    int64_t patch_size = 16,
    int64_t num_register_tokens = 0,
    int64_t img_size = 224
);

} // namespace layers
} // namespace vggt
