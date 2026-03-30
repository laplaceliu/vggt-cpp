#include "vision_transformer.h"
#include "drop_path.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <random>

namespace vggt {
namespace layers {

namespace {

// Truncating normal initialization (custom implementation)
void trunc_normal_tensor_(torch::Tensor tensor, double std = 0.02) {
    auto size = tensor.sizes().vec();
    double a = std;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    auto n = tensor.numel();
    std::vector<double> w(n);
    for (auto& v : w) {
        v = std::tan(dis(gen) * M_PI * a);
        v = v / std::sqrt(1.0 + v * v);
        v = v * std::sqrt(2.0 / M_PI);
    }
    // Use .data() to bypass gradient tracking during initialization
    tensor.data().copy_(torch::tensor(w, tensor.options()).view(size));
}

void init_weights_vit_timm(torch::nn::Module& module, const std::string& name = "") {
    if (name.find("Linear") != std::string::npos) {
        auto* linear = module.as<torch::nn::Linear>();
        if (linear) {
            trunc_normal_tensor_(linear->weight, 0.02);
            if (linear->bias.defined()) {
                torch::nn::init::zeros_(linear->bias);
            }
        }
    }
}

void named_apply(
    std::function<void(torch::nn::Module&, const std::string&)> fn,
    torch::nn::Module& module,
    const std::string& name = "",
    bool depth_first = true,
    bool include_root = false
) {
    if (!depth_first && include_root) {
        fn(module, name);
    }
    for (auto& child : module.named_children()) {
        std::string child_name = name.empty() ? child.key() : name + "." + child.key();
        named_apply(fn, *child.value(), child_name, depth_first, true);
    }
    if (depth_first && include_root) {
        fn(module, name);
    }
}

} // anonymous namespace

DinoVisionTransformerImpl::DinoVisionTransformerImpl(
    int64_t img_size,
    int64_t patch_size,
    int64_t in_chans,
    int64_t embed_dim,
    int64_t depth,
    int64_t num_heads,
    double mlp_ratio,
    bool qkv_bias,
    bool ffn_bias,
    bool proj_bias,
    double drop_path_rate,
    bool drop_path_uniform,
    torch::Tensor init_values,
    torch::nn::AnyModule embed_layer,
    torch::nn::AnyModule act_layer,
    torch::nn::AnyModule block_fn,
    std::string ffn_layer,
    int64_t block_chunks,
    int64_t num_register_tokens,
    bool interpolate_antialias,
    double interpolate_offset,
    bool qk_norm
) : img_size_(img_size),
    patch_size_(patch_size),
    in_chans_(in_chans),
    embed_dim_(embed_dim),
    depth_(depth),
    num_heads_(num_heads),
    num_register_tokens_(num_register_tokens),
    interpolate_antialias_(interpolate_antialias),
    interpolate_offset_(interpolate_offset),
    use_reentrant_(false),
    chunked_blocks_(false) {

    // Set defaults
    torch::nn::AnyModule actual_embed_layer;
    if (embed_layer.is_empty()) {
        actual_embed_layer = torch::nn::AnyModule(PatchEmbed(
            img_size, patch_size, in_chans, embed_dim,
            torch::nn::AnyModule(), true
        ));
    } else {
        actual_embed_layer = embed_layer;
    }

    torch::nn::AnyModule actual_act_layer;
    if (act_layer.is_empty()) {
        actual_act_layer = torch::nn::AnyModule(torch::nn::GELU());
    } else {
        actual_act_layer = act_layer;
    }

    // Create norm_layer
    auto norm_layer = torch::nn::AnyModule(torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));

    // Create patch_embed using AnyModule
    patch_embed_ = actual_embed_layer;
    register_module("patch_embed", patch_embed_.ptr());

    // Calculate num_patches
    num_patches_ = (img_size / patch_size) * (img_size / patch_size);

    // Initialize cls_token
    cls_token = register_parameter("cls_token", torch::zeros({1, 1, embed_dim}));
    torch::nn::init::normal_(cls_token, /*std=*/1e-6);

    // Initialize pos_embed
    pos_embed = register_parameter("pos_embed", torch::zeros({1, num_patches_ + 1, embed_dim}));
    trunc_normal_tensor_(pos_embed, /*std=*/0.02);

    // Initialize register_tokens if num_register_tokens > 0
    if (num_register_tokens > 0) {
        register_tokens = register_parameter(
            "register_tokens",
            torch::zeros({1, num_register_tokens, embed_dim})
        );
        torch::nn::init::normal_(register_tokens, /*std=*/1e-6);
    } else {
        register_tokens = register_parameter("register_tokens", torch::zeros({0, 0, embed_dim}));
    }

    // Initialize mask_token
    mask_token = register_parameter("mask_token", torch::zeros({1, 1, embed_dim}));
    trunc_normal_tensor_(mask_token, /*std=*/0.02);

    // Calculate drop_path rates
    std::vector<double> dpr;
    if (drop_path_uniform) {
        dpr = std::vector<double>(depth, drop_path_rate);
    } else {
        for (int64_t i = 0; i < depth; i++) {
            dpr.push_back(drop_path_rate * i / (depth - 1));
        }
    }

    // Create blocks
    for (int64_t i = 0; i < depth; i++) {
        auto block = torch::nn::AnyModule(NestedTensorBlock(
            embed_dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            proj_bias,
            ffn_bias,
            0.0,  // drop
            0.0,  // attn_drop
            init_values,
            dpr[i],
            actual_act_layer,
            norm_layer,
            torch::nn::AnyModule(),  // attn_class
            torch::nn::AnyModule(),  // ffn_layer
            qk_norm,
            true,  // fused_attn
            torch::nn::AnyModule()   // rope
        ));
        blocks_.push_back(block);
        register_module("block_" + std::to_string(i), block.ptr());
    }

    // Create norm
    norm_ = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));

    // Create head
    head_ = register_module("head", torch::nn::Linear(embed_dim, embed_dim));

    init_weights();
}

void DinoVisionTransformerImpl::init_weights() {
    trunc_normal_tensor_(pos_embed, 0.02);
    torch::nn::init::normal_(cls_token, std::sqrt(1.0 / embed_dim_));
    if (num_register_tokens_ > 0 && register_tokens.numel() > 0) {
        torch::nn::init::normal_(register_tokens, std::sqrt(1.0 / embed_dim_));
    }
    named_apply(init_weights_vit_timm, *this);
}

torch::Tensor DinoVisionTransformerImpl::interpolate_pos_encoding(
    torch::Tensor x, int64_t w, int64_t h
) {
    auto previous_dtype = x.dtype();
    int64_t npatch = x.size(1) - 1;
    int64_t N = pos_embed.size(1) - 1;

    if (npatch == N && w == h) {
        return pos_embed;
    }

    auto pos_embed_float = pos_embed.to(torch::kFloat32);
    auto class_pos_embed = pos_embed_float.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1), torch::indexing::Slice()});
    auto patch_pos_embed = pos_embed_float.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()});

    int64_t dim = x.size(2);
    int64_t w0 = w / patch_size_;
    int64_t h0 = h / patch_size_;
    int64_t M = static_cast<int64_t>(std::sqrt(N));

    torch::Tensor patch_pos_embed_interp;
    if (interpolate_offset_ > 0.0) {
        double sx = static_cast<double>(w0 + interpolate_offset_) / M;
        double sy = static_cast<double>(h0 + interpolate_offset_) / M;
        patch_pos_embed_interp = torch::nn::functional::interpolate(
            patch_pos_embed.reshape({1, M, M, dim}).permute({0, 3, 1, 2}),
            torch::nn::functional::InterpolateFuncOptions()
                .scale_factor(std::vector<double>({sx, sy}))
                .mode(torch::kBicubic)
                .align_corners(false)
                .antialias(interpolate_antialias_)
        );
    } else {
        patch_pos_embed_interp = torch::nn::functional::interpolate(
            patch_pos_embed.reshape({1, M, M, dim}).permute({0, 3, 1, 2}),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>({w0, h0}))
                .mode(torch::kBicubic)
                .align_corners(false)
                .antialias(interpolate_antialias_)
        );
    }

    patch_pos_embed_interp = patch_pos_embed_interp.permute({0, 2, 3, 1}).view({1, -1, dim});
    return torch::cat({class_pos_embed, patch_pos_embed_interp}, /*dim=*/1).to(previous_dtype);
}

torch::Tensor DinoVisionTransformerImpl::prepare_tokens_with_masks(
    torch::Tensor x, torch::Tensor masks
) {
    auto B = x.size(0);
    auto w = x.size(2);
    auto h = x.size(3);

    x = patch_embed_.forward<torch::Tensor>(x);

    if (masks.defined() && masks.numel() > 0) {
        auto mask_token_expanded = mask_token.view({1, 1, -1});
        auto mask_reshaped = masks.unsqueeze(-1);
        x = torch::where(mask_reshaped, mask_token_expanded.to(x.dtype()), x);
    }

    // Add cls_token
    auto cls_tokens = cls_token.expand({B, -1, -1});
    x = torch::cat({cls_tokens, x}, /*dim=*/1);

    // Add positional encoding
    x = x + interpolate_pos_encoding(x, w, h);

    // Add register tokens if present
    if (num_register_tokens_ > 0 && register_tokens.numel() > 0) {
        auto reg_tokens = register_tokens.expand({B, -1, -1});
        auto x_cls = x.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1), torch::indexing::Slice()});
        auto x_patch = x.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()});
        std::vector<torch::Tensor> concat_tensors;
        concat_tensors.push_back(x_cls);
        concat_tensors.push_back(reg_tokens);
        concat_tensors.push_back(x_patch);
        x = torch::cat(concat_tensors, /*dim=*/1);
    }

    return x;
}

std::vector<torch::Tensor> DinoVisionTransformerImpl::forward_features_list(
    std::vector<torch::Tensor> x_list,
    std::vector<torch::Tensor> masks_list
) {
    std::vector<torch::Tensor> x;
    for (size_t i = 0; i < x_list.size(); i++) {
        x.push_back(prepare_tokens_with_masks(x_list[i], masks_list[i]));
    }

    for (auto& block : blocks_) {
        // Same as above - normal forward pass
        x[0] = block.forward<torch::Tensor>(x[0]);
    }

    std::vector<torch::Tensor> output;
    for (size_t i = 0; i < x.size(); i++) {
        output.push_back(norm_->forward(x[i]));
    }
    return output;
}

torch::Tensor DinoVisionTransformerImpl::forward_features(
    torch::Tensor x, torch::Tensor masks
) {
    x = prepare_tokens_with_masks(x, masks);

    for (auto& block : blocks_) {
        x = block.forward<torch::Tensor>(x);
    }

    return norm_->forward(x);
}

torch::Tensor DinoVisionTransformerImpl::forward(
    torch::Tensor x, torch::Tensor masks
) {
    auto ret = forward_features(x, masks);
    return head_->forward(ret.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1), torch::indexing::Slice()})).squeeze(1);
}

std::vector<torch::Tensor> DinoVisionTransformerImpl::get_intermediate_layers(
    torch::Tensor x,
    int64_t n,
    bool reshape,
    bool return_class_token,
    bool norm
) {
    x = prepare_tokens_with_masks(x, torch::Tensor());

    std::vector<torch::Tensor> outputs;

    std::vector<int64_t> blocks_to_take;
    if (n > 0) {
        for (int64_t i = depth_ - n; i < depth_; i++) {
            blocks_to_take.push_back(i);
        }
    }

    for (int64_t idx = 0; idx < depth_; idx++) {
        x = blocks_[idx].forward(x);
        if (std::find(blocks_to_take.begin(), blocks_to_take.end(), idx) != blocks_to_take.end()) {
            outputs.push_back(x);
        }
    }

    if (norm) {
        for (auto& out : outputs) {
            out = norm_->forward(out);
        }
    }

    if (reshape) {
        int64_t B = x.size(0);
        int64_t w = x.size(1) / patch_size_;
        int64_t h = x.size(2) / patch_size_;
        std::vector<torch::Tensor> reshaped_outputs;
        for (auto& out : outputs) {
            auto reshaped = out.index({torch::indexing::Slice(), torch::indexing::Slice(1 + num_register_tokens_, torch::indexing::None), torch::indexing::Slice()})
                .reshape({B, w, h, -1})
                .permute({0, 3, 1, 2})
                .contiguous();
            reshaped_outputs.push_back(reshaped);
        }
        outputs = reshaped_outputs;
    } else {
        for (auto& out : outputs) {
            out = out.index({torch::indexing::Slice(), torch::indexing::Slice(1 + num_register_tokens_, torch::indexing::None), torch::indexing::Slice()});
        }
    }

    return outputs;
}

// Factory functions
DinoVisionTransformer vit_small(int64_t patch_size, int64_t num_register_tokens, int64_t img_size) {
    return DinoVisionTransformer(DinoVisionTransformerImpl(
        img_size,                          // img_size
        patch_size,                        // patch_size
        3,                                 // in_chans
        384,                               // embed_dim
        12,                                // depth
        6,                                 // num_heads
        4.0,                               // mlp_ratio
        true,                              // qkv_bias
        true,                              // ffn_bias
        true,                              // proj_bias
        0.0,                               // drop_path_rate
        false,                             // drop_path_uniform
        {},                                // init_values
        torch::nn::AnyModule(),            // embed_layer
        torch::nn::AnyModule(),            // act_layer
        torch::nn::AnyModule(),            // block_fn
        "mlp",                             // ffn_layer
        1,                                 // block_chunks
        num_register_tokens,               // num_register_tokens
        false,                             // interpolate_antialias
        0.1,                               // interpolate_offset
        false                              // qk_norm
    ));
}

DinoVisionTransformer vit_base(int64_t patch_size, int64_t num_register_tokens, int64_t img_size) {
    return DinoVisionTransformer(DinoVisionTransformerImpl(
        img_size,                          // img_size
        patch_size,                        // patch_size
        3,                                 // in_chans
        768,                               // embed_dim
        12,                                // depth
        12,                               // num_heads
        4.0,                              // mlp_ratio
        true,                             // qkv_bias
        true,                             // ffn_bias
        true,                             // proj_bias
        0.0,                              // drop_path_rate
        false,                            // drop_path_uniform
        {},                               // init_values
        torch::nn::AnyModule(),           // embed_layer
        torch::nn::AnyModule(),           // act_layer
        torch::nn::AnyModule(),           // block_fn
        "mlp",                            // ffn_layer
        1,                                // block_chunks
        num_register_tokens,              // num_register_tokens
        false,                            // interpolate_antialias
        0.1,                              // interpolate_offset
        false                             // qk_norm
    ));
}

DinoVisionTransformer vit_large(int64_t patch_size, int64_t num_register_tokens, int64_t img_size) {
    return DinoVisionTransformer(DinoVisionTransformerImpl(
        img_size,                          // img_size
        patch_size,                       // patch_size
        3,                                // in_chans
        1024,                             // embed_dim
        24,                               // depth
        16,                               // num_heads
        4.0,                              // mlp_ratio
        true,                             // qkv_bias
        true,                             // ffn_bias
        true,                             // proj_bias
        0.0,                              // drop_path_rate
        false,                            // drop_path_uniform
        {},                               // init_values
        torch::nn::AnyModule(),           // embed_layer
        torch::nn::AnyModule(),           // act_layer
        torch::nn::AnyModule(),           // block_fn
        "mlp",                            // ffn_layer
        1,                                // block_chunks
        num_register_tokens,              // num_register_tokens
        false,                            // interpolate_antialias
        0.1,                              // interpolate_offset
        false                             // qk_norm
    ));
}

DinoVisionTransformer vit_giant2(int64_t patch_size, int64_t num_register_tokens, int64_t img_size) {
    return DinoVisionTransformer(DinoVisionTransformerImpl(
        img_size,                          // img_size
        patch_size,                       // patch_size
        3,                                // in_chans
        1536,                             // embed_dim
        40,                               // depth
        24,                               // num_heads
        4.0,                              // mlp_ratio
        true,                             // qkv_bias
        true,                             // ffn_bias
        true,                             // proj_bias
        0.0,                              // drop_path_rate
        false,                            // drop_path_uniform
        {},                               // init_values
        torch::nn::AnyModule(),           // embed_layer
        torch::nn::AnyModule(),           // act_layer
        torch::nn::AnyModule(),           // block_fn
        "mlp",                            // ffn_layer
        1,                                // block_chunks
        num_register_tokens,              // num_register_tokens
        false,                            // interpolate_antialias
        0.1,                              // interpolate_offset
        false                             // qk_norm
    ));
}

} // namespace layers
} // namespace vggt
