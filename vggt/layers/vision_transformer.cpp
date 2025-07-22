/**
 * @file vision_transformer.cpp
 * @brief Implementation of Vision Transformer
 *
 * This file implements the DinoVisionTransformer class and related helper functions
 * for Vision Transformers based on the DINOv2 architecture.
 */

#include "vision_transformer.h"
#include <cmath>
#include <iostream>
#include <functional>

namespace vggt {
namespace layers {

torch::nn::Module named_apply(
    std::function<void(torch::nn::Module&, const std::string&)> fn,
    torch::nn::Module& module,
    const std::string& name,
    bool depth_first,
    bool include_root) {
    
    if (!depth_first && include_root) {
        fn(module, name);
    }
    
    for (auto& child : module.named_children()) {
        std::string child_name = name.empty() ? child.key() : name + "." + child.key();
        named_apply(fn, child.value(), child_name, depth_first, true);
    }
    
    if (depth_first && include_root) {
        fn(module, name);
    }
    
    return module;
}

torch::Tensor BlockChunkImpl::forward(const torch::Tensor& x) {
    torch::Tensor output = x;
    for (auto& block : *this) {
        output = block->as<torch::nn::Module>()->forward(output);
    }
    return output;
}

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
    c10::optional<double> init_values,
    const std::string& ffn_layer,
    int64_t block_chunks,
    int64_t num_register_tokens,
    bool interpolate_antialias,
    double interpolate_offset,
    bool qk_norm) {
    
    // Store parameters
    embed_dim_ = embed_dim;
    num_tokens_ = 1;
    n_blocks_ = depth;
    num_heads_ = num_heads;
    patch_size_ = patch_size;
    num_register_tokens_ = num_register_tokens;
    interpolate_antialias_ = interpolate_antialias;
    interpolate_offset_ = interpolate_offset;
    use_reentrant_ = false; // hardcoded to false
    
    // Create patch embedding
    patch_embed_ = PatchEmbed(
        PatchEmbedOptions()
            .img_size(img_size)
            .patch_size(patch_size)
            .in_chans(in_chans)
            .embed_dim(embed_dim));
    
    int64_t num_patches = patch_embed_->num_patches();
    
    // Create class token, position embedding, and register tokens
    cls_token_ = register_parameter("cls_token", torch::zeros({1, 1, embed_dim}));
    pos_embed_ = register_parameter("pos_embed", torch::zeros({1, num_patches + num_tokens_, embed_dim}));
    
    if (num_register_tokens > 0) {
        register_tokens_ = register_parameter("register_tokens", torch::zeros({1, num_register_tokens, embed_dim}));
    }
    
    // Create drop path rates
    std::vector<double> dpr;
    if (drop_path_uniform) {
        dpr.resize(depth, drop_path_rate);
    } else {
        for (int64_t i = 0; i < depth; ++i) {
            dpr.push_back(i * drop_path_rate / (depth - 1));
        }
    }
    
    // Create blocks
    std::vector<torch::nn::ModuleHolder<BlockImpl>> blocks_list;
    
    auto norm_layer = torch::nn::LayerNormOptions(embed_dim).eps(1e-6);
    
    // Create FFN factory based on ffn_layer type
    std::function<torch::nn::Module(int64_t, int64_t, torch::nn::Module, bool)> ffn_factory;
    
    if (ffn_layer == "mlp") {
        std::cout << "Using MLP layer as FFN" << std::endl;
        ffn_factory = [](int64_t dim, int64_t hidden_dim, torch::nn::Module act_layer, bool bias) {
            return Mlp(MlpOptions(dim, hidden_dim, dim).act_layer(act_layer).drop(0.0).bias(bias));
        };
    } else if (ffn_layer == "swiglu" || ffn_layer == "swiglufused") {
        std::cout << "Using SwiGLU layer as FFN" << std::endl;
        ffn_factory = [](int64_t dim, int64_t hidden_dim, torch::nn::Module act_layer, bool bias) {
            return SwiGLUFFNFused(SwiGLUFFNFusedOptions(dim, hidden_dim).bias(bias));
        };
    } else if (ffn_layer == "identity") {
        std::cout << "Using Identity layer as FFN" << std::endl;
        ffn_factory = [](int64_t dim, int64_t hidden_dim, torch::nn::Module act_layer, bool bias) {
            return torch::nn::Identity();
        };
    } else {
        throw std::runtime_error("Unknown FFN layer type: " + ffn_layer);
    }
    
    // Create blocks
    for (int64_t i = 0; i < depth; ++i) {
        auto block = Block(BlockOptions()
            .dim(embed_dim)
            .num_heads(num_heads)
            .mlp_ratio(mlp_ratio)
            .qkv_bias(qkv_bias)
            .proj_bias(proj_bias)
            .ffn_bias(ffn_bias)
            .drop_path(dpr[i])
            .norm_layer(torch::nn::LayerNorm(norm_layer))
            .act_layer(torch::nn::GELU())
            .ffn_layer(ffn_factory(embed_dim, static_cast<int64_t>(embed_dim * mlp_ratio), torch::nn::GELU(), ffn_bias))
            .init_values(init_values)
            .qk_norm(qk_norm));
        
        blocks_list.push_back(block);
    }
    
    // Create chunked blocks if needed
    if (block_chunks > 0) {
        chunked_blocks_ = true;
        std::vector<BlockChunk> chunked_blocks;
        int64_t chunksize = depth / block_chunks;
        
        for (int64_t i = 0; i < depth; i += chunksize) {
            auto chunk = BlockChunk();
            
            // Add identity blocks for proper indexing
            for (int64_t j = 0; j < i; ++j) {
                chunk->push_back(torch::nn::Identity());
            }
            
            // Add actual blocks
            for (int64_t j = i; j < std::min(i + chunksize, depth); ++j) {
                chunk->push_back(blocks_list[j]);
            }
            
            chunked_blocks.push_back(chunk);
        }
        
        blocks_ = torch::nn::ModuleList();
        for (auto& chunk : chunked_blocks) {
            blocks_->push_back(chunk);
        }
    } else {
        chunked_blocks_ = false;
        blocks_ = torch::nn::ModuleList();
        for (auto& block : blocks_list) {
            blocks_->push_back(block);
        }
    }
    
    // Create normalization layer, head, and mask token
    norm_ = torch::nn::LayerNorm(norm_layer);
    head_ = torch::nn::Identity();
    mask_token_ = register_parameter("mask_token", torch::zeros({1, embed_dim}));
    
    // Initialize weights
    init_weights();
}

void DinoVisionTransformerImpl::init_weights() {
    torch::nn::init::trunc_normal_(pos_embed_, 0.0, 0.02);
    torch::nn::init::normal_(cls_token_, 0.0, 1e-6);
    
    if (register_tokens_.defined()) {
        torch::nn::init::normal_(register_tokens_, 0.0, 1e-6);
    }
    
    named_apply(init_weights_vit_timm, *this);
}

torch::Tensor DinoVisionTransformerImpl::interpolate_pos_encoding(const torch::Tensor& x, int64_t w, int64_t h) {
    auto previous_dtype = x.scalar_type();
    int64_t npatch = x.size(1) - 1;
    int64_t N = pos_embed_.size(1) - 1;
    
    if (npatch == N && w == h) {
        return pos_embed_;
    }
    
    auto pos_embed = pos_embed_.to(torch::kFloat32);
    auto class_pos_embed = pos_embed.index({torch::indexing::Slice(), 0});
    auto patch_pos_embed = pos_embed.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)});
    int64_t dim = x.size(-1);
    
    int64_t w0 = w / patch_size_;
    int64_t h0 = h / patch_size_;
    int64_t M = static_cast<int64_t>(std::sqrt(N));
    
    TORCH_CHECK(N == M * M, "Number of patches must be a perfect square");
    
    torch::nn::functional::InterpolateFuncOptions options;
    options.mode(torch::kBicubic);
    options.align_corners(false);
    
    if (interpolate_antialias_) {
        options.antialias(true);
    }
    
    if (interpolate_offset_ > 0) {
        // Historical kludge: add a small number to avoid floating point error
        double sx = static_cast<double>(w0 + interpolate_offset_) / M;
        double sy = static_cast<double>(h0 + interpolate_offset_) / M;
        options.scale_factor(std::vector<double>{sx, sy});
    } else {
        options.size(std::vector<int64_t>{w0, h0});
    }
    
    auto patch_pos_embed_interp = torch::nn::functional::interpolate(
        patch_pos_embed.reshape({1, M, M, dim}).permute({0, 3, 1, 2}),
        options);
    
    TORCH_CHECK(patch_pos_embed_interp.size(2) == w0 && patch_pos_embed_interp.size(3) == h0,
               "Interpolated position encoding size mismatch");
    
    patch_pos_embed_interp = patch_pos_embed_interp.permute({0, 2, 3, 1}).reshape({1, -1, dim});
    
    return torch::cat({class_pos_embed.unsqueeze(0), patch_pos_embed_interp}, 1).to(previous_dtype);
}

torch::Tensor DinoVisionTransformerImpl::prepare_tokens_with_masks(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& masks) {
    
    auto B = x.size(0);
    auto nc = x.size(1);
    auto w = x.size(2);
    auto h = x.size(3);
    
    auto tokens = patch_embed_(x);
    
    if (masks.has_value()) {
        auto mask_value = mask_token_.to(tokens.dtype()).unsqueeze(0);
        tokens = torch::where(masks.value().unsqueeze(-1), mask_value, tokens);
    }
    
    tokens = torch::cat({cls_token_.expand({tokens.size(0), -1, -1}), tokens}, 1);
    tokens = tokens + interpolate_pos_encoding(tokens, w, h);
    
    if (register_tokens_.defined()) {
        tokens = torch::cat({
            tokens.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)}),
            register_tokens_.expand({tokens.size(0), -1, -1}),
            tokens.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)})
        }, 1);
    }
    
    return tokens;
}

std::vector<std::unordered_map<std::string, torch::Tensor>> DinoVisionTransformerImpl::forward_features_list(
    const std::vector<torch::Tensor>& x_list,
    const std::vector<torch::Tensor>& masks_list) {
    
    std::vector<torch::Tensor> x;
    for (size_t i = 0; i < x_list.size(); ++i) {
        x.push_back(prepare_tokens_with_masks(x_list[i], masks_list[i]));
    }
    
    for (auto& block_module : *blocks_) {
        auto& block = block_module.value();
        if (is_training()) {
            // In PyTorch, we would use checkpoint here, but in LibTorch we'll just call forward
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = block->as<torch::nn::Module>()->forward(x[i]);
            }
        } else {
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = block->as<torch::nn::Module>()->forward(x[i]);
            }
        }
    }
    
    std::vector<std::unordered_map<std::string, torch::Tensor>> output;
    for (size_t i = 0; i < x.size(); ++i) {
        auto x_norm = norm_(x[i]);
        std::unordered_map<std::string, torch::Tensor> item;
        
        item["x_norm_clstoken"] = x_norm.index({torch::indexing::Slice(), 0});
        item["x_norm_regtokens"] = x_norm.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(1, num_register_tokens_ + 1)
        });
        item["x_norm_patchtokens"] = x_norm.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(num_register_tokens_ + 1, torch::indexing::None)
        });
        item["x_prenorm"] = x[i];
        item["masks"] = masks_list[i];
        
        output.push_back(item);
    }
    
    return output;
}

std::unordered_map<std::string, torch::Tensor> DinoVisionTransformerImpl::forward_features(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& masks) {
    
    // Check if x is a list
    if (x.dim() == 1 && x.dtype() == torch::kInt64) {
        throw std::runtime_error("List input not supported in C++ implementation");
    }
    
    auto tokens = prepare_tokens_with_masks(x, masks);
    
    for (auto& block_module : *blocks_) {
        auto& block = block_module.value();
        if (is_training()) {
            // In PyTorch, we would use checkpoint here, but in LibTorch we'll just call forward
            tokens = block->as<torch::nn::Module>()->forward(tokens);
        } else {
            tokens = block->as<torch::nn::Module>()->forward(tokens);
        }
    }
    
    auto x_norm = norm_(tokens);
    
    std::unordered_map<std::string, torch::Tensor> output;
    output["x_norm_clstoken"] = x_norm.index({torch::indexing::Slice(), 0});
    output["x_norm_regtokens"] = x_norm.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(1, num_register_tokens_ + 1)
    });
    output["x_norm_patchtokens"] = x_norm.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(num_register_tokens_ + 1, torch::indexing::None)
    });
    output["x_prenorm"] = tokens;
    
    if (masks.has_value()) {
        output["masks"] = masks.value();
    }
    
    return output;
}

std::vector<torch::Tensor> DinoVisionTransformerImpl::_get_intermediate_layers_not_chunked(
    const torch::Tensor& x,
    const torch::Tensor& n) {
    
    auto tokens = prepare_tokens_with_masks(x);
    std::vector<torch::Tensor> output;
    int64_t total_block_len = blocks_->size();
    
    // Determine which blocks to take
    std::vector<int64_t> blocks_to_take;
    if (n.dim() == 0) {
        // n is a scalar, take the n last blocks
        int64_t n_val = n.item<int64_t>();
        for (int64_t i = total_block_len - n_val; i < total_block_len; ++i) {
            blocks_to_take.push_back(i);
        }
    } else {
        // n is a tensor of indices
        for (int64_t i = 0; i < n.size(0); ++i) {
            blocks_to_take.push_back(n[i].item<int64_t>());
        }
    }
    
    for (int64_t i = 0; i < total_block_len; ++i) {
        auto& block = (*blocks_)[i].value();
        tokens = block->as<torch::nn::Module>()->forward(tokens);
        
        if (std::find(blocks_to_take.begin(), blocks_to_take.end(), i) != blocks_to_take.end()) {
            output.push_back(tokens);
        }
    }
    
    TORCH_CHECK(output.size() == blocks_to_take.size(),
               "Only " + std::to_string(output.size()) + " / " + std::to_string(blocks_to_take.size()) + " blocks found");
    
    return output;
}

std::vector<torch::Tensor> DinoVisionTransformerImpl::_get_intermediate_layers_chunked(
    const torch::Tensor& x,
    const torch::Tensor& n) {
    
    auto tokens = prepare_tokens_with_masks(x);
    std::vector<torch::Tensor> output;
    int64_t i = 0;
    int64_t total_block_len = (*blocks_)[blocks_->size() - 1].value()->as<BlockChunk>()->size();
    
    // Determine which blocks to take
    std::vector<int64_t> blocks_to_take;
    if (n.dim() == 0) {
        // n is a scalar, take the n last blocks
        int64_t n_val = n.item<int64_t>();
        for (int64_t j = total_block_len - n_val; j < total_block_len; ++j) {
            blocks_to_take.push_back(j);
        }
    } else {
        // n is a tensor of indices
        for (int64_t j = 0; j < n.size(0); ++j) {
            blocks_to_take.push_back(n[j].item<int64_t>());
        }
    }
    
    for (auto& block_chunk_module : *blocks_) {
        auto& block_chunk = block_chunk_module.value()->as<BlockChunk>();
        
        for (int64_t j = i; j < block_chunk->size(); ++j) {
            auto& block = (*block_chunk)[j].value();
            tokens = block->as<torch::nn::Module>()->forward(tokens);
            
            if (std::find(blocks_to_take.begin(), blocks_to_take.end(), i) != blocks_to_take.end()) {
                output.push_back(tokens);
            }
            
            i++;
        }
    }
    
    TORCH_CHECK(output.size() == blocks_to_take.size(),
               "Only " + std::to_string(output.size()) + " / " + std::to_string(blocks_to_take.size()) + " blocks found");
    
    return output;
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> DinoVisionTransformerImpl::get_intermediate_layers(
    const torch::Tensor& x,
    const torch::Tensor& n,
    bool reshape,
    bool return_class_token,
    bool norm) {
    
    std::vector<torch::Tensor> outputs;
    if (chunked_blocks_) {
        outputs = _get_intermediate_layers_chunked(x, n);
    } else {
        outputs = _get_intermediate_layers_not_chunked(x, n);
    }
    
    if (norm) {
        for (auto& out : outputs) {
            out = norm_(out);
        }
    }
    
    std::vector<torch::Tensor> class_tokens;
    for (auto& out : outputs) {
        class_tokens.push_back(out.index({torch::indexing::Slice(), 0}));
    }
    
    for (auto& out : outputs) {
        out = out.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(1 + num_register_tokens_, torch::indexing::None)
        });
    }
    
    if (reshape) {
        auto B = x.size(0);
        auto w = x.size(2);
        auto h = x.size(3);
        
        for (auto& out : outputs) {
            out = out.reshape({B, w / patch_size_, h / patch_size_, -1})
                     .permute({0, 3, 1, 2})
                     .contiguous();
        }
    }
    
    return std::make_tuple(outputs, class_tokens);
}

torch::Tensor DinoVisionTransformerImpl::forward(
    torch::Tensor x,
    c10::optional<torch::Tensor> masks,
    bool is_training) {
    
    auto ret = forward_features(x, masks);
    
    if (is_training) {
        return ret["x_norm_clstoken"];
    } else {
        return head_->as<torch::nn::Module>()->forward(ret["x_norm_clstoken"]);
    }
}

void init_weights_vit_timm(torch::nn::Module& module, const std::string& name) {
    if (auto* linear = module.as<torch::nn::Linear>()) {
        torch::nn::init::trunc_normal_(linear->weight, 0.0, 0.02);
        if (linear->bias.defined()) {
            torch::nn::init::zeros_(linear->bias);
        }
    }
}

DinoVisionTransformer vit_small(int64_t patch_size, int64_t num_register_tokens) {
    return DinoVisionTransformer(
        DinoVisionTransformerOptions()
            .patch_size(patch_size)
            .embed_dim(384)
            .depth(12)
            .num_heads(6)
            .mlp_ratio(4.0)
            .num_register_tokens(num_register_tokens));
}

DinoVisionTransformer vit_base(int64_t patch_size, int64_t num_register_tokens) {
    return DinoVisionTransformer(
        DinoVisionTransformerOptions()
            .patch_size(patch_size)
            .embed_dim(768)
            .depth(12)
            .num_heads(12)
            .mlp_ratio(4.0)
            .num_register_tokens(num_register_tokens));
}

DinoVisionTransformer vit_large(int64_t patch_size, int64_t num_register_tokens) {
    return DinoVisionTransformer(
        DinoVisionTransformerOptions()
            .patch_size(patch_size)
            .embed_dim(1024)
            .depth(24)
            .num_heads(16)
            .mlp_ratio(4.0)
            .num_register_tokens(num_register_tokens));
}

DinoVisionTransformer vit_giant2(int64_t patch_size, int64_t num_register_tokens) {
    return DinoVisionTransformer(
        DinoVisionTransformerOptions()
            .patch_size(patch_size)
            .embed_dim(1536)
            .depth(40)
            .num_heads(24)
            .mlp_ratio(4.0)
            .num_register_tokens(num_register_tokens));
}

} // namespace layers
} // namespace vggt