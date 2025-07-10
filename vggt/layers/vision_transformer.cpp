#include "vision_transformer.h"
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>

using namespace Eigen;

DinoVisionTransformer::DinoVisionTransformer(
    int img_size,
    int patch_size,
    int in_chans,
    int embed_dim,
    int depth,
    int num_heads,
    float mlp_ratio,
    bool qkv_bias,
    bool proj_bias,
    bool ffn_bias,
    float drop_path_rate,
    bool drop_path_uniform,
    float init_values,
    const std::string& ffn_layer,
    int block_chunks,
    int num_register_tokens,
    bool interpolate_antialias,
    float interpolate_offset,
    bool qk_norm
) : img_size_(img_size),
    patch_size_(patch_size),
    in_chans_(in_chans),
    embed_dim_(embed_dim),
    depth_(depth),
    num_heads_(num_heads),
    mlp_ratio_(mlp_ratio),
    num_register_tokens_(num_register_tokens),
    interpolate_antialias_(interpolate_antialias),
    interpolate_offset_(interpolate_offset),
    patch_embed_(img_size, patch_size, in_chans, embed_dim),
    norm_(embed_dim) {

    // Initialize cls token
    cls_token_ = MatrixXf::Zero(1, embed_dim);
    std::normal_distribution<float> dist(0.0f, 1e-6f);
    std::mt19937 gen;
    for (int i = 0; i < embed_dim; ++i) {
        cls_token_(0, i) = dist(gen);
    }

    // Initialize position embedding
    int num_patches = patch_embed_.get_num_patches();
    pos_embed_ = MatrixXf::Zero(1, num_patches + 1, embed_dim);
    // Apply trunc normal initialization (simplified)
    for (int i = 0; i < pos_embed_.size(); ++i) {
        pos_embed_.data()[i] = dist(gen) * 0.02f;
    }

    // Initialize register tokens if needed
    if (num_register_tokens_ > 0) {
        register_tokens_ = MatrixXf::Zero(1, num_register_tokens_, embed_dim);
        for (int i = 0; i < register_tokens_.size(); ++i) {
            register_tokens_.data()[i] = dist(gen);
        }
    }

    // Initialize mask token
    mask_token_ = MatrixXf::Zero(1, embed_dim);

    // Create drop path rates
    std::vector<float> dpr(depth_);
    if (drop_path_uniform) {
        std::fill(dpr.begin(), dpr.end(), drop_path_rate);
    } else {
        for (int i = 0; i < depth_; ++i) {
            dpr[i] = drop_path_rate * float(i) / float(depth_ - 1);
        }
    }

    // Create blocks
    for (int i = 0; i < depth_; ++i) {
        std::shared_ptr<Mlp> mlp;
        if (ffn_layer == "mlp") {
            mlp = std::make_shared<Mlp>(
                embed_dim,
                int(embed_dim * mlp_ratio),
                embed_dim,
                ffn_bias
            );
        } else if (ffn_layer == "swiglu") {
            mlp = std::make_shared<SwiGLUFFN>(
                embed_dim,
                int(embed_dim * mlp_ratio),
                embed_dim,
                ffn_bias
            );
        } else {
            throw std::invalid_argument("Unsupported ffn_layer");
        }

        blocks_.push_back(std::make_shared<Block>(
            embed_dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            proj_bias,
            ffn_bias,
            dpr[i],
            norm_,
            mlp,
            init_values,
            qk_norm
        ));
    }
}

Eigen::Tensor<float, 3> DinoVisionTransformer::interpolate_pos_encoding(
    const Eigen::Tensor<float, 4>& x, int w, int h) {

    int npatch = x.dimension(1) - 1;
    int N = pos_embed_.cols() - 1;
    if (npatch == N && w == h) {
        // No interpolation needed
        TensorMap<Tensor<float, 3>> pos_embed_tensor(
            pos_embed_.data(),
            1, pos_embed_.rows(), pos_embed_.cols()
        );
        return pos_embed_tensor;
    }

    // Get class and patch position embeddings
    MatrixXf class_pos_embed = pos_embed_.block(0, 0, 1, embed_dim_);
    MatrixXf patch_pos_embed = pos_embed_.block(0, 1, 1, N * embed_dim_);

    // Reshape and interpolate
    int w0 = w / patch_size_;
    int h0 = h / patch_size_;
    int M = static_cast<int>(std::sqrt(N));

    // TODO: Implement bicubic interpolation
    // For now, just return the original position embedding
    // (This is a simplified version - actual implementation would need proper interpolation)

    TensorMap<Tensor<float, 3>> pos_embed_tensor(
        pos_embed_.data(),
        1, pos_embed_.rows(), pos_embed_.cols()
    );
    return pos_embed_tensor;
}

Eigen::Tensor<float, 3> DinoVisionTransformer::prepare_tokens(const Eigen::Tensor<float, 4>& x) {
    int B = x.dimension(0);
    int w = x.dimension(2);
    int h = x.dimension(3);

    // Patch embedding
    auto x_embed = patch_embed_.forward(x);

    // Add cls token
    Tensor<float, 3> cls_tokens(B, 1, embed_dim_);
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < embed_dim_; ++i) {
            cls_tokens(b, 0, i) = cls_token_(0, i);
        }
    }
    x_embed = concatenate({cls_tokens, x_embed}, 1);

    // Add position embedding
    auto pos_embed = interpolate_pos_encoding(x, w, h);
    x_embed += pos_embed;

    // Add register tokens if needed
    if (num_register_tokens_ > 0) {
        Tensor<float, 3> reg_tokens(B, num_register_tokens_, embed_dim_);
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < num_register_tokens_; ++t) {
                for (int i = 0; i < embed_dim_; ++i) {
                    reg_tokens(b, t, i) = register_tokens_(0, t, i);
                }
            }
        }
        x_embed = concatenate({x_embed.slice(0, 1), reg_tokens, x_embed.slice(1, x_embed.dimension(1)-1)}, 1);
    }

    return x_embed;
}

Eigen::Tensor<float, 4> DinoVisionTransformer::forward(const Eigen::Tensor<float, 4>& x) {
    auto x_embed = prepare_tokens(x);

    // Apply transformer blocks
    for (auto& block : blocks_) {
        x_embed = block->forward(x_embed);
    }

    // Apply norm
    x_embed = norm_.forward(x_embed);

    return x_embed;
}
