#include "base_track_predictor.h"

namespace vggt {
namespace dependency {
namespace track_modules {

BaseTrackerPredictorImpl::BaseTrackerPredictorImpl(
    int64_t stride,
    int64_t corr_levels,
    int64_t corr_radius,
    int64_t latent_dim,
    int64_t hidden_size,
    bool use_spaceatt,
    int64_t depth,
    bool fine
) : stride(stride),
    latent_dim(latent_dim),
    corr_levels(corr_levels),
    corr_radius(corr_radius),
    hidden_size(hidden_size),
    fine(fine),
    flows_emb_dim(latent_dim / 2),
    transformer_dim(corr_levels * std::pow(corr_radius * 2 + 1, 2) + latent_dim * 2) {

    // Adjust transformer_dim to be divisible by 4
    if (fine) {
        // Old dummy code, will be removed when training next model
        transformer_dim += (transformer_dim % 2 == 0) ? 4 : 5;
    } else {
        transformer_dim += (4 - transformer_dim % 4) % 4;
    }

    int64_t space_depth = use_spaceatt ? depth : 0;
    int64_t time_depth = depth;

    updateformer = register_module("updateformer", EfficientUpdateFormer(
        space_depth,
        time_depth,
        transformer_dim,
        hidden_size,
        8, // num_heads (default value from Python)
        latent_dim + 2,
        4.0, // mlp_ratio
        use_spaceatt
    ));

    norm = register_module("norm", torch::nn::GroupNorm(1, latent_dim));

    // Linear layer to update track feats at each iteration
    ffeat_updater = register_module("ffeat_updater", torch::nn::Sequential(
        torch::nn::Linear(latent_dim, latent_dim),
        torch::nn::GELU()
    ));

    if (!fine) {
        vis_predictor = register_module("vis_predictor", torch::nn::Sequential(
            torch::nn::Linear(latent_dim, 1)
        ));
    }
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor> BaseTrackerPredictorImpl::forward_with_feat(
    torch::Tensor query_points,
    torch::Tensor fmaps,
    int64_t iters,
    int64_t down_ratio
) {
    // 直接调用 forward 函数，并手动实现返回特征的逻辑
    auto B = query_points.size(0);
    auto N = query_points.size(1);
    auto D = query_points.size(2);

    auto S = fmaps.size(1);
    auto C = fmaps.size(2);
    auto HH = fmaps.size(3);
    auto WW = fmaps.size(4);

    assert(D == 2);

    // 缩放输入查询点
    torch::Tensor scaled_query_points = query_points.clone();
    if (down_ratio > 1) {
        scaled_query_points = scaled_query_points / static_cast<float>(down_ratio);
    }
    scaled_query_points = scaled_query_points / static_cast<float>(stride);

    // 使用查询点初始化坐标
    auto coords = scaled_query_points.clone().reshape({B, 1, N, 2}).repeat({1, S, 1, 1});

    // 在查询帧中采样查询点的特征
    auto query_track_feat = sample_features4d(fmaps.index({torch::indexing::Slice(), 0}), coords.index({torch::indexing::Slice(), 0}));

    // 使用查询特征初始化轨迹特征
    auto track_feats = query_track_feat.unsqueeze(1).repeat({1, S, 1, 1});  // B, S, N, C

    // 调用 forward 函数获取坐标预测和可见性估计
    auto result = forward(query_points, fmaps, iters, false, down_ratio);
    auto [coord_preds, vis_e] = std::get<std::tuple<std::vector<torch::Tensor>, torch::Tensor>>(result);

    return {coord_preds, vis_e, track_feats, query_track_feat};
}

std::variant<
    std::tuple<std::vector<torch::Tensor>, torch::Tensor>,
    std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor>
> BaseTrackerPredictorImpl::forward(
    torch::Tensor query_points,
    torch::Tensor fmaps,
    int64_t iters,
    bool return_feat,
    int64_t down_ratio
) {
    auto B = query_points.size(0);
    auto N = query_points.size(1);
    auto D = query_points.size(2);

    auto S = fmaps.size(1);
    auto C = fmaps.size(2);
    auto HH = fmaps.size(3);
    auto WW = fmaps.size(4);

    assert(D == 2);

    // Scale the input query_points
    if (down_ratio > 1) {
        query_points = query_points / static_cast<float>(down_ratio);
    }
    query_points = query_points / static_cast<float>(stride);

    // Initialize coords with query points
    auto coords = query_points.clone().reshape({B, 1, N, 2}).repeat({1, S, 1, 1});

    // Sample features of query points in query frame
    auto query_track_feat = sample_features4d(fmaps.index({torch::indexing::Slice(), 0}), coords.index({torch::indexing::Slice(), 0}));

    // Initialize track feats with query feats
    auto track_feats = query_track_feat.unsqueeze(1).repeat({1, S, 1, 1});  // B, S, N, C

    // Backup initial coords
    auto coords_backup = coords.clone();

    // Construct correlation block
    CorrBlock fcorr_fn(fmaps, corr_levels, corr_radius);

    std::vector<torch::Tensor> coord_preds;

    // Iterative refinement
    for (int64_t itr = 0; itr < iters; ++itr) {
        // Detach gradients from last iteration
        coords = coords.detach();

        // Compute correlation
        fcorr_fn.corr(track_feats);
        auto fcorrs = fcorr_fn.sample(coords);  // B, S, N, corrdim

        auto corrdim = fcorrs.size(3);

        // Reshape fcorrs: B, S, N, corrdim -> B*N, S, corrdim
        auto fcorrs_ = einops::rearrange(fcorrs, "b s n c -> (b n) s c");

        // Movement of current coords relative to query points
        auto flows = einops::rearrange(coords - coords.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)}),
                                      "b s n c -> (b n) s c");

        // Get 2D embedding for flows
        auto flows_emb = get_2d_embedding(flows, flows_emb_dim, false);

        // Concatenate flows with embedding
        flows_emb = torch::cat({flows_emb, flows}, -1);

        // Reshape track_feats: B, S, N, latent_dim -> B*N, S, latent_dim
        auto track_feats_ = einops::rearrange(track_feats, "b s n c -> (b n) s c");

        // Concatenate as input for transformer
        auto transformer_input = torch::cat({flows_emb, fcorrs_, track_feats_}, 2);

        // Pad features if needed
        if (transformer_input.size(2) < transformer_dim) {
            auto pad_dim = transformer_dim - transformer_input.size(2);
            auto pad = torch::zeros_like(flows_emb.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, pad_dim)}));
            transformer_input = torch::cat({transformer_input, pad}, 2);
        }

        // 2D positional embedding
        auto pos_embed = get_2d_sincos_pos_embed(transformer_dim, {HH, WW}).to(query_points.device());
        auto sampled_pos_emb = sample_features4d(pos_embed.expand({B, -1, -1, -1}), coords.index({torch::indexing::Slice(), 0}));
        sampled_pos_emb = einops::rearrange(sampled_pos_emb, "b n c -> (b n) c").unsqueeze(1);

        auto x = transformer_input + sampled_pos_emb;

        // Reshape: B*N, S, C -> B, N, S, C
        x = einops::rearrange(x, "(b n) s d -> b n s d", einops::axis("b", B));

        // Compute delta coordinates and delta track features
        auto delta = updateformer->forward(x);

        // Reshape: B, N, S, C -> B*N, S, C
        delta = einops::rearrange(delta, "b n s d -> (b n) s d", einops::axis("b", B));

        auto delta_coords_ = delta.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 2)});
        auto delta_feats_ = delta.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(2)});

        // Reshape for feature update
        track_feats_ = track_feats_.reshape({B * N * S, latent_dim});
        delta_feats_ = delta_feats_.reshape({B * N * S, latent_dim});

        // Update track features
        track_feats_ = ffeat_updater->forward(norm->forward(delta_feats_)) + track_feats_;
        track_feats = track_feats_.reshape({B, N, S, latent_dim}).permute({0, 2, 1, 3});  // B x S x N x C

        // Update coordinates
        coords = coords + delta_coords_.reshape({B, N, S, 2}).permute({0, 2, 1, 3});

        // Force coord0 as query
        coords.index_put_({torch::indexing::Slice(), 0}, coords_backup.index({torch::indexing::Slice(), 0}));

        // Scale coordinates back to original image scale
        if (down_ratio > 1) {
            coord_preds.push_back(coords * stride * down_ratio);
        } else {
            coord_preds.push_back(coords * stride);
        }
    }

    torch::Tensor vis_e;
    if (!fine) {
        vis_e = vis_predictor->forward(track_feats.reshape({B * S * N, latent_dim})).reshape({B, S, N});
        vis_e = torch::sigmoid(vis_e);
    }

    if (return_feat) {
        return std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor>{coord_preds, vis_e, track_feats, query_track_feat};
    } else {
        return std::tuple<std::vector<torch::Tensor>, torch::Tensor>{coord_preds, vis_e};
    }
}

} // namespace track_modules
} // namespace dependency
} // namespace vggt
