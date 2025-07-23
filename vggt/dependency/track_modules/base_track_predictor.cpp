#include "base_track_predictor.h"
#include "utils.h"
#include <torch/torch.h>
#include <vector>

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
) : stride_(stride),
    latent_dim_(latent_dim),
    corr_levels_(corr_levels),
    corr_radius_(corr_radius),
    hidden_size_(hidden_size),
    fine_(fine),
    flows_emb_dim_(latent_dim / 2) {
    
    // Calculate transformer dimension
    transformer_dim_ = corr_levels_ * std::pow(corr_radius_ * 2 + 1, 2) + latent_dim_ * 2;

    if (fine_) {
        // This is the old dummy code, will be removed when training next model
        transformer_dim_ += (transformer_dim_ % 2 == 0) ? 4 : 5;
    } else {
        transformer_dim_ += (4 - transformer_dim_ % 4) % 4;
    }

    int64_t space_depth = use_spaceatt ? depth : 0;
    int64_t time_depth = depth;

    // Initialize updateformer
    updateformer_ = register_module("updateformer", 
        EfficientUpdateFormer(
            space_depth,
            time_depth,
            transformer_dim_,
            hidden_size_,
            latent_dim_ + 2,
            4.0,
            use_spaceatt
        )
    );

    // Initialize norm layer
    norm_ = register_module("norm", torch::nn::GroupNorm(1, latent_dim_));

    // Initialize feature updater
    ffeat_updater_ = register_module("ffeat_updater", 
        torch::nn::Sequential(
            torch::nn::Linear(latent_dim_, latent_dim_),
            torch::nn::GELU()
        )
    );

    // Initialize visibility predictor if not fine
    if (!fine_) {
        vis_predictor_ = register_module("vis_predictor", 
            torch::nn::Sequential(
                torch::nn::Linear(latent_dim_, 1)
            )
        );
    }
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor> BaseTrackerPredictorImpl::forward(
    torch::Tensor query_points,
    torch::Tensor fmaps,
    int64_t iters,
    bool return_feat,
    int64_t down_ratio
) {
    if (return_feat) {
        auto [coord_preds, vis_e, track_feats, query_track_feat] = forward_with_feat(query_points, fmaps, iters, down_ratio);
        return std::make_tuple(coord_preds, vis_e);
    }
    
    auto batch_size = query_points.size(0);
    auto num_tracks = query_points.size(1);
    auto dim = query_points.size(2);
    
    auto seq_len = fmaps.size(1);
    auto channels = fmaps.size(2);
    auto height = fmaps.size(3);
    auto width = fmaps.size(4);
    
    // Scale the input query_points
    if (down_ratio > 1) {
        query_points = query_points / static_cast<float>(down_ratio);
    }
    query_points = query_points / static_cast<float>(stride_);
    
    // Initialize coords with query points
    auto coords = query_points.clone().reshape({batch_size, 1, num_tracks, 2})
                  .repeat({1, seq_len, 1, 1});
    
    // Sample features of query points in query frame
    auto query_track_feat = sample_features4d(fmaps.index({torch::indexing::Slice(), 0}), coords.index({torch::indexing::Slice(), 0}));
    
    // Initialize track features with query features
    auto track_feats = query_track_feat.unsqueeze(1).repeat({1, seq_len, 1, 1});
    
    // Backup initial coordinates
    auto coords_backup = coords.clone();
    
    // Construct correlation block
    CorrBlock fcorr_fn(fmaps, corr_levels_, corr_radius_);
    
    std::vector<torch::Tensor> coord_preds;
    
    // Iterative refinement
    for (int64_t itr = 0; itr < iters; ++itr) {
        // Detach gradients from last iteration
        coords = coords.detach();
        
        // Compute correlation
        fcorr_fn.corr(track_feats);
        auto fcorrs = fcorr_fn.sample(coords);
        
        auto corrdim = fcorrs.size(3);
        
        auto fcorrs_ = fcorrs.permute({0, 2, 1, 3}).reshape({batch_size * num_tracks, seq_len, corrdim});
        
        // Movement of current coords relative to query points
        auto flows = (coords - coords.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)}))
                    .permute({0, 2, 1, 3}).reshape({batch_size * num_tracks, seq_len, 2});
        
        auto flows_emb = get_2d_embedding(flows, flows_emb_dim_, false);
        
        // Concatenate flows with embedding
        flows_emb = torch::cat({flows_emb, flows}, 2);
        
        auto track_feats_ = track_feats.permute({0, 2, 1, 3}).reshape({batch_size * num_tracks, seq_len, latent_dim_});
        
        // Concatenate as input for transformers
        auto transformer_input = torch::cat({flows_emb, fcorrs_, track_feats_}, 2);
        
        if (transformer_input.size(2) < transformer_dim_) {
            // Pad features to match dimension
            auto pad_dim = transformer_dim_ - transformer_input.size(2);
            auto pad = torch::zeros_like(flows_emb.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, pad_dim)}));
            transformer_input = torch::cat({transformer_input, pad}, 2);
        }
        
        // 2D positional embedding
        auto pos_embed = get_2d_sincos_pos_embed(transformer_dim_, std::make_tuple(height, width)).to(query_points.device());
        auto sampled_pos_emb = sample_features4d(pos_embed.expand({batch_size, -1, -1, -1}), coords.index({torch::indexing::Slice(), 0}));
        sampled_pos_emb = torch::rearrange(sampled_pos_emb, "b n c -> (b n) c").unsqueeze(1);
        
        auto x = transformer_input + sampled_pos_emb;
        
        // Reshape for transformer
        x = torch::rearrange(x, "(b n) s d -> b n s d", {{"b", batch_size}});
        
        // Compute delta coordinates and delta track features
        auto delta = updateformer_->forward(x);
        delta = torch::rearrange(delta, "b n s d -> (b n) s d", {{"b", batch_size}});
        auto delta_coords_ = delta.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 2)});
        auto delta_feats_ = delta.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(2, torch::indexing::None)});
        
        track_feats_ = track_feats_.reshape({batch_size * num_tracks * seq_len, latent_dim_});
        delta_feats_ = delta_feats_.reshape({batch_size * num_tracks * seq_len, latent_dim_});
        
        // Update track features
        track_feats_ = ffeat_updater_->forward(norm_->forward(delta_feats_)) + track_feats_;
        track_feats = track_feats_.reshape({batch_size, num_tracks, seq_len, latent_dim_}).permute({0, 2, 1, 3});
        
        // Update coordinates
        coords = coords + delta_coords_.reshape({batch_size, num_tracks, seq_len, 2}).permute({0, 2, 1, 3});
        
        // Force coord0 as query
        coords.index_put_({torch::indexing::Slice(), 0}, coords_backup.index({torch::indexing::Slice(), 0}));
        
        // Scale predicted tracks to original image scale
        if (down_ratio > 1) {
            coord_preds.push_back(coords * stride_ * down_ratio);
        } else {
            coord_preds.push_back(coords * stride_);
        }
    }
    
    // Calculate visibility if not fine
    torch::Tensor vis_e;
    if (!fine_) {
        vis_e = vis_predictor_->forward(track_feats.reshape({batch_size * seq_len * num_tracks, latent_dim_}))
                .reshape({batch_size, seq_len, num_tracks});
        vis_e = torch::sigmoid(vis_e);
    }
    
    return std::make_tuple(coord_preds, vis_e);
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor> BaseTrackerPredictorImpl::forward_with_feat(
    torch::Tensor query_points,
    torch::Tensor fmaps,
    int64_t iters,
    int64_t down_ratio
) {
    auto batch_size = query_points.size(0);
    auto num_tracks = query_points.size(1);
    auto dim = query_points.size(2);
    
    auto seq_len = fmaps.size(1);
    auto channels = fmaps.size(2);
    auto height = fmaps.size(3);
    auto width = fmaps.size(4);
    
    // Scale the input query_points
    if (down_ratio > 1) {
        query_points = query_points / static_cast<float>(down_ratio);
    }
    query_points = query_points / static_cast<float>(stride_);
    
    // Initialize coords with query points
    auto coords = query_points.clone().reshape({batch_size, 1, num_tracks, 2})
                  .repeat({1, seq_len, 1, 1});
    
    // Sample features of query points in query frame
    auto query_track_feat = sample_features4d(fmaps.index({torch::indexing::Slice(), 0}), coords.index({torch::indexing::Slice(), 0}));
    
    // Initialize track features with query features
    auto track_feats = query_track_feat.unsqueeze(1).repeat({1, seq_len, 1, 1});
    
    // Backup initial coordinates
    auto coords_backup = coords.clone();
    
    // Construct correlation block
    CorrBlock fcorr_fn(fmaps, corr_levels_, corr_radius_);
    
    std::vector<torch::Tensor> coord_preds;
    
    // Iterative refinement
    for (int64_t itr = 0; itr < iters; ++itr) {
        // Detach gradients from last iteration
        coords = coords.detach();
        
        // Compute correlation
        fcorr_fn.corr(track_feats);
        auto fcorrs = fcorr_fn.sample(coords);
        
        auto corrdim = fcorrs.size(3);
        
        auto fcorrs_ = fcorrs.permute({0, 2, 1, 3}).reshape({batch_size * num_tracks, seq_len, corrdim});
        
        // Movement of current coords relative to query points
        auto flows = (coords - coords.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)}))
                    .permute({0, 2, 1, 3}).reshape({batch_size * num_tracks, seq_len, 2});
        
        auto flows_emb = get_2d_embedding(flows, flows_emb_dim_, false);
        
        // Concatenate flows with embedding
        flows_emb = torch::cat({flows_emb, flows}, 2);
        
        auto track_feats_ = track_feats.permute({0, 2, 1, 3}).reshape({batch_size * num_tracks, seq_len, latent_dim_});
        
        // Concatenate as input for transformers
        auto transformer_input = torch::cat({flows_emb, fcorrs_, track_feats_}, 2);
        
        if (transformer_input.size(2) < transformer_dim_) {
            // Pad features to match dimension
            auto pad_dim = transformer_dim_ - transformer_input.size(2);
            auto pad = torch::zeros_like(flows_emb.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, pad_dim)}));
            transformer_input = torch::cat({transformer_input, pad}, 2);
        }
        
        // 2D positional embedding
        auto pos_embed = get_2d_sincos_pos_embed(transformer_dim_, std::make_tuple(height, width)).to(query_points.device());
        auto sampled_pos_emb = sample_features4d(pos_embed.expand({batch_size, -1, -1, -1}), coords.index({torch::indexing::Slice(), 0}));
        sampled_pos_emb = torch::rearrange(sampled_pos_emb, "b n c -> (b n) c").unsqueeze(1);
        
        auto x = transformer_input + sampled_pos_emb;
        
        // Reshape for transformer
        x = torch::rearrange(x, "(b n) s d -> b n s d", {{"b", batch_size}});
        
        // Compute delta coordinates and delta track features
        auto delta = updateformer_->forward(x);
        delta = torch::rearrange(delta, "b n s d -> (b n) s d", {{"b", batch_size}});
        auto delta_coords_ = delta.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 2)});
        auto delta_feats_ = delta.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(2, torch::indexing::None)});
        
        track_feats_ = track_feats_.reshape({batch_size * num_tracks * seq_len, latent_dim_});
        delta_feats_ = delta_feats_.reshape({batch_size * num_tracks * seq_len, latent_dim_});
        
        // Update track features
        track_feats_ = ffeat_updater_->forward(norm_->forward(delta_feats_)) + track_feats_;
        track_feats = track_feats_.reshape({batch_size, num_tracks, seq_len, latent_dim_}).permute({0, 2, 1, 3});
        
        // Update coordinates
        coords = coords + delta_coords_.reshape({batch_size, num_tracks, seq_len, 2}).permute({0, 2, 1, 3});
        
        // Force coord0 as query
        coords.index_put_({torch::indexing::Slice(), 0}, coords_backup.index({torch::indexing::Slice(), 0}));
        
        // Scale predicted tracks to original image scale
        if (down_ratio > 1) {
            coord_preds.push_back(coords * stride_ * down_ratio);
        } else {
            coord_preds.push_back(coords * stride_);
        }
    }
    
    // Calculate visibility if not fine
    torch::Tensor vis_e;
    if (!fine_) {
        vis_e = vis_predictor_->forward(track_feats.reshape({batch_size * seq_len * num_tracks, latent_dim_}))
                .reshape({batch_size, seq_len, num_tracks});
        vis_e = torch::sigmoid(vis_e);
    }
    
    return std::make_tuple(coord_preds, vis_e, track_feats, query_track_feat);
}

} // namespace track_modules
} // namespace dependency
} // namespace vggt