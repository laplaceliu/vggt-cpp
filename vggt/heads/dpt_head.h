#ifndef VGGT_HEADS_DPT_HEAD_H
#define VGGT_HEADS_DPT_HEAD_H

#include <vector>
#include <torch/torch.h>

namespace vggt {
namespace heads {

class DPTHeadImpl : public torch::nn::Module {
public:
    DPTHeadImpl(
        int dim_in,
        int patch_size = 14,
        int output_dim = 4,
        const std::string& activation = "inv_log",
        const std::string& conf_activation = "expp1",
        int features = 256,
        const std::vector<int>& out_channels = {256, 512, 1024, 1024},
        const std::vector<int>& intermediate_layer_idx = {4, 11, 17, 23},
        bool pos_embed = true,
        bool feature_only = false,
        int down_ratio = 1
    );

    torch::Tensor forward(
        const std::vector<torch::Tensor>& aggregated_tokens_list,
        const torch::Tensor& images,
        int patch_start_idx,
        int frames_chunk_size = 8
    );

private:
    torch::Tensor _forward_impl(
        const std::vector<torch::Tensor>& aggregated_tokens_list,
        const torch::Tensor& images,
        int patch_start_idx,
        int frames_start_idx = -1,
        int frames_end_idx = -1
    );

    torch::Tensor _apply_pos_embed(
        const torch::Tensor& x,
        int W,
        int H,
        float ratio = 0.1
    );

    torch::Tensor scratch_forward(
        const std::vector<torch::Tensor>& features
    );

    int patch_size;
    std::string activation;
    std::string conf_activation;
    bool pos_embed;
    bool feature_only;
    int down_ratio;
    std::vector<int> intermediate_layer_idx;

    torch::nn::LayerNorm norm;
    torch::nn::ModuleList projects;
    torch::nn::ModuleList resize_layers;
    torch::nn::Module scratch;
};

TORCH_MODULE(DPTHead);

} // namespace heads
} // namespace vggt

#endif // VGGT_HEADS_DPT_HEAD_H