#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <utility>
#include <functional>
#include <tuple>

namespace vggt {
namespace layers {
struct PairHash {
    size_t operator()(const std::pair<int64_t, int64_t>& key) const {
        auto hash_combine = [](size_t seed, size_t value) {
            return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
        };

        size_t seed = std::hash<int64_t>{}(key.first);
        seed = hash_combine(seed, std::hash<int64_t>{}(key.second));
        return seed;
    }
};

struct TupleHash {
    size_t operator()(const std::tuple<int64_t, int64_t, torch::Device, torch::Dtype>& key) const {
        auto hash_combine = [](size_t seed, size_t value) {
            return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
        };

        size_t seed = 0;
        seed = hash_combine(seed, std::hash<int64_t>{}(std::get<0>(key)));
        seed = hash_combine(seed, std::hash<int64_t>{}(std::get<1>(key)));
        seed = hash_combine(seed, std::hash<torch::Device>{}(std::get<2>(key)));
        seed = hash_combine(seed, std::hash<torch::Dtype>{}(std::get<3>(key)));
        return seed;
    }
};

class PositionGetter {
public:
    PositionGetter() = default;

    torch::Tensor operator()(int64_t batch_size, int64_t height, int64_t width, const torch::Device& device);

private:
    std::unordered_map<std::pair<int64_t, int64_t>, torch::Tensor, PairHash> position_cache_;
};

class RotaryPositionEmbedding2DImpl : public torch::nn::Module {
public:
    RotaryPositionEmbedding2DImpl(double frequency = 100.0, double scaling_factor = 1.0);

    torch::Tensor forward(torch::Tensor tokens, torch::Tensor positions);

private:
    std::pair<torch::Tensor, torch::Tensor> compute_frequency_components(
        int64_t dim, int64_t seq_len, const torch::Device& device, torch::Dtype dtype);

    static torch::Tensor rotate_features(torch::Tensor x);

    torch::Tensor apply_1d_rope(
        torch::Tensor tokens, torch::Tensor positions, torch::Tensor cos_comp, torch::Tensor sin_comp);

    double base_frequency_;
    double scaling_factor_;
    std::unordered_map<std::tuple<int64_t, int64_t, torch::Device, torch::Dtype>,
                       std::pair<torch::Tensor, torch::Tensor>, TupleHash> frequency_cache_;
};

TORCH_MODULE(RotaryPositionEmbedding2D);
}
} // namespace vggt
