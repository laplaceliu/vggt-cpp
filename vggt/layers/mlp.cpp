#include "mlp.h"

namespace vggt {
namespace layers {

MlpImpl::MlpImpl(int64_t in_features, int64_t hidden_features, int64_t out_features,
                 torch::nn::AnyModule act_layer, double drop, bool bias)
    : act(std::move(act_layer)) {
    out_features = out_features == -1 ? in_features : out_features;
    hidden_features = hidden_features == -1 ? in_features : hidden_features;

    fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(in_features, hidden_features).bias(bias)));
    register_module("act", act.ptr());
    drop1 = register_module("drop1", torch::nn::Dropout(drop));
    fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(hidden_features, out_features).bias(bias)));
    drop2 = register_module("drop2", torch::nn::Dropout(drop));
}

torch::Tensor MlpImpl::forward(torch::Tensor x) {
    x = fc1->forward(x);
    x = act.forward(x);
    x = drop1->forward(x);
    x = fc2->forward(x);
    x = drop2->forward(x);
    return x;
}

} // namespace layers
} // namespace vggt