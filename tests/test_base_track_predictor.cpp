#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/dependency/track_modules/base_track_predictor.h"

namespace vggt {
namespace dependency {
namespace track_modules {
namespace {

using ReturnVariant = std::variant<
    std::tuple<std::vector<torch::Tensor>, torch::Tensor>,
    std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor>
>;

TEST(BaseTrackPredictorTest, ConstructorDefault) {
    torch::manual_seed(42);
    BaseTrackerPredictor predictor;
    EXPECT_TRUE(predictor);
}

TEST(BaseTrackPredictorTest, ConstructorWithParams) {
    torch::manual_seed(42);
    BaseTrackerPredictor predictor(4, 5, 4, 128, 384, true, 6, false);
    EXPECT_TRUE(predictor);
}

TEST(BaseTrackPredictorTest, ConstructorFineMode) {
    torch::manual_seed(42);
    BaseTrackerPredictor predictor(1, 3, 3, 32, 256, false, 4, true);
    EXPECT_TRUE(predictor);
}

TEST(BaseTrackPredictorTest, ForwardBasic) {
    torch::manual_seed(42);
    BaseTrackerPredictor predictor;
    EXPECT_TRUE(predictor);

    torch::Tensor query_points = torch::tensor({{{0.5, 0.5}, {0.25, 0.75}}});
    torch::Tensor fmaps = torch::rand({1, 2, 128, 16, 16});

    // Call forward directly through impl pointer
    BaseTrackerPredictorImpl* impl = predictor.ptr().get();
    ReturnVariant result = impl->forward(query_points, fmaps, 4, false, 1);
    auto result_tuple = std::get<std::tuple<std::vector<torch::Tensor>, torch::Tensor>>(result);
    
    auto coord_preds = std::get<0>(result_tuple);
    auto vis_e = std::get<1>(result_tuple);

    EXPECT_EQ(coord_preds.size(), 4);
    EXPECT_EQ(vis_e.dim(), 3);
    EXPECT_EQ(vis_e.size(0), 1);
    EXPECT_EQ(vis_e.size(1), 2);
    EXPECT_EQ(vis_e.size(2), 2);
}

TEST(BaseTrackPredictorTest, ForwardWithFeat) {
    torch::manual_seed(42);
    BaseTrackerPredictor predictor;
    EXPECT_TRUE(predictor);

    torch::Tensor query_points = torch::tensor({{{0.5, 0.5}}});
    torch::Tensor fmaps = torch::rand({1, 3, 128, 16, 16});

    auto [coord_preds, vis_e, track_feats, query_track_feat] = predictor->forward_with_feat(
        query_points, fmaps, 4, 1
    );

    EXPECT_EQ(coord_preds.size(), 4);
    EXPECT_TRUE(track_feats.defined());
    EXPECT_TRUE(query_track_feat.defined());
}

TEST(BaseTrackPredictorTest, ForwardDifferentIters) {
    torch::manual_seed(42);
    BaseTrackerPredictor predictor;
    EXPECT_TRUE(predictor);

    torch::Tensor query_points = torch::tensor({{{0.5, 0.5}}});
    torch::Tensor fmaps = torch::rand({1, 2, 128, 16, 16});

    BaseTrackerPredictorImpl* impl = predictor.ptr().get();
    ReturnVariant result1 = impl->forward(query_points, fmaps, 2, false, 1);
    ReturnVariant result2 = impl->forward(query_points, fmaps, 6, false, 1);

    auto coord_preds1 = std::get<0>(std::get<std::tuple<std::vector<torch::Tensor>, torch::Tensor>>(result1));
    auto coord_preds2 = std::get<0>(std::get<std::tuple<std::vector<torch::Tensor>, torch::Tensor>>(result2));

    EXPECT_EQ(coord_preds1.size(), 2);
    EXPECT_EQ(coord_preds2.size(), 6);
}

TEST(BaseTrackPredictorTest, ForwardWithDownRatio) {
    torch::manual_seed(42);
    BaseTrackerPredictor predictor;
    EXPECT_TRUE(predictor);

    torch::Tensor query_points = torch::tensor({{{32.0, 32.0}}});
    torch::Tensor fmaps = torch::rand({1, 2, 128, 16, 16});

    BaseTrackerPredictorImpl* impl = predictor.ptr().get();
    ReturnVariant result = impl->forward(query_points, fmaps, 4, false, 4);
    auto coord_preds = std::get<0>(std::get<std::tuple<std::vector<torch::Tensor>, torch::Tensor>>(result));

    EXPECT_FALSE(coord_preds.empty());
}

TEST(BaseTrackPredictorTest, ForwardMultipleQueryPoints) {
    torch::manual_seed(42);
    BaseTrackerPredictor predictor;
    EXPECT_TRUE(predictor);

    torch::Tensor query_points = torch::rand({2, 10, 2});
    torch::Tensor fmaps = torch::rand({2, 3, 128, 16, 16});

    BaseTrackerPredictorImpl* impl = predictor.ptr().get();
    ReturnVariant result = impl->forward(query_points, fmaps, 4, false, 1);
    auto vis_e = std::get<1>(std::get<std::tuple<std::vector<torch::Tensor>, torch::Tensor>>(result));

    EXPECT_EQ(vis_e.size(0), 2);
    EXPECT_EQ(vis_e.size(1), 3);
    EXPECT_EQ(vis_e.size(2), 10);
}

TEST(BaseTrackPredictorTest, ForwardMultipleFrames) {
    torch::manual_seed(42);
    BaseTrackerPredictor predictor;
    EXPECT_TRUE(predictor);

    torch::Tensor query_points = torch::tensor({{{0.5, 0.5}}});
    torch::Tensor fmaps = torch::rand({1, 8, 128, 16, 16});

    BaseTrackerPredictorImpl* impl = predictor.ptr().get();
    ReturnVariant result = impl->forward(query_points, fmaps, 4, false, 1);
    auto coord_preds = std::get<0>(std::get<std::tuple<std::vector<torch::Tensor>, torch::Tensor>>(result));

    EXPECT_EQ(coord_preds.back().size(1), 8);
}

TEST(BaseTrackPredictorTest, ForwardFineModeNoVis) {
    torch::manual_seed(42);
    BaseTrackerPredictor predictor(4, 5, 4, 128, 384, true, 6, true);
    EXPECT_TRUE(predictor);

    torch::Tensor query_points = torch::tensor({{{0.5, 0.5}}});
    torch::Tensor fmaps = torch::rand({1, 2, 128, 16, 16});

    BaseTrackerPredictorImpl* impl = predictor.ptr().get();
    ReturnVariant result = impl->forward(query_points, fmaps, 4, false, 1);
    auto [coord_preds, vis_e] = std::get<std::tuple<std::vector<torch::Tensor>, torch::Tensor>>(result);

    EXPECT_FALSE(coord_preds.empty());
}

TEST(BaseTrackPredictorTest, ForwardPreservesGrad) {
    torch::manual_seed(42);
    BaseTrackerPredictor predictor;
    EXPECT_TRUE(predictor);

    torch::Tensor query_points = torch::tensor({{{0.5, 0.5}}}, torch::TensorOptions().requires_grad(true));
    torch::Tensor fmaps = torch::rand({1, 2, 128, 16, 16});

    BaseTrackerPredictorImpl* impl = predictor.ptr().get();
    ReturnVariant result = impl->forward(query_points, fmaps, 4, false, 1);
    auto coord_preds = std::get<0>(std::get<std::tuple<std::vector<torch::Tensor>, torch::Tensor>>(result));

    coord_preds.back().sum().backward();
    EXPECT_TRUE(query_points.grad().defined());
}

} // namespace
} // namespace track_modules
} // namespace dependency
} // namespace vggt
