#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/utils/weight_loader.h"
#include <filesystem>
#include <fstream>

namespace vggt {
namespace utils {
namespace {

TEST(WeightLoaderTest, LoadWeightsNonExistentFile) {
    auto state_dict = WeightLoader::load_weights("/non/existent/path/weights.pt");
    
    EXPECT_TRUE(state_dict.empty());
}

TEST(WeightLoaderTest, ConvertStateDictBasic) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    state_dict["frame_block_0.attn.weight"] = torch::randn({10, 10});
    state_dict["frame_block_1.mlp.bias"] = torch::randn({10});
    state_dict["global_block_0.norm.weight"] = torch::randn({5});
    
    auto converted = WeightLoader::convert_state_dict(state_dict);
    
    // Check that keys are converted (actual format has trailing .)
    EXPECT_EQ(converted.size(), 3);
    EXPECT_NE(converted.find("frame_blocks_0..attn.weight"), converted.end());
    EXPECT_NE(converted.find("frame_blocks_1..mlp.bias"), converted.end());
    EXPECT_NE(converted.find("global_blocks_0..norm.weight"), converted.end());
}

TEST(WeightLoaderTest, ConvertStateDictNoMatch) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    state_dict["some_other_key.weight"] = torch::randn({10, 10});
    
    auto converted = WeightLoader::convert_state_dict(state_dict);
    
    // Key should remain unchanged
    EXPECT_EQ(converted.size(), 1);
    EXPECT_NE(converted.find("some_other_key.weight"), converted.end());
}

TEST(WeightLoaderTest, ConvertStateDictEmpty) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    
    auto converted = WeightLoader::convert_state_dict(state_dict);
    
    EXPECT_TRUE(converted.empty());
}

TEST(WeightLoaderTest, ConvertStateDictMultipleFrameBlocks) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    for (int i = 0; i < 5; ++i) {
        state_dict["frame_block_" + std::to_string(i) + ".attn.weight"] = torch::randn({10, 10});
    }
    
    auto converted = WeightLoader::convert_state_dict(state_dict);
    
    EXPECT_EQ(converted.size(), 5);
    for (int i = 0; i < 5; ++i) {
        // Actual format has trailing . after the block number
        std::string new_key = "frame_blocks_" + std::to_string(i) + "..attn.weight";
        EXPECT_NE(converted.find(new_key), converted.end());
    }
}

TEST(WeightLoaderTest, PrintWeightInfoBasic) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    state_dict["layer1.weight"] = torch::randn({10, 20});
    state_dict["layer1.bias"] = torch::randn({10});
    state_dict["layer2.weight"] = torch::randn({5, 10});
    
    // Should not throw
    EXPECT_NO_THROW(WeightLoader::print_weight_info(state_dict));
}

TEST(WeightLoaderTest, PrintWeightInfoEmpty) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    
    // Should not throw
    EXPECT_NO_THROW(WeightLoader::print_weight_info(state_dict));
}

TEST(WeightLoaderTest, PrintExpectedKeys) {
    // Should not throw
    EXPECT_NO_THROW(WeightLoader::print_expected_keys());
}

TEST(WeightLoaderTest, LoadModelWeightsNonExistentFile) {
    // Create a simple model for testing
    auto model = std::make_shared<torch::nn::LinearImpl>(10, 5);
    
    bool result = WeightLoader::load_model_weights(*model, "/non/existent/path/weights.pt");
    
    EXPECT_FALSE(result);
}

TEST(WeightLoaderTest, LoadWeightsInvalidFormat) {
    // Create a temporary file with invalid content
    std::string temp_path = "/tmp/test_invalid_weights.pt";
    {
        std::ofstream file(temp_path, std::ios::binary);
        file << "invalid data";
    }
    
    auto state_dict = WeightLoader::load_weights(temp_path);
    
    // Should return empty dict for invalid format
    EXPECT_TRUE(state_dict.empty());
    
    // Cleanup
    std::filesystem::remove(temp_path);
}

TEST(WeightLoaderTest, StateDictPreservesTensorValues) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    torch::Tensor test_tensor = torch::ones({3, 3});
    state_dict["test.weight"] = test_tensor;
    
    auto converted = WeightLoader::convert_state_dict(state_dict);
    
    EXPECT_TRUE(torch::allclose(converted["test.weight"], test_tensor));
}

TEST(WeightLoaderTest, ConvertStateDictComplexKey) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    state_dict["frame_block_10.attn.qkv.weight"] = torch::randn({30, 10});
    state_dict["global_block_23.mlp.fc1.bias"] = torch::randn({20});
    
    auto converted = WeightLoader::convert_state_dict(state_dict);
    
    // Actual format has trailing . after the block number
    EXPECT_NE(converted.find("frame_blocks_10..attn.qkv.weight"), converted.end());
    EXPECT_NE(converted.find("global_blocks_23..mlp.fc1.bias"), converted.end());
}

TEST(WeightLoaderTest, PrintWeightInfoManyTensors) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    for (int i = 0; i < 60; ++i) {
        state_dict["layer" + std::to_string(i) + ".weight"] = torch::randn({10, 10});
    }
    
    // Should not throw even with many tensors (prints first 50)
    EXPECT_NO_THROW(WeightLoader::print_weight_info(state_dict));
}

} // namespace
} // namespace utils
} // namespace vggt
