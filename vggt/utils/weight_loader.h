#pragma once

#include <string>
#include <unordered_map>
#include <torch/torch.h>

namespace vggt {
namespace utils {

/**
 * VGGT Weight Loader
 * 
 * Loads pretrained VGGT weights from PyTorch .pt files into the C++ model.
 * Handles key name mapping between Python and C++ naming conventions.
 */
class WeightLoader {
public:
    WeightLoader() = default;
    
    /**
     * Load weights from a PyTorch checkpoint file.
     * 
     * @param weight_path Path to the .pt or .pth file containing model weights
     * @return unordered_map of parameter name to tensor
     */
    static std::unordered_map<std::string, torch::Tensor> load_weights(const std::string& weight_path);
    
    /**
     * Convert Python-style state_dict to C++ model format.
     * Handles key name transformations between Python and C++ naming.
     * 
     * @param state_dict The Python model's state dictionary
     * @return Transformed state dictionary with C++ compatible key names
     */
    static std::unordered_map<std::string, torch::Tensor> convert_state_dict(
        const std::unordered_map<std::string, torch::Tensor>& state_dict);
    
    /**
     * Load weights directly into a VGGT model.
     * 
     * @param model The VGGT model to load weights into
     * @param weight_path Path to the .pt or .pth file
     * @return true if successful, false otherwise
     */
    static bool load_model_weights(
        torch::nn::Module& model, 
        const std::string& weight_path);
    
    /**
     * Print weight information for debugging.
     * 
     * @param state_dict The state dictionary to print
     */
    static void print_weight_info(
        const std::unordered_map<std::string, torch::Tensor>& state_dict);
    
    /**
     * Get expected weight keys for VGGT model.
     * Useful for debugging missing weights.
     */
    static void print_expected_keys();
};

} // namespace utils
} // namespace vggt
