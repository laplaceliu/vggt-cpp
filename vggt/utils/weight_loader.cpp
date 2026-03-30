#include "weight_loader.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>

namespace vggt {
namespace utils {

std::unordered_map<std::string, torch::Tensor> WeightLoader::load_weights(
    const std::string& weight_path) {
    
    std::unordered_map<std::string, torch::Tensor> state_dict;
    
    try {
        // First check if this is a ZIP archive (safetensors format from Hugging Face)
        std::ifstream file_check(weight_path, std::ios::binary);
        if (!file_check.is_open()) {
            std::cerr << "Error: Could not open file: " << weight_path << std::endl;
            return {};
        }
        
        // Read first 4 bytes to check for ZIP signature
        char header[4];
        file_check.read(header, 4);
        file_check.close();
        
        bool is_zip = (header[0] == 'P' && header[1] == 'K' && header[2] == '\x03' && header[3] == '\x04');
        
        if (is_zip) {
            // This is a ZIP archive - use torch::serialize::InputArchive
            std::cerr << "Detected ZIP archive format, using torch::serialize::InputArchive" << std::endl;
            
            // Try to load using InputArchive which handles ZIP format
            torch::serialize::InputArchive archive;
            archive.load_from(weight_path);
            
            // Read all tensors from the archive
            std::vector<torch::Tensor> tensors;
            torch::load(tensors, weight_path);
            
            // The ZIP format stores tensors with keys, need different approach
            // For Hugging Face format, use torch::jit::load which handles this
            // But since jit::load may not be available, we need another approach
            
            // Try using the zipfile Python module to extract, then load the pickle
            // Since we can't do that easily in C++, let's try loading as a regular torch file
            std::cerr << "Warning: ZIP format requires extraction. Trying direct load..." << std::endl;
            
            // Fallback: try loading as a pickled file directly
            std::ifstream file(weight_path, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open file: " << weight_path << std::endl;
                return {};
            }
            
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);
            
            std::vector<char> data(size);
            if (!file.read(data.data(), size)) {
                std::cerr << "Error: Could not read file: " << weight_path << std::endl;
                return {};
            }
            
            // Try loading using pickle_load
            torch::IValue ivalue = torch::pickle_load(data);
            
            if (ivalue.isGenericDict()) {
                auto dict = ivalue.toGenericDict();
                for (auto& item : dict) {
                    std::string key = item.key().toStringRef();
                    if (item.value().isTensor()) {
                        state_dict[key] = item.value().toTensor();
                    }
                }
            } else if (ivalue.isTensor()) {
                state_dict["model"] = ivalue.toTensor();
            } else {
                std::cerr << "Warning: Unexpected checkpoint format" << std::endl;
            }
        } else {
            // Standard pickle format - use pickle_load
            std::ifstream file(weight_path, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open file: " << weight_path << std::endl;
                return {};
            }
            
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);
            
            std::vector<char> data(size);
            if (!file.read(data.data(), size)) {
                std::cerr << "Error: Could not read file: " << weight_path << std::endl;
                return {};
            }
            
            // Load using torch::pickle_load
            torch::IValue ivalue = torch::pickle_load(data);
            
            if (ivalue.isGenericDict()) {
                auto dict = ivalue.toGenericDict();
                for (auto& item : dict) {
                    std::string key = item.key().toStringRef();
                    if (item.value().isTensor()) {
                        state_dict[key] = item.value().toTensor();
                    }
                }
            } else if (ivalue.isTensor()) {
                state_dict["model"] = ivalue.toTensor();
            } else {
                std::cerr << "Warning: Unexpected checkpoint format in " << weight_path << std::endl;
            }
        }
        
    } catch (const c10::Error& e) {
        std::cerr << "Error loading weights from " << weight_path << ": " << e.what() << std::endl;
        return {};
    } catch (const std::exception& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        return {};
    }
    
    return state_dict;
}

std::unordered_map<std::string, torch::Tensor> WeightLoader::convert_state_dict(
    const std::unordered_map<std::string, torch::Tensor>& state_dict) {
    
    std::unordered_map<std::string, torch::Tensor> converted;
    
    for (const auto& [key, value] : state_dict) {
        std::string new_key = key;
        
        // Convert Python naming to C++ naming
        // Python uses '.' for module hierarchy, C++ uses '_' for nested modules
        
        // Handle block indexing: frame_block_0 -> frame_blocks_[0]
        size_t pos;
        while ((pos = new_key.find("frame_block_")) != std::string::npos) {
            size_t end_pos = pos + 12;
            size_t num_end = new_key.find('.', end_pos);
            std::string num_str = new_key.substr(end_pos, num_end - end_pos);
            new_key.replace(pos, 12 + num_str.length(), 
                "frame_blocks_" + num_str + ".");
        }
        
        while ((pos = new_key.find("global_block_")) != std::string::npos) {
            size_t end_pos = pos + 13;
            size_t num_end = new_key.find('.', end_pos);
            std::string num_str = new_key.substr(end_pos, num_end - end_pos);
            new_key.replace(pos, 13 + num_str.length(), 
                "global_blocks_" + num_str + ".");
        }
        
        converted[new_key] = value;
    }
    
    return converted;
}

bool WeightLoader::load_model_weights(
    torch::nn::Module& model, 
    const std::string& weight_path) {
    
    std::cout << "Loading weights from: " << weight_path << std::endl;
    
    auto state_dict = load_weights(weight_path);
    if (state_dict.empty()) {
        std::cerr << "Failed to load weights" << std::endl;
        return false;
    }
    
    std::cout << "Loaded " << state_dict.size() << " tensors from checkpoint" << std::endl;
    
    // Convert state dict keys
    auto converted_dict = convert_state_dict(state_dict);
    
    // Try to match and load weights manually
    std::cout << "Matching weights with model parameters..." << std::endl;
    
    auto model_params = model.named_parameters();
    int matched = 0;
    
    for (auto& param : model_params) {
        const std::string& key = param.key();
        torch::Tensor& tensor = param.value();
        
        if (converted_dict.find(key) != converted_dict.end()) {
            auto& weight = converted_dict[key];
            if (weight.sizes().vec() == tensor.sizes().vec()) {
                tensor.copy_(weight);
                matched++;
            } else {
                std::cerr << "Size mismatch for " << key 
                          << ": model=" << tensor.sizes() 
                          << " checkpoint=" << weight.sizes() << std::endl;
            }
        }
    }
    
    std::cout << "Successfully matched " << matched << " out of " 
              << model_params.size() << " model parameters" << std::endl;
    
    return matched > 0;
}

void WeightLoader::print_weight_info(
    const std::unordered_map<std::string, torch::Tensor>& state_dict) {
    
    std::cout << "\n=== Weight Information ===" << std::endl;
    std::cout << "Total tensors: " << state_dict.size() << std::endl;
    std::cout << "\nWeight keys:\n" << std::endl;
    
    int count = 0;
    for (const auto& [key, tensor] : state_dict) {
        std::cout << "  " << key << " : " << tensor.sizes() << " (" << tensor.dtype() << ")" << std::endl;
        count++;
        if (count >= 50) {
            std::cout << "  ... and " << (state_dict.size() - count) << " more" << std::endl;
            break;
        }
    }
    std::cout << "==========================\n" << std::endl;
}

void WeightLoader::print_expected_keys() {
    std::cout << R"(
=== Expected VGGT Weight Keys ===

Aggregator:
  aggregator.patch_embed.cls_token
  aggregator.patch_embed.pos_embed
  aggregator.patch_embed.register_tokens
  aggregator.patch_embed.mask_token
  aggregator.patch_embed.patch_embed.proj.weight
  aggregator.patch_embed.patch_embed.proj.bias
  aggregator.patch_embed.blocks_*.attn.qkv.weight
  aggregator.patch_embed.blocks_*.attn.qkv.bias
  aggregator.patch_embed.blocks_*.attn.proj.weight
  aggregator.patch_embed.blocks_*.attn.proj.bias
  aggregator.patch_embed.blocks_*.mlp.fc1.weight
  aggregator.patch_embed.blocks_*.mlp.fc1.bias
  aggregator.patch_embed.blocks_*.mlp.fc2.weight
  aggregator.patch_embed.blocks_*.mlp.fc2.bias
  aggregator.patch_embed.norm.weight
  aggregator.patch_embed.norm.bias
  aggregator.frame_block_*.attn.qkv.weight
  aggregator.frame_block_*.mlp.*
  aggregator.global_block_*.attn.qkv.weight
  aggregator.global_block_*.mlp.*
  aggregator.camera_token
  aggregator.register_token

Camera Head:
  camera_head.token_norm.weight
  camera_head.token_norm.bias
  camera_head.trunk_norm.weight
  camera_head.trunk_norm.bias
  camera_head.empty_pose_tokens
  camera_head.embed_pose.weight
  camera_head.embed_pose.bias
  camera_head.poseLN_modulation.0.weight
  camera_head.poseLN_modulation.0.bias
  camera_head.poseLN_modulation.2.weight
  camera_head.poseLN_modulation.2.bias
  camera_head.adaln_norm.weight
  camera_head.adaln_norm.bias
  camera_head.pose_branch.* 

DPT Heads (point_head, depth_head):
  *_head.norm.weight
  *_head.norm.bias
  *_head.projects_*.weight
  *_head.projects_*.bias
  *_head.resize_layers_*.weight
  *_head.resize_layers_*.bias
  *_head.layer*_rn.weight
  *_head.refinenet*.out_conv.weight
  *_head.output_conv*.weight
  *_head.output_conv*.bias

Track Head:
  track_head.feature_extractor.*
  track_head.tracker.*
)";
}

} // namespace utils
} // namespace vggt
