#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <sstream>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "vggt/models/vggt.h"
#include "vggt/utils/load_fn.h"
#include "vggt/utils/weight_loader.h"
#include "vggt/utils/geometry.h"
#include "vggt/utils/rotation.h"

namespace fs = std::filesystem;

// Global model instance
std::unique_ptr<vggt::models::VGGTImpl> g_model = nullptr;

/**
 * Initialize the VGGT model
 * 
 * @param weight_path Path to pretrained weights (optional)
 * @param device_str Device to run on (cuda or cpu)
 * @return true if initialization successful
 */
bool init_model(const std::string& weight_path = "", const std::string& device_str = "cuda") {
    std::cout << "Initializing VGGT model..." << std::endl;
    
    // Determine device
    torch::Device device = torch::kCPU;
    if (device_str == "cuda" && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA device" << std::endl;
    } else {
        std::cout << "Using CPU device" << std::endl;
    }
    
    // Create model
    g_model = std::make_unique<vggt::models::VGGTImpl>(
        518,    // img_size
        14,     // patch_size
        1024    // embed_dim
    );
    
    // Move model to device
    g_model->to(device);
    g_model->eval();
    
    // Load pretrained weights if provided
    if (!weight_path.empty() && fs::exists(weight_path)) {
        std::cout << "Loading pretrained weights from: " << weight_path << std::endl;
        if (vggt::utils::WeightLoader::load_model_weights(*g_model, weight_path)) {
            std::cout << "Successfully loaded pretrained weights" << std::endl;
        } else {
            std::cerr << "Warning: Failed to load pretrained weights, using random initialization" << std::endl;
        }
    } else if (!weight_path.empty()) {
        std::cerr << "Warning: Weight file not found: " << weight_path << std::endl;
    } else {
        std::cout << "No pretrained weights provided, using random initialization" << std::endl;
    }
    
    return true;
}

/**
 * Run inference on a list of images
 * 
 * @param image_paths List of image file paths
 * @param query_points Optional query points for tracking [N, 2]
 * @param device_str Device to run on
 * @return Dictionary of predictions
 */
std::unordered_map<std::string, torch::Tensor> run_inference(
    const std::vector<std::string>& image_paths,
    const torch::Tensor& query_points = {},
    const std::string& device_str = "cuda") {
    
    if (!g_model) {
        throw std::runtime_error("Model not initialized. Call init_model() first.");
    }
    
    torch::Device device = (device_str == "cuda" && torch::cuda::is_available()) 
        ? torch::kCUDA : torch::kCPU;
    
    std::cout << "\n=== Running Inference ===" << std::endl;
    std::cout << "Processing " << image_paths.size() << " images..." << std::endl;
    
    // Load and preprocess images
    std::cout << "Loading and preprocessing images..." << std::endl;
    
    // Use NoGradGuard during preprocessing to avoid gradient issues
    torch::Tensor images_tensor;
    torch::Tensor coords_tensor;
    {
        torch::NoGradGuard no_grad_pre;
        auto result = vggt::utils::load_and_preprocess_images_square(image_paths, 518);
        images_tensor = std::get<0>(result);
        coords_tensor = std::get<1>(result);
        
        // Explicitly set requires_grad to false
        images_tensor.set_requires_grad(false);
        coords_tensor.set_requires_grad(false);
        
        // Move to device while still in no_grad context
        images_tensor = images_tensor.to(device);
    }
    
    std::cout << "Input shape: " << images_tensor.sizes() << std::endl;
    
    // Prepare query points if provided
    torch::Tensor query_pts = {};
    if (query_points.defined() && query_points.numel() > 0) {
        query_pts = query_points.to(device);
        std::cout << "Query points shape: " << query_pts.sizes() << std::endl;
    }
    
    // Run inference with timing
    std::cout << "Running model forward pass..." << std::endl;
    
    torch::NoGradGuard no_grad;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto predictions = g_model->forward(images_tensor, query_pts);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Inference completed in " << duration.count() << " ms" << std::endl;
    
    // Print prediction shapes
    std::cout << "\n=== Prediction Shapes ===" << std::endl;
    for (const auto& [key, tensor] : predictions) {
        if (tensor.defined() && tensor.numel() > 0) {
            std::cout << "  " << key << ": " << tensor.sizes() << std::endl;
        }
    }
    
    return predictions;
}

/**
 * Visualize and save predictions
 */
void visualize_predictions(
    const std::unordered_map<std::string, torch::Tensor>& predictions,
    const std::vector<std::string>& image_paths,
    const std::string& output_dir = "./output") {
    
    // Create output directory
    fs::create_directories(output_dir);
    
    std::cout << "\n=== Saving Results ===" << std::endl;
    
    // Save depth map
    if (predictions.at("depth").defined()) {
        auto depth = predictions.at("depth").cpu();
        // depth shape: [B, S, H, W, 1]
        for (int b = 0; b < depth.size(0); b++) {
            for (int s = 0; s < depth.size(1); s++) {
                auto depth_frame = depth[b][s];
                if (depth_frame.dim() > 2) {
                    depth_frame = depth_frame.squeeze();
                }
                
                // Convert to 8-bit for saving
                auto min_val = depth_frame.min();
                auto max_val = depth_frame.max();
                auto depth_normalized = (depth_frame - min_val) / (max_val - min_val + 1e-8);
                
                cv::Mat depth_cv(depth_normalized.size(0), depth_normalized.size(1), CV_8UC1);
                auto depth_data = depth_normalized.to(torch::kU8).cpu().data_ptr<uint8_t>();
                std::memcpy(depth_cv.data, depth_data, depth_normalized.numel());
                
                std::string save_path = output_dir + "/depth_b" + std::to_string(b) + 
                                       "_s" + std::to_string(s) + ".png";
                cv::imwrite(save_path, depth_cv);
                std::cout << "Saved: " << save_path << std::endl;
            }
        }
    }
    
    // Save camera poses (pose encoding)
    if (predictions.at("pose_enc").defined()) {
        auto pose_enc = predictions.at("pose_enc").cpu();
        // pose_enc shape: [B, S, 9]
        
        std::string pose_file_path = output_dir + "/poses.txt";
        std::ofstream pose_file(pose_file_path);
        pose_file << "# Camera poses [B, S, 9] - format: translation(3), quaternion(4), FoV(2)\n";
        
        for (int b = 0; b < pose_enc.size(0); b++) {
            for (int s = 0; s < pose_enc.size(1); s++) {
                auto pose = pose_enc[b][s];
                pose_file << "frame_" << b << "_" << s << ": ";
                for (int i = 0; i < pose.size(0); i++) {
                    pose_file << pose[i].item<float>();
                    if (i < pose.size(0) - 1) pose_file << " ";
                }
                pose_file << "\n";
            }
        }
        pose_file.close();
        std::cout << "Saved: " << pose_file_path << std::endl;
    }
    
    std::cout << "Results saved to: " << output_dir << std::endl;
}

/**
 * Print usage information
 */
void print_usage(const char* program_name) {
    std::cout << R"(
VGGT-CPP Demo
=============

Usage: 
  )" << program_name << R"( [options]

Options:
  -h, --help              Show this help message
  -m, --model <path>      Path to pretrained model weights (.pt file)
  -i, --images <paths>    Comma-separated list of image paths (at least 2 images)
  -o, --output <dir>      Output directory (default: ./output)
  -d, --device <device>   Device to use: cuda or cpu (default: cuda if available)
  -q, --query <points>    Query points for tracking, format: x1,y1,x2,y2,...
  -l, --list-weights      Print expected weight key names
  -p, --print-weights     Print weight info from checkpoint file

Examples:
  )" << program_name << R"( -m vggt_weights.pt -i img1.jpg,img2.jpg,img3.jpg
  )" << program_name << R"( -m vggt_weights.pt -i img1.jpg,img2.jpg -q 100,200,300,400
  )" << program_name << R"( -i img1.jpg,img2.jpg -d cpu  # Run on CPU without weights

Notes:
  - At least 2 images are required for camera pose estimation
  - If no model weights are provided, the model runs with random initialization
  - Query points are in pixel coordinates (x, y)
)" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "======================================" << std::endl;
    std::cout << "  VGGT-CPP Inference Demo" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // Parse command line arguments
    std::string weight_path;
    std::string image_paths_str;
    std::string output_dir = "./output";
    std::string device_str = "cuda";
    std::string query_points_str;
    bool list_weights = false;
    std::string print_weights_path;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            weight_path = argv[++i];
        } else if ((arg == "-i" || arg == "--images") && i + 1 < argc) {
            image_paths_str = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_dir = argv[++i];
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            device_str = argv[++i];
        } else if ((arg == "-q" || arg == "--query") && i + 1 < argc) {
            query_points_str = argv[++i];
        } else if (arg == "-l" || arg == "--list-weights") {
            list_weights = true;
        } else if ((arg == "-p" || arg == "--print-weights") && i + 1 < argc) {
            print_weights_path = argv[++i];
        }
    }
    
    if (list_weights) {
        vggt::utils::WeightLoader::print_expected_keys();
        return 0;
    }
    
    if (!print_weights_path.empty()) {
        auto weights = vggt::utils::WeightLoader::load_weights(print_weights_path);
        vggt::utils::WeightLoader::print_weight_info(weights);
        return 0;
    }
    
    if (image_paths_str.empty()) {
        std::cerr << "Error: No images provided. Use -i or --images option." << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Parse image paths
    std::vector<std::string> image_paths;
    std::stringstream ss(image_paths_str);
    std::string path;
    while (std::getline(ss, path, ',')) {
        image_paths.push_back(path);
    }
    
    if (image_paths.size() < 2) {
        std::cerr << "Error: At least 2 images are required for camera pose estimation." << std::endl;
        return 1;
    }
    
    // Check image files exist
    for (const auto& img_path : image_paths) {
        if (!fs::exists(img_path)) {
            std::cerr << "Error: Image file not found: " << img_path << std::endl;
            return 1;
        }
    }
    
    // Initialize model
    if (!init_model(weight_path, device_str)) {
        std::cerr << "Error: Failed to initialize model" << std::endl;
        return 1;
    }
    
    // Parse query points if provided
    torch::Tensor query_points;
    if (!query_points_str.empty()) {
        std::vector<float> pts;
        std::stringstream ss_pts(query_points_str);
        std::string pt;
        while (std::getline(ss_pts, pt, ',')) {
            pts.push_back(std::stof(pt));
        }
        
        if (pts.size() % 2 != 0) {
            std::cerr << "Error: Query points must be in pairs (x, y)" << std::endl;
            return 1;
        }
        
        query_points = torch::tensor(pts).view({1, -1, 2});  // [1, N, 2] - add batch dimension
        std::cout << "Using " << query_points.size(1) << " query points" << std::endl;
    }
    
    // Run inference
    try {
        auto predictions = run_inference(image_paths, query_points, device_str);
        
        // Visualize and save results
        visualize_predictions(predictions, image_paths, output_dir);
        
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nDemo completed successfully!" << std::endl;
    return 0;
}
