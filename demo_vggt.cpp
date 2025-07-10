#include <iomanip>
#include <iostream>
#include <torch/torch.h>
#include "vggt/vggt.h"

int main() {
    try {
        // Initialize VGGT model
        auto model = vggt::VGGT();
        std::cout << "VGGT model initialized successfully" << std::endl;

        // Create dummy input: 5 frames of 512x512 RGB images
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor images = torch::rand({5, 3, 512, 512}, options);  // [S, C, H, W]

        // Add batch dimension and normalize to [0, 1] range
        images = images.unsqueeze(0) / 255.0f;  // [B, S, C, H, W]

        // Create dummy query points for tracking (10 points)
        torch::Tensor query_points = torch::rand({10, 2}, options) * 512.0f;

        // Forward pass
        auto results = model->forward(images, query_points);

        // Print some results
        std::cout << "\nResults:" << std::endl;
        std::cout << "Pose encoding shape: " << results["pose_enc"].sizes() << std::endl;
        std::cout << "Depth maps shape: " << results["depth"].sizes() << std::endl;
        std::cout << "World points shape: " << results["world_points"].sizes() << std::endl;
        std::cout << "Track points shape: " << results["track"].sizes() << std::endl;

        // Print first tracked point positions
        auto first_track = results["track"][0].slice(0, 0, 3);  // First 3 frames
        std::cout << "\nFirst tracked point positions (first 3 frames):\n"
                  << first_track << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
