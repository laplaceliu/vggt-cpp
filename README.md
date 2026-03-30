# VGGT-CPP

A C++ implementation of [VGGT (Visual Geometry Grounded Transformer)](https://github.com/facebookresearch/vggt) for camera pose estimation, depth prediction, and point tracking from image sequences.

## Features

- **Camera Pose Estimation**: Predicts camera poses (translation + rotation + FoV) from image sequences
- **Depth Prediction**: Generates dense depth maps using DPT (Dense Prediction Transformer) architecture
- **3D World Points**: Outputs 3D world coordinates for each pixel
- **Point Tracking**: Tracks query points across frames using correlation pyramid and transformer refinement

## Architecture

The model consists of:
- **Aggregator**: DINOv2-based vision transformer with alternating frame/global attention
- **CameraHead**: AdaLN-based camera pose prediction with iterative refinement
- **DPTHead**: Multi-scale feature fusion for depth and 3D point prediction
- **TrackHead**: Correlation-based point tracking with EfficientUpdateFormer

## Requirements

- CUDA 11.8+ / CUDA 12.x
- libtorch (PyTorch C++ frontend)
- OpenCV 4.x
- Eigen3
- CMake 3.15+

## Build Instructions

### 1. Install Dependencies

```bash
# Install all third-party dependencies (run once)
./build_thirdparty.sh

# Or install specific dependencies
./build_thirdparty.sh libtorch
./build_thirdparty.sh opencv
./build_thirdparty.sh eigen
```

### 2. Build the Project

```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build with specific CUDA architecture (optional)
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89

# Compile
make -j
```

## Usage

### Demo Application

The `demo_vggt` executable provides a complete inference pipeline:

```bash
# Basic usage (requires at least 2 images)
./demo_vggt -i img1.jpg,img2.jpg,img3.jpg

# With pretrained weights
./demo_vggt -m vggt_weights.pt -i img1.jpg,img2.jpg,img3.jpg

# With query points for tracking
./demo_vggt -m vggt_weights.pt -i img1.jpg,img2.jpg -q 100,200,300,400

# CPU mode (no CUDA required)
./demo_vggt -i img1.jpg,img2.jpg -d cpu

# Specify output directory
./demo_vggt -m vggt_weights.pt -i img1.jpg,img2.jpg -o ./output

# Print expected weight keys (for debugging)
./demo_vggt -l

# Print weight info from checkpoint file
./demo_vggt -p vggt_weights.pt
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `-m, --model <path>` | Path to pretrained model weights (.pt file) |
| `-i, --images <paths>` | Comma-separated list of image paths (at least 2) |
| `-o, --output <dir>` | Output directory (default: ./output) |
| `-d, --device` | Device: cuda or cpu (default: cuda) |
| `-q, --query <points>` | Query points for tracking (x1,y1,x2,y2,...) |
| `-l, --list-weights` | Print expected weight key names |
| `-p, --print-weights <path>` | Print weight info from checkpoint |

### Output

The demo produces:
- `depth_b{b}_s{s}.png` - Depth maps for each frame
- `poses.txt` - Camera poses [translation(3), quaternion(4), FoV(2)]

## Pretrained Weights

Pretrained VGGT weights can be downloaded from:
- [HuggingFace VGGT](https://huggingface.co/facebook/vggt)
- [PyTorch VGGT](https://github.com/facebookresearch/vggt)

Example weight loading (Python):
```python
from vggt.models.vggt import VGGT
model = VGGT.from_pretrained("facebook/vggt")
```

## C++ API

### Basic Inference

```cpp
#include "vggt/models/vggt.h"
#include "vggt/utils/load_fn.h"
#include "vggt/utils/weight_loader.h"

// Create model
auto model = std::make_unique<vggt::models::VGGTImpl>(
    518,    // img_size
    14,     // patch_size
    1024    // embed_dim
);
model->eval();

// Load weights (optional)
vggt::utils::WeightLoader::load_model_weights(*model, "weights.pt");

// Load and preprocess images
auto [images, coords] = vggt::utils::load_and_preprocess_images_square(
    {"img1.jpg", "img2.jpg"}, 518);

// Run inference
torch::NoGradGuard no_grad;
auto predictions = model->forward(images);

// Access results
auto& depth = predictions["depth"];        // [B, S, H, W, 1]
auto& pose_enc = predictions["pose_enc"];   // [B, S, 9]
auto& world_points = predictions["world_points"];  // [B, S, H, W, 3]
```

### Weight Loader API

```cpp
// Load weights from checkpoint file
auto state_dict = vggt::utils::WeightLoader::load_weights("weights.pt");

// Convert Python-style keys to C++ style
auto converted = vggt::utils::WeightLoader::convert_state_dict(state_dict);

// Load into model
vggt::utils::WeightLoader::load_model_weights(model, "weights.pt");

// Print weight information
vggt::utils::WeightLoader::print_weight_info(state_dict);
vggt::utils::WeightLoader::print_expected_keys();
```

## Project Structure

```
vggt-cpp/
├── vggt/
│   ├── models/           # Model implementations
│   │   ├── vggt.cpp/h    # Main VGGT model
│   │   └── aggregator.cpp/h  # Feature aggregator
│   ├── heads/            # Prediction heads
│   │   ├── camera_head.cpp/h
│   │   ├── dpt_head.cpp/h
│   │   └── track_head.cpp/h
│   ├── layers/           # Neural network layers
│   │   ├── attention.cpp/h
│   │   ├── block.cpp/h
│   │   ├── rope.cpp/h
│   │   └── vision_transformer.cpp/h
│   ├── utils/            # Utility functions
│   │   ├── load_fn.cpp/h    # Image loading
│   │   ├── weight_loader.cpp/h  # Weight loading
│   │   ├── geometry.cpp/h
│   │   └── rotation.cpp/h
│   └── dependency/       # External dependencies
│       └── track_modules/
├── demo_vggt.cpp         # Demo application
├── CMakeLists.txt
└── build_thirdparty.sh
```

## Gradient Checkpointing

During training, gradient checkpointing can reduce memory usage by recomputing activations instead of storing them. This is enabled by default in Python but the C++ implementation uses standard forward passes (libtorch's checkpoint API is not available in all versions).

To enable training mode:
```cpp
model->train(true);  // Enable training mode
```

Note: The current implementation uses standard forward passes without memory optimization.

## Memory Requirements

- **Inference**: ~8GB GPU memory for a single image at 518x518
- **Training**: Requires gradient checkpointing for large batch sizes

## Citation

If you use this code, please cite the original VGGT paper:

```bibtex
@article{vggt2024,
  title={Visual Geometry Grounded Transformer},
  author={},
  journal={arXiv},
  year={2024}
}
```

## License

This project is licensed under the Apache 2.0 License. See LICENSE file for details.

## Acknowledgments

- VGGT team at Meta AI for the original Python implementation
- DINOv2 team for the vision transformer backbone
- VGGSfM team for the tracking modules
