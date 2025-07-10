# VGGT-CPP: Visual Geometry and Graph Tracking Library

VGGT-CPP是一个C++库，提供视觉几何和图形跟踪功能，是[VGGT](https://github.com/laplaceliu/vggt)的C++实现版本。该库提供了一系列用于计算机视觉和3D几何处理的工具函数。

## 功能特点

- 几何变换和坐标处理
- 相机姿态编码和解码
- 旋转矩阵和四元数转换
- 图像加载和预处理
- 点云投影和变换
- 视觉跟踪结果可视化

## 依赖项

- C++14或更高版本
- CMake 3.14或更高版本
- LibTorch (PyTorch C++ API)
- OpenCV 4.x

## 安装

### 从源代码构建

1. 克隆仓库：

```bash
git clone https://github.com/laplaceliu/vggt-cpp.git
cd vggt-cpp
```

2. 创建构建目录：

```bash
mkdir build && cd build
```

3. 配置和构建：

```bash
cmake ..
make -j4
```

4. 安装（可选）：

```bash
sudo make install
```

## 使用示例

### 图像处理

```cpp
#include <vggt/vggt.h>
#include <iostream>

int main() {
    // 加载图像
    torch::Tensor image = vggt::utils::load_rgb("path/to/image.jpg");

    // 调整图像大小
    torch::Tensor resized = vggt::utils::resize_image(image, 480, 640);

    // 归一化图像
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    torch::Tensor normalized = vggt::utils::normalize_image(resized, mean, std);

    std::cout << "Processed image shape: " << normalized.sizes() << std::endl;

    return 0;
}
```

### 相机姿态编码

```cpp
#include <vggt/vggt.h>
#include <iostream>

int main() {
    // 创建相机外参矩阵（相机到世界坐标变换）
    torch::Tensor extrinsics = torch::eye(4);
    extrinsics[0][3] = 1.0; // X轴平移

    // 创建相机内参矩阵
    torch::Tensor intrinsics = torch::zeros({3, 3});
    intrinsics[0][0] = 500.0; // fx
    intrinsics[1][1] = 500.0; // fy
    intrinsics[0][2] = 320.0; // cx
    intrinsics[1][2] = 240.0; // cy
    intrinsics[2][2] = 1.0;

    // 编码相机姿态
    torch::Tensor pose_encoding = vggt::utils::extri_intri_to_pose_encoding(
        extrinsics.unsqueeze(0), intrinsics.unsqueeze(0));

    std::cout << "Pose encoding shape: " << pose_encoding.sizes() << std::endl;

    // 解码相机姿态
    auto [decoded_extrinsics, decoded_intrinsics] =
        vggt::utils::pose_encoding_to_extri_intri(pose_encoding);

    return 0;
}
```

### 跟踪可视化

```cpp
#include <vggt/vggt.h>
#include <iostream>

int main() {
    // 加载图像序列
    std::vector<torch::Tensor> images;
    for (int i = 0; i < 10; ++i) {
        std::string path = "path/to/sequence/frame_" + std::to_string(i) + ".jpg";
        images.push_back(vggt::utils::load_rgb(path));
    }

    // 创建图像序列张量
    torch::Tensor image_sequence = torch::stack(images);

    // 假设我们有检测框、分数和标签
    torch::Tensor boxes = torch::rand({10, 5, 4}) * 640; // [seq_len, num_objects, 4]
    torch::Tensor scores = torch::rand({10, 5});         // [seq_len, num_objects]
    torch::Tensor labels = torch::zeros({10, 5}, torch::kLong); // [seq_len, num_objects]

    // 类别名称
    std::vector<std::string> class_names = {"person", "car", "bicycle"};

    // 可视化跟踪结果
    std::vector<cv::Mat> result_frames = vggt::utils::draw_tracking_results(
        image_sequence, boxes, scores, labels, class_names, 0.5);

    // 保存跟踪视频
    vggt::utils::save_tracking_video(result_frames, "tracking_result.avi", 30);

    return 0;
}
```

## 示例程序

库中包含了几个示例程序，展示了VGGT-CPP的基本用法：

- `image_processing_example`: 演示图像加载和处理功能
- `pose_encoding_example`: 演示相机姿态编码和解码功能
- `tracking_visualization_example`: 演示跟踪结果可视化功能

要运行这些示例，请在构建目录中执行：

```bash
./examples/image_processing_example path/to/image.jpg
./examples/pose_encoding_example
./examples/tracking_visualization_example path/to/image.jpg
```

## 许可证

MIT License

## 贡献

欢迎提交问题和拉取请求！
