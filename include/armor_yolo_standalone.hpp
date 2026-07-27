#ifndef ARMOR_YOLO_STANDALONE_HPP
#define ARMOR_YOLO_STANDALONE_HPP

#include "armor_types.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief 装甲板 YOLO 检测器 (独立版本)
 *
 * 从 rosbag 读取的图像中检测装甲板角点
 * 使用 OpenCV DNN 加载 ONNX 模型进行推理
 */
class ArmorYoloStandalone {
public:
    /**
     * @brief 构造函数
     */
    ArmorYoloStandalone();

    /**
     * @brief 加载 ONNX 模型
     * @param model_path ONNX 模型文件路径
     * @return 是否成功加载
     */
    bool loadModel(const std::string& model_path);

    /**
     * @brief 执行装甲板检测
     * @param img 输入图像 (BGR 格式)
     * @return 检测到的装甲板列表
     */
    std::vector<DetectedArmor> detect(const cv::Mat& img);

    /**
     * @brief 设置检测颜色
     * @param color 0=蓝色, 1=红色
     */
    void setDetectColor(int color) { detect_color_ = color; }

    /**
     * @brief 设置置信度阈值
     */
    void setConfThreshold(float thr) { conf_threshold_ = thr; }

    /**
     * @brief 设置 NMS 阈值
     */
    void setNmsThreshold(float thr) { nms_threshold_ = thr; }

private:
    cv::dnn::Net net_;                      // DNN 网络
    float conf_threshold_ = 0.65f;          // 置信度阈值
    float nms_threshold_ = 0.45f;           // NMS 阈值
    int detect_color_ = 0;                  // 检测颜色
    static constexpr int INPUT_WIDTH = 640; // 网络输入宽度
    static constexpr int INPUT_HEIGHT = 640;// 网络输入高度
};

#endif // ARMOR_YOLO_STANDALONE_HPP
