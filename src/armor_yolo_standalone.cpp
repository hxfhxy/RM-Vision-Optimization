#include "armor_yolo_standalone.hpp"
#include <cmath>

/**
 * @brief 构造函数
 */
ArmorYoloStandalone::ArmorYoloStandalone() {}

/**
 * @brief 加载 YOLO ONNX 模型
 */
bool ArmorYoloStandalone::loadModel(const std::string& model_path) {
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
        std::cout << "[YOLO] Model loaded: " << model_path << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[YOLO] Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 执行装甲板检测
 *
 * 流程:
 * 1. 图像预处理 → Blob
 * 2. 网络前向推理
 * 3. 解析输出 → 角点 + 置信度
 * 4. NMS 过滤
 * 5. 返回检测结果
 */
std::vector<DetectedArmor> ArmorYoloStandalone::detect(const cv::Mat& img) {
    std::vector<DetectedArmor> final_armors;

    if (img.empty() || net_.empty()) {
        return final_armors;
    }

    // 1. 预处理: 图像 → Blob
    // 输入图像是 BGR (rosbag_reader 转换后)
    // 模型期望 BGR 输入, 使用 swapRB=false (不做通道交换)
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0,
                                          cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
                                          cv::Scalar(0, 0, 0), false, false);
    net_.setInput(blob);

    // 2. 前向推理
    std::vector<cv::Mat> net_outputs;
    net_.forward(net_outputs, net_.getUnconnectedOutLayersNames());

    if (net_outputs.empty()) {
        return final_armors;
    }

    // 3. 解析输出
    cv::Mat output = net_outputs[0];
    cv::Mat output_buffer(output.size[1], output.size[2], CV_32F, output.ptr<float>());

    // 【修改点 1】: 将 cv::Rect 替换为 cv::RotatedRect
    std::vector<cv::RotatedRect> rotated_boxes;
    std::vector<float> confidences;
    std::vector<DetectedArmor> temp_armors;

    float x_factor = img.cols / static_cast<float>(INPUT_WIDTH);
    float y_factor = img.rows / static_cast<float>(INPUT_HEIGHT);

    for (int i = 0; i < output_buffer.rows; ++i) {
        // 置信度 (sigmoid)
        float confidence = output_buffer.at<float>(i, 8);
        confidence = 1.0f / (1.0f + std::exp(-confidence));
        if (confidence < conf_threshold_) continue;

        // 分类得分
        cv::Mat color_scores = output_buffer.row(i).colRange(9, 13);
        cv::Mat classes_scores = output_buffer.row(i).colRange(13, 22);

        cv::Point class_id, color_id;
        double score_num = 0.0, score_color = 0.0;
        cv::minMaxLoc(classes_scores, nullptr, &score_num, nullptr, &class_id);
        cv::minMaxLoc(color_scores, nullptr, &score_color, nullptr, &color_id);

        // 颜色过滤: 模型 color_id 0=红, 1=蓝
        int model_color = 1 - detect_color_;
        if (color_id.x >= 2) continue;
        if (color_id.x != model_color) continue;

        // 角点坐标
        cv::Point2f p1_lt(output_buffer.at<float>(i, 0) * x_factor,
                          output_buffer.at<float>(i, 1) * y_factor);
        cv::Point2f p2_lb(output_buffer.at<float>(i, 2) * x_factor,
                          output_buffer.at<float>(i, 3) * y_factor);
        cv::Point2f p3_rb(output_buffer.at<float>(i, 4) * x_factor,
                          output_buffer.at<float>(i, 5) * y_factor);
        cv::Point2f p4_rt(output_buffer.at<float>(i, 6) * x_factor,
                          output_buffer.at<float>(i, 7) * y_factor);

        // 构建检测结果
        DetectedArmor armor;
        armor.pnp_corners = {p1_lt, p2_lb, p3_rb, p4_rt};
        armor.number = class_id.x;
        armor.class_confidence = static_cast<float>(score_num);
        armor.confidence = confidence;
        
        // 【修改点 2】: 利用 4 个角点直接生成最小外接旋转矩形 (RotatedRect)
        armor.rect = cv::minAreaRect(armor.pnp_corners);

        rotated_boxes.push_back(armor.rect);
        confidences.push_back(confidence);
        temp_armors.push_back(armor);
    }

    // 4. 【修改点 3】: 换用 OpenCV 原生的 Rotated NMS 过滤算法
    std::vector<int> indices;
    if (!rotated_boxes.empty()) {
        cv::dnn::NMSBoxes(rotated_boxes, confidences, conf_threshold_, nms_threshold_, indices);
    }

    // 5. 整理结果
    cv::Point2f imgCenter(img.cols / 2.0f, img.rows / 2.0f);
    for (int valid_index : indices) {
        if (valid_index < 0 || valid_index >= static_cast<int>(temp_armors.size())) continue;
        DetectedArmor armor = temp_armors[valid_index];
        armor.dist_to_center = cv::norm(armor.rect.center - imgCenter);
        final_armors.push_back(std::move(armor));
    }

    return final_armors;
}
