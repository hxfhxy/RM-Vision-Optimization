#include "visualizer.hpp"
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cmath>

// 颜色定义 (BGR 格式)
const cv::Scalar Visualizer::COLOR_PNP(0, 255, 0);         // 绿色
const cv::Scalar Visualizer::COLOR_BA(0, 0, 255);          // 红色
const cv::Scalar Visualizer::COLOR_TEXT(255, 255, 255);     // 白色

// BA对比视频写入器
static cv::VideoWriter ba_video_writer_;
static bool ba_video_initialized_ = false;

/**
 * @brief 构造函数
 */
Visualizer::Visualizer(const std::string& output_dir, const std::string& video_name)
    : output_dir_(output_dir), video_name_(video_name) {
    ensureOutputDir();
}

/**
 * @brief 确保输出目录存在
 */
void Visualizer::ensureOutputDir() const {
    std::filesystem::create_directories(output_dir_);
}

/**
 * @brief 绘制单帧的重投影对比图 (三方法: PnP, BA, Temporal)
 */
void Visualizer::drawReprojectionComparison(
    cv::Mat& image,
    const std::vector<cv::Point2f>& pnp_corners,
    const std::vector<cv::Point2f>& ba_corners,
    const std::vector<cv::Point2f>& /*g2o_corners*/,  // 保留接口兼容, 不再使用
    const std::vector<cv::Point2f>& temporal_corners,
    uint64_t frame_index) {

    // 绘制 PnP 角点 (绿色圆圈)
    for (const auto& pt : pnp_corners) {
        cv::circle(image, pt, 6, COLOR_PNP, 2);
    }
    for (int i = 0; i < 4; ++i) {
        cv::line(image, pnp_corners[i], pnp_corners[(i+1)%4], COLOR_PNP, 2);
    }

    // 绘制 BA 角点 (红色叉号)
    for (const auto& pt : ba_corners) {
        cv::drawMarker(image, pt, COLOR_BA, cv::MARKER_CROSS, 16, 2);
    }
    for (int i = 0; i < 4; ++i) {
        cv::line(image, ba_corners[i], ba_corners[(i+1)%4], COLOR_BA, 1, cv::LINE_AA);
    }

    // 绘制图例
    int legend_x = 10, legend_y = 30;
    cv::putText(image, "Frame: " + std::to_string(frame_index),
               cv::Point(legend_x, legend_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2);
    legend_y += 30;

    auto drawLegendItem = [&](const std::string& text, const cv::Scalar& color) {
        cv::circle(image, cv::Point(legend_x + 10, legend_y - 5), 5, color, -1);
        cv::putText(image, text, cv::Point(legend_x + 25, legend_y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        legend_y += 25;
    };

    drawLegendItem("PnP (baseline)", COLOR_PNP);
    drawLegendItem("BA (custom LM)", COLOR_BA);
}

/**
 * @brief 保存当前帧到视频
 */
void Visualizer::writeVideoFrame(const cv::Mat& image) {
    if (!video_initialized_) {
        std::string video_path = output_dir_ + "/" + video_name_ + "_output.mp4";
        video_writer_.open(video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                          20, image.size());
        video_initialized_ = true;
    }
    if (video_writer_.isOpened()) {
        video_writer_.write(image);
    }
}

/**
 * @brief 保存BA对比帧到单独视频
 */
void Visualizer::writeVideoFrameBA(const cv::Mat& image) {
    if (!ba_video_initialized_) {
        std::string video_path = output_dir_ + "/" + video_name_ + "_ba_effect.mp4";
        ba_video_writer_.open(video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                             20, image.size());
        ba_video_initialized_ = true;
    }
    if (ba_video_writer_.isOpened()) {
        ba_video_writer_.write(image);
    }
}

/**
 * @brief 释放视频写入器
 */
void Visualizer::releaseVideo() {
    if (video_writer_.isOpened()) {
        video_writer_.release();
    }
    if (ba_video_writer_.isOpened()) {
        ba_video_writer_.release();
    }
    ba_video_initialized_ = false;
}

/**
 * @brief 获取方法对应的颜色
 */
cv::Scalar Visualizer::getMethodColor(const std::string& method) const {
    // OpenCV 的颜色是 BGR 格式
    if (method == "YOLO_BA") return cv::Scalar(0, 0, 255);    // 红色
    if (method == "PCA_BA") return cv::Scalar(255, 255, 0);   // 青色 (Cyan)
    
    // 兜底颜色
    return cv::Scalar(128, 128, 128);  // 灰色 
}

/**
 * @brief 获取方法对应的线宽
 */
int Visualizer::getMethodLineWidth(const std::string& method) const {
    return 1;  // 所有曲线统一为 1px
}

/**
 * @brief 通用曲线图绘制
 */
void Visualizer::plotCurve(const std::string& title,
                           const std::string& y_label,
                           const std::map<std::string, std::vector<double>>& data_series,
                           const std::string& filename,
                           int image_width, int image_height) {
    // 创建图像 (白色背景)
    cv::Mat plot(image_height, image_width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 图表区域
    int margin_left = 80, margin_right = 180, margin_top = 60, margin_bottom = 60;
    int plot_width = image_width - margin_left - margin_right;
    int plot_height = image_height - margin_top - margin_bottom;

    // 绘制边框
    cv::rectangle(plot, cv::Point(margin_left, margin_top),
                  cv::Point(margin_left + plot_width, margin_top + plot_height),
                  cv::Scalar(0, 0, 0), 1);

    // 找到数据范围
    size_t max_len = 0;
    double y_min = 1e10, y_max = -1e10;
    for (const auto& [method, data] : data_series) {
        max_len = std::max(max_len, data.size());
        for (double v : data) {
            y_min = std::min(y_min, v);
            y_max = std::max(y_max, v);
        }
    }

    if (max_len == 0 || y_max <= y_min) return;

    // 添加 10% 的 y 轴边距
    double y_range = y_max - y_min;
    y_min -= y_range * 0.1;
    y_max += y_range * 0.1;
    y_range = y_max - y_min;

    // 绘制标题
    cv::putText(plot, title, cv::Point(image_width / 2 - 100, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // 绘制 Y 轴标签
    cv::putText(plot, y_label, cv::Point(10, margin_top + plot_height / 2),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    // 绘制 Y 轴刻度 (5 格)
    for (int i = 0; i <= 5; ++i) {
        int y = margin_top + plot_height - i * plot_height / 5;
        double value = y_min + i * y_range / 5.0;
        cv::line(plot, cv::Point(margin_left - 5, y), cv::Point(margin_left, y),
                cv::Scalar(0, 0, 0), 1);
        cv::putText(plot, cv::format("%.1f", value),
                   cv::Point(5, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 0, 0), 1);
    }

    // 绘制 X 轴标签
    cv::putText(plot, "Frame", cv::Point(margin_left + plot_width / 2, image_height - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    // 绘制网格线 (淡灰色)
    for (int i = 1; i <= 5; ++i) {
        int y = margin_top + plot_height - i * plot_height / 5;
        cv::line(plot, cv::Point(margin_left, y), cv::Point(margin_left + plot_width, y),
                cv::Scalar(230, 230, 230), 1);
    }

    // 绘制各方法的曲线
    int legend_y = margin_top + 20;
    for (auto it = data_series.rbegin(); it != data_series.rend(); ++it) {
        const std::string& method = it->first;
        const std::vector<double>& data = it->second;

        if (data.empty()) continue;

        cv::Scalar color = getMethodColor(method);
        int line_width = getMethodLineWidth(method);

        // 绘制折线
        for (size_t i = 1; i < data.size(); ++i) {
            int x1 = margin_left + static_cast<int>((i - 1) * plot_width / (max_len - 1));
            int x2 = margin_left + static_cast<int>(i * plot_width / (max_len - 1));
            int y1 = margin_top + plot_height - static_cast<int>((data[i-1] - y_min) / y_range * plot_height);
            int y2 = margin_top + plot_height - static_cast<int>((data[i] - y_min) / y_range * plot_height);
            cv::line(plot, cv::Point(x1, y1), cv::Point(x2, y2), color, line_width, cv::LINE_AA);
        }

        // 绘制图例
        cv::line(plot, cv::Point(margin_left + plot_width + 10, legend_y),
                cv::Point(margin_left + plot_width + 30, legend_y), color, 2);
        cv::putText(plot, method, cv::Point(margin_left + plot_width + 35, legend_y + 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        legend_y += 25;
    }

    // 保存图像
    std::string filepath = output_dir_ + "/" + filename;
    cv::imwrite(filepath, plot);
    std::cout << "[Visualizer] Plot saved to: " << filepath << std::endl;
}

/**
 * @brief 生成重投影误差对比曲线图
 */
void Visualizer::plotReprojErrorCurve(const Evaluator& evaluator) {
    std::map<std::string, std::vector<double>> data;
    for (const auto& method : evaluator.getMethodNames()) {
        const auto& frames = evaluator.getFrameData(method);
        std::vector<double> errors;
        for (const auto& f : frames) {
            errors.push_back(f.reprojection_error);
        }
        data[method] = errors;
    }
    plotCurve("Reprojection Error Comparison", "Error (px)", data,
              video_name_ + "_reproj_error.png");
}

/**
 * @brief 生成位姿曲线图 (球坐标系: 距离、方位角、仰角)
 */
void Visualizer::plotPoseCurves(const Evaluator& evaluator) {
    // 距离 (mm)
    {
        std::map<std::string, std::vector<double>> data;
        for (const auto& method : evaluator.getMethodNames()) {
            const auto& frames = evaluator.getFrameData(method);
            std::vector<double> vals;
            for (const auto& f : frames) vals.push_back(f.distance);
            data[method] = vals;
        }
        plotCurve("Distance", "Distance (mm)", data, video_name_ + "_pose_distance.png");
    }
    // 方位角 (度): atan2(x, z)
    {
        std::map<std::string, std::vector<double>> data;
        for (const auto& method : evaluator.getMethodNames()) {
            const auto& frames = evaluator.getFrameData(method);
            std::vector<double> vals;
            for (const auto& f : frames)
                vals.push_back(std::atan2(f.x, f.z) * 180.0 / CV_PI);
            data[method] = vals;
        }
        plotCurve("Azimuth (Yaw)", "Azimuth (deg)", data, video_name_ + "_pose_azimuth.png");
    }
    // 装甲板朝向 Yaw 角
    {
        std::map<std::string, std::vector<double>> data;
        for (const auto& method : evaluator.getMethodNames()) {
            const auto& frames = evaluator.getFrameData(method);
            std::vector<double> vals;
            for (const auto& f : frames) vals.push_back(f.yaw);
            data[method] = vals;
        }
        plotCurve("Yaw Angle", "Yaw (deg)", data, video_name_ + "_pose_yaw.png");
    }
}

/**
 * @brief 生成深度对比图
 */
void Visualizer::plotDepthComparison(const Evaluator& evaluator) {
    std::map<std::string, std::vector<double>> data;
    for (const auto& method : evaluator.getMethodNames()) {
        const auto& frames = evaluator.getFrameData(method);
        std::vector<double> vals;
        for (const auto& f : frames) vals.push_back(f.z);
        data[method] = vals;
    }
    plotCurve("Depth (Z) Comparison", "Z (mm)", data, video_name_ + "_depth_comparison.png");
}

/**
 * @brief 生成耗时柱状图
 */
void Visualizer::plotTimingBarChart(const Evaluator& evaluator) {
    std::vector<std::string> labels;
    std::vector<double> values;
    std::vector<cv::Scalar> colors;

    for (const auto& method : evaluator.getMethodNames()) {
        auto stats = evaluator.computeSummary(method);
        labels.push_back(method);
        values.push_back(stats.mean_compute_time_ms);
        colors.push_back(getMethodColor(method));
    }

    plotBarChart("Average Computation Time", labels, values, colors,
                 video_name_ + "_timing.png");
}

/**
 * @brief 绘制柱状图
 */
void Visualizer::plotBarChart(const std::string& title,
                              const std::vector<std::string>& labels,
                              const std::vector<double>& values,
                              const std::vector<cv::Scalar>& colors,
                              const std::string& filename) {
    int width = 800, height = 500;
    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    int margin_left = 100, margin_right = 50, margin_top = 60, margin_bottom = 80;
    int plot_width = width - margin_left - margin_right;
    int plot_height = height - margin_top - margin_bottom;

    // 绘制标题
    cv::putText(plot, title, cv::Point(width / 2 - 120, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    if (values.empty()) return;

    double max_val = *std::max_element(values.begin(), values.end());
    if (max_val <= 0) max_val = 1.0;

    int n = static_cast<int>(labels.size());
    int bar_width = plot_width / (n * 2);

    for (int i = 0; i < n; ++i) {
        int bar_height = static_cast<int>(values[i] / max_val * plot_height * 0.9);
        int x = margin_left + i * (plot_width / n) + (plot_width / n - bar_width) / 2;
        int y = margin_top + plot_height - bar_height;

        // 绘制柱子
        cv::rectangle(plot, cv::Point(x, y), cv::Point(x + bar_width, margin_top + plot_height),
                     colors[i], -1);
        cv::rectangle(plot, cv::Point(x, y), cv::Point(x + bar_width, margin_top + plot_height),
                     cv::Scalar(0, 0, 0), 1);

        // 绘制数值
        cv::putText(plot, cv::format("%.2f ms", values[i]),
                   cv::Point(x - 10, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

        // 绘制标签
        cv::putText(plot, labels[i], cv::Point(x, margin_top + plot_height + 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    }

    // 绘制 Y 轴
    cv::line(plot, cv::Point(margin_left, margin_top),
            cv::Point(margin_left, margin_top + plot_height), cv::Scalar(0, 0, 0), 1);

    std::string filepath = output_dir_ + "/" + filename;
    cv::imwrite(filepath, plot);
    std::cout << "[Visualizer] Bar chart saved to: " << filepath << std::endl;
}

/**
 * @brief 生成 X 轴和 Y 轴的独立时序图
 */
void Visualizer::plotXYCurves(const Evaluator& evaluator) {
    // 绘制 X 轴时序图
    std::map<std::string, std::vector<double>> data_x;
    for (const auto& method : evaluator.getMethodNames()) {
        const auto& frames = evaluator.getFrameData(method);
        std::vector<double> vals;
        for (const auto& f : frames) vals.push_back(f.x);
        data_x[method] = vals;
    }
    plotCurve("X Coordinate Timeline", "X (mm)", data_x, video_name_ + "_pose_x.png");

    // 绘制 Y 轴时序图
    std::map<std::string, std::vector<double>> data_y;
    for (const auto& method : evaluator.getMethodNames()) {
        const auto& frames = evaluator.getFrameData(method);
        std::vector<double> vals;
        for (const auto& f : frames) vals.push_back(f.y);
        data_y[method] = vals;
    }
    plotCurve("Y Coordinate Timeline", "Y (mm)", data_y, video_name_ + "_pose_y.png");
}

/**
 * @brief 生成 X-Z 平面的鸟瞰轨迹图 (Bird's Eye View Trajectory)
 */
void Visualizer::plotTrajectoryXZ(const Evaluator& evaluator) {
    int width = 800, height = 800;
    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    int margin = 80;
    int plot_w = width - 2 * margin;
    int plot_h = height - 2 * margin;

    // 1. 遍历所有数据，寻找 X 和 Z 的物理极值，用于建立坐标系
    double x_min = 1e10, x_max = -1e10, z_min = 1e10, z_max = -1e10;
    for (const auto& method : evaluator.getMethodNames()) {
        for (const auto& f : evaluator.getFrameData(method)) {
            x_min = std::min(x_min, f.x);
            x_max = std::max(x_max, f.x);
            z_min = std::min(z_min, f.z);
            z_max = std::max(z_max, f.z);
        }
    }

    if (x_max <= x_min || z_max <= z_min) return; // 无效数据保护

    // 两边留白 10%
    double x_range = x_max - x_min; 
    x_min -= x_range * 0.1; x_max += x_range * 0.1; x_range = x_max - x_min;
    
    double z_range = z_max - z_min; 
    z_min -= z_range * 0.1; z_max += z_range * 0.1; z_range = z_max - z_min;

    // 2. 绘制网格与边框
    cv::rectangle(plot, cv::Point(margin, margin), cv::Point(margin + plot_w, margin + plot_h), cv::Scalar(0, 0, 0), 1);
    
    // 绘制 Z 轴辅助线 (横线)
    for (int i = 1; i <= 5; ++i) {
        int y = margin + plot_h - i * plot_h / 5;
        double z_val = z_min + i * z_range / 5.0;
        cv::line(plot, cv::Point(margin, y), cv::Point(margin + plot_w, y), cv::Scalar(230, 230, 230), 1);
        cv::putText(plot, cv::format("%.0f", z_val), cv::Point(margin - 45, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
    }
    
    // 绘制 X 轴辅助线 (竖线)
    for (int i = 1; i <= 5; ++i) {
        int x = margin + i * plot_w / 5;
        double x_val = x_min + i * x_range / 5.0;
        cv::line(plot, cv::Point(x, margin), cv::Point(x, margin + plot_h), cv::Scalar(230, 230, 230), 1);
        cv::putText(plot, cv::format("%.0f", x_val), cv::Point(x - 15, margin + plot_h + 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
    }

    // 3. 绘制各方法的 X-Z 轨迹
    int legend_y = margin + 20;
    std::vector<std::string> methods = evaluator.getMethodNames();
    // 强制逆序遍历方法，确保蓝线覆盖红线
    for (auto it = methods.rbegin(); it != methods.rend(); ++it) {
        const std::string& method = *it;
        const auto& frames = evaluator.getFrameData(method);
        if (frames.empty()) continue;

        cv::Scalar color = getMethodColor(method);
        
        // 连线与散点
        for (size_t i = 1; i < frames.size(); ++i) {
            int x1 = margin + static_cast<int>((frames[i-1].x - x_min) / x_range * plot_w);
            int y1 = margin + plot_h - static_cast<int>((frames[i-1].z - z_min) / z_range * plot_h);
            int x2 = margin + static_cast<int>((frames[i].x - x_min) / x_range * plot_w);
            int y2 = margin + plot_h - static_cast<int>((frames[i].z - z_min) / z_range * plot_h);
            
            // 画轨迹线段
            cv::line(plot, cv::Point(x1, y1), cv::Point(x2, y2), color, 1, cv::LINE_AA);
            // 画数据点
            cv::circle(plot, cv::Point(x2, y2), 2, color, -1);
        }

        // 绘制图例
        cv::line(plot, cv::Point(margin + 20, legend_y), cv::Point(margin + 40, legend_y), color, 2);
        cv::putText(plot, method, cv::Point(margin + 45, legend_y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        legend_y += 25;
    }

    // 4. 标题与坐标轴标签
    cv::putText(plot, "Bird's Eye View Trajectory (X-Z Plane)", cv::Point(width / 2 - 180, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(plot, "X (mm) -> Right", cv::Point(width / 2 - 50, height - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(plot, "Z (mm) -> Depth", cv::Point(10, height / 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    // 5. 保存图像
    std::string filepath = output_dir_ + "/" + video_name_ + "_trajectory_xz.png";
    cv::imwrite(filepath, plot);
    std::cout << "[Visualizer] X-Z Trajectory saved to: " << filepath << std::endl;
}

/**
 * @brief 生成所有图表 (更新加入新增的 3 张图表)
 */
void Visualizer::generateAllPlots(const Evaluator& evaluator) {
    plotReprojErrorCurve(evaluator);
    plotPoseCurves(evaluator);
    plotDepthComparison(evaluator);
    plotTimingBarChart(evaluator);
    
    // 调用新增的图表绘制
    plotXYCurves(evaluator);
    plotTrajectoryXZ(evaluator);
}