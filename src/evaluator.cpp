#include "evaluator.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

/**
 * @brief 添加一帧的评估数据
 */
void Evaluator::addFrame(const std::string& method, const FrameMetrics& metrics) {
    frame_data_[method].push_back(metrics);
}

/**
 * @brief 计算某方法的汇总统计
 */
Evaluator::SummaryStats Evaluator::computeSummary(const std::string& method) const {
    SummaryStats stats = {};
    auto it = frame_data_.find(method);
    if (it == frame_data_.end() || it->second.empty()) {
        return stats;
    }

    const auto& frames = it->second;
    stats.total_frames = frames.size();

    // 提取各维度数据
    std::vector<double> reproj_errors, z_vals, compute_times;
    std::vector<double> pose_deltas, yaw_deltas;

    for (size_t i = 0; i < frames.size(); ++i) {
        reproj_errors.push_back(frames[i].reprojection_error);
        z_vals.push_back(frames[i].z);
        compute_times.push_back(frames[i].compute_time_ms);

        // 计算相邻帧变化量
        if (i > 0) {
            double dx = frames[i].x - frames[i-1].x;
            double dy = frames[i].y - frames[i-1].y;
            double dz = frames[i].z - frames[i-1].z;
            pose_deltas.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));

            double dyaw = frames[i].yaw - frames[i-1].yaw;
            // 处理角度环绕
            while (dyaw > 180.0) dyaw -= 360.0;
            while (dyaw < -180.0) dyaw += 360.0;
            yaw_deltas.push_back(std::abs(dyaw));
        }
    }

    // 重投影误差统计
    stats.mean_reproj_error = mean(reproj_errors);
    stats.max_reproj_error = *std::max_element(reproj_errors.begin(), reproj_errors.end());
    stats.median_reproj_error = median(reproj_errors);
    stats.std_reproj_error = stddev(reproj_errors);

    // z 轴稳定性
    stats.z_std = stddev(z_vals);
    auto [z_min, z_max] = std::minmax_element(z_vals.begin(), z_vals.end());
    stats.z_range = *z_max - *z_min;

    // 平滑度 (变化量的方差, 越小越平滑)
    stats.pose_smoothness = pose_deltas.empty() ? 0.0 : stddev(pose_deltas);
    stats.yaw_smoothness = yaw_deltas.empty() ? 0.0 : stddev(yaw_deltas);

    // 耗时统计
    stats.mean_compute_time_ms = mean(compute_times);
    stats.max_compute_time_ms = *std::max_element(compute_times.begin(), compute_times.end());

    return stats;
}

/**
 * @brief 获取所有帧的数据
 */
const std::vector<Evaluator::FrameMetrics>& Evaluator::getFrameData(const std::string& method) const {
    static const std::vector<FrameMetrics> empty;
    auto it = frame_data_.find(method);
    return (it != frame_data_.end()) ? it->second : empty;
}

/**
 * @brief 获取所有方法名称
 */
std::vector<std::string> Evaluator::getMethodNames() const {
    std::vector<std::string> names;
    for (const auto& pair : frame_data_) {
        names.push_back(pair.first);
    }
    return names;
}

/**
 * @brief 自动判断场景类型
 *
 * 基于 PnP 结果的位姿变化特征:
 * - 计算帧间位移和 yaw 变化的均值
 * - 根据阈值分类
 */
/**
 * @brief 将评估结果写入 CSV 文件
 */
void Evaluator::writeToCSV(const std::string& filepath) const {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        std::cerr << "[Evaluator] Cannot open file: " << filepath << std::endl;
        return;
    }

    // 写入表头
    ofs << "frame_index";
    for (const auto& method : getMethodNames()) {
        ofs << "," << method << "_reproj_error"
            << "," << method << "_x"
            << "," << method << "_y"
            << "," << method << "_z"
            << "," << method << "_yaw"
            << "," << method << "_time_ms";
    }
    ofs << std::endl;

    // 找到最大帧数
    size_t max_frames = 0;
    for (const auto& pair : frame_data_) {
        max_frames = std::max(max_frames, pair.second.size());
    }

    // 写入数据
    for (size_t i = 0; i < max_frames; ++i) {
        ofs << i;
        for (const auto& method : getMethodNames()) {
            const auto& frames = frame_data_.at(method);
            if (i < frames.size()) {
                ofs << "," << std::fixed << std::setprecision(4) << frames[i].reprojection_error
                    << "," << std::fixed << std::setprecision(2) << frames[i].x
                    << "," << std::fixed << std::setprecision(2) << frames[i].y
                    << "," << std::fixed << std::setprecision(2) << frames[i].z
                    << "," << std::fixed << std::setprecision(2) << frames[i].yaw
                    << "," << std::fixed << std::setprecision(3) << frames[i].compute_time_ms;
            } else {
                ofs << ",,,,,";
            }
        }
        ofs << std::endl;
    }

    ofs.close();
    std::cout << "[Evaluator] CSV written to: " << filepath << std::endl;
}

/**
 * @brief 将汇总统计写入文本文件
 */
void Evaluator::writeSummary(const std::string& filepath, const std::string& video_name) const {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        std::cerr << "[Evaluator] Cannot open file: " << filepath << std::endl;
        return;
    }

    ofs << "=== 装甲板位姿优化评估报告 ===" << std::endl;
    ofs << "视频: " << video_name << std::endl;
    ofs << std::endl;

    for (const auto& method : getMethodNames()) {
        SummaryStats stats = computeSummary(method);
        ofs << "--- " << method << " ---" << std::endl;
        ofs << "  总帧数:           " << stats.total_frames << std::endl;
        ofs << "  平均重投影误差:   " << std::fixed << std::setprecision(3) << stats.mean_reproj_error << " px" << std::endl;
        ofs << "  最大重投影误差:   " << stats.max_reproj_error << " px" << std::endl;
        ofs << "  中位重投影误差:   " << stats.median_reproj_error << " px" << std::endl;
        ofs << "  重投影误差标准差: " << stats.std_reproj_error << " px" << std::endl;
        ofs << "  Z轴标准差:        " << std::fixed << std::setprecision(2) << stats.z_std << " mm" << std::endl;
        ofs << "  Z轴范围:          " << stats.z_range << " mm" << std::endl;
        ofs << "  位姿平滑度:       " << std::fixed << std::setprecision(3) << stats.pose_smoothness << " mm" << std::endl;
        ofs << "  Yaw角平滑度:      " << stats.yaw_smoothness << " deg" << std::endl;
        ofs << "  平均耗时:         " << std::fixed << std::setprecision(3) << stats.mean_compute_time_ms << " ms" << std::endl;
        ofs << "  最大耗时:         " << stats.max_compute_time_ms << " ms" << std::endl;
        ofs << std::endl;
    }

    ofs.close();
    std::cout << "[Evaluator] Summary written to: " << filepath << std::endl;
}

/**
 * @brief 打印汇总到控制台
 */
void Evaluator::printSummary(const std::string& video_name) const {
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  视频: " << video_name << std::endl;
    std::cout << "========================================" << std::endl;

    // 打印表头
    std::cout << std::left << std::setw(12) << "方法"
              << std::right << std::setw(10) << "平均误差"
              << std::setw(10) << "最大误差"
              << std::setw(10) << "Z轴标准差"
              << std::setw(10) << "平滑度"
              << std::setw(10) << "平均耗时"
              << std::endl;
    std::cout << std::string(62, '-') << std::endl;

    for (const auto& method : getMethodNames()) {
        SummaryStats stats = computeSummary(method);
        std::cout << std::left << std::setw(12) << method
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(10) << stats.mean_reproj_error
                  << std::setw(10) << stats.max_reproj_error
                  << std::setw(10) << stats.z_std
                  << std::setw(10) << stats.pose_smoothness
                  << std::setw(10) << stats.mean_compute_time_ms
                  << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief 计算均值
 */
double Evaluator::mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

/**
 * @brief 计算标准差
 */
double Evaluator::stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = mean(v);
    double sum_sq = 0.0;
    for (double x : v) {
        sum_sq += (x - m) * (x - m);
    }
    return std::sqrt(sum_sq / (v.size() - 1));
}

/**
 * @brief 计算中位数
 */
double Evaluator::median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) / 2.0 : v[n/2];
}
