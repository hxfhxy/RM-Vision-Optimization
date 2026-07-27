#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "armor_types.hpp"
#include <vector>
#include <string>
#include <fstream>

/**
 * @brief 精度评估器
 *
 * 计算多种评估指标:
 * 1. 平均/最大重投影误差 (像素)
 * 2. 深度方向标准差 (z 轴稳定性)
 * 3. 位姿平滑度 (相邻帧变化量的方差)
 * 4. 单帧耗时 (ms)
 */
class Evaluator {
public:
    /**
     * @brief 方法名称常量
     */
    static constexpr const char* METHOD_PNP = "YOLO";
    static constexpr const char* METHOD_BA = "BA";

    /**
     * @brief 单帧评估数据
     */
    struct FrameMetrics {
        uint64_t frame_index;
        double reprojection_error;      // 重投影误差 (像素)
        double x, y, z;                 // 位置 (mm)
        double yaw;                     // 偏航角 (度)
        double distance;                // 距离 (mm)
        double compute_time_ms;         // 计算耗时 (ms)
    };

    /**
     * @brief 汇总统计
     */
    struct SummaryStats {
        double mean_reproj_error;       // 平均重投影误差
        double max_reproj_error;        // 最大重投影误差
        double median_reproj_error;     // 中位重投影误差
        double std_reproj_error;        // 重投影误差标准差

        double z_std;                   // z 轴标准差 (深度稳定性)
        double z_range;                 // z 轴范围 (max - min)

        double pose_smoothness;         // 位姿平滑度 (相邻帧变化量的方差)
        double yaw_smoothness;          // yaw 角平滑度

        double mean_compute_time_ms;    // 平均计算耗时
        double max_compute_time_ms;     // 最大计算耗时

        uint64_t total_frames;          // 总帧数
    };

    /**
     * @brief 添加一帧的评估数据
     */
    void addFrame(const std::string& method, const FrameMetrics& metrics);

    /**
     * @brief 计算某方法的汇总统计
     */
    SummaryStats computeSummary(const std::string& method) const;

    /**
     * @brief 获取所有帧的数据
     */
    const std::vector<FrameMetrics>& getFrameData(const std::string& method) const;

    /**
     * @brief 获取所有方法名称
     */
    std::vector<std::string> getMethodNames() const;

    /**
     * @brief 将评估结果写入 CSV 文件
     */
    void writeToCSV(const std::string& filepath) const;

    /**
     * @brief 将汇总统计写入文本文件
     */
    void writeSummary(const std::string& filepath, const std::string& video_name) const;

    /**
     * @brief 打印汇总到控制台
     */
    void printSummary(const std::string& video_name) const;

private:
    // 存储各方法的帧数据
    std::map<std::string, std::vector<FrameMetrics>> frame_data_;

    /**
     * @brief 计算向量的均值
     */
    static double mean(const std::vector<double>& v);

    /**
     * @brief 计算向量的标准差
     */
    static double stddev(const std::vector<double>& v);

    /**
     * @brief 计算向量的中位数
     */
    static double median(std::vector<double> v);
};

#endif // EVALUATOR_HPP
