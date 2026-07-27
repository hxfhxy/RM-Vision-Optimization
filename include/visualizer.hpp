#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include "armor_types.hpp"
#include "evaluator.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief 可视化模块
 */
class Visualizer {
public:
    Visualizer(const std::string& output_dir, const std::string& video_name);

    void drawReprojectionComparison(
        cv::Mat& image,
        const std::vector<cv::Point2f>& pnp_corners,
        const std::vector<cv::Point2f>& ba_corners,
        const std::vector<cv::Point2f>& g2o_corners,
        const std::vector<cv::Point2f>& temporal_corners,
        uint64_t frame_index);

    void writeVideoFrame(const cv::Mat& image);
    void writeVideoFrameBA(const cv::Mat& image);

    void plotReprojErrorCurve(const Evaluator& evaluator);
    void plotPoseCurves(const Evaluator& evaluator);
    void plotDepthComparison(const Evaluator& evaluator);
    
    // ================= 新增图表声明 =================
    void plotXYCurves(const Evaluator& evaluator);
    void plotTrajectoryXZ(const Evaluator& evaluator);
    // ===============================================

    void plotTimingBarChart(const Evaluator& evaluator);
    void generateAllPlots(const Evaluator& evaluator);
    void releaseVideo();

private:
    std::string output_dir_;
    std::string video_name_;
    cv::VideoWriter video_writer_;
    bool video_initialized_ = false;

    // 颜色常量
    static const cv::Scalar COLOR_PNP;
    static const cv::Scalar COLOR_BA;
    static const cv::Scalar COLOR_TEXT;

    void plotCurve(const std::string& title,
                   const std::string& y_label,
                   const std::map<std::string, std::vector<double>>& data_series,
                   const std::string& filename,
                   int image_width = 1200, int image_height = 600);

    void plotBarChart(const std::string& title,
                      const std::vector<std::string>& labels,
                      const std::vector<double>& values,
                      const std::vector<cv::Scalar>& colors,
                      const std::string& filename);

    cv::Scalar getMethodColor(const std::string& method) const;
    int getMethodLineWidth(const std::string& method) const;
    void ensureOutputDir() const;
};

#endif // VISUALIZER_HPP