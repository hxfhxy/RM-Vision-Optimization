/**
 * @file corner_refiner.hpp
 * @brief PCA + 梯度角点精调器
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class CornerRefiner {
public:
    static std::vector<cv::Point2f> refine(
        const cv::Mat& gray,
        const std::vector<cv::Point2f>& corners,
        int search_radius = 8,
        float min_grad_threshold = 30.0f);

    static std::pair<cv::Point2f, cv::Point2f> getLastAxes() { return last_axes_; }
    static std::pair<cv::Point2f, cv::Point2f> getLastCenters() { return last_centers_; }

private:
    static std::pair<cv::Point2f, cv::Point2f> last_axes_;
    static std::pair<cv::Point2f, cv::Point2f> last_centers_;
};
