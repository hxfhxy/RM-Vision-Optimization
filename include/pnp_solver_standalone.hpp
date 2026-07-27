#ifndef PNP_SOLVER_STANDALONE_HPP
#define PNP_SOLVER_STANDALONE_HPP

#include "armor_types.hpp"
#include <opencv2/opencv.hpp>

/**
 * @brief PnP 解算器 (独立版本)
 *
 * 使用 IPPE 算法解算装甲板位姿, 包含双解验证
 */
class PnpSolverStandalone {
public:
    /**
     * @brief 使用 IPPE 算法解算 PnP
     * @param armor 装甲板检测结果 (输入角点, 输出位姿)
     * @param K 相机内参矩阵
     * @param dist 畸变系数
     */
    static void solve(DetectedArmor& armor, const cv::Mat& K, const cv::Mat& dist);
};

#endif // PNP_SOLVER_STANDALONE_HPP
