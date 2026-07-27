#ifndef ARMOR_TYPES_HPP
#define ARMOR_TYPES_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <cmath>

/**
 * @brief 装甲板检测结果结构体
 * @note 包含YOLO检测的角点、PnP解算的位姿、置信度等信息
 */
struct DetectedArmor {
    cv::RotatedRect rect;                       // 最小外接旋转矩形
    std::vector<cv::Point2f> pnp_corners;       // PnP角点 (4个: 左上、左下、右下、右上)
    double yaw = 0.0;                           // 偏航角 (度)
    double pitch = 0.0;                         // 俯仰角 (度)
    double distance = 0.0;                      // 到相机的距离 (mm)
    double dist_to_center = 0.0;                // 到图像中心的距离 (像素)
    double confidence = 0.0;                    // 检测置信度
    cv::Mat tvec;                               // 平移向量 (3x1, mm)
    cv::Mat rvec;                               // 旋转向量 (3x1, Rodrigues)
    int number = -1;                            // 识别出的数字ID
    float class_confidence = 0.0f;              // 分类置信度
    double armor_yaw_rad = 0.0;                 // 装甲板自身朝向角 (弧度)
};

/**
 * @brief 位姿结果结构体 (用于优化器输出)
 */
struct PoseResult {
    double x = 0.0;// 位置 x (mm)
    double y = 0.0;// 位置 y (mm)
    double z = 0.0;// 位置 z (mm)
    double yaw = 0.0;// 偏航角 (度)
    double pitch = 0.0;// 俯仰角 (度)
    double distance = 0.0;// 到相机的距离 (mm)
    double reprojection_error = 0.0;// 重投影误差 (像素)
    double compute_time_ms = 0.0;// 优化计算耗时 (毫秒)
    cv::Mat rvec;// 旋转向量 (3x1, Rodrigues)
    cv::Mat tvec;// 平移向量 (3x1, mm)
    PoseResult() = default;
};

    /**
     * @brief 从 tvec 和 rvec 构造 PoseResult
     *
     * yaw 提取策略:
     * - 如果 R 接近 Rz(yaw) (Z 轴旋转), 用 atan2(R(1,0), R(0,0))
     * - 否则用法向量方法 (兼容 PnP 的完整旋转矩阵)
     */
    static PoseResult fromRvecTvec(const cv::Mat& rvec, const cv::Mat& tvec) {
        PoseResult res;
        res.tvec = tvec.clone();
        res.rvec = rvec.clone();
        res.x = tvec.at<double>(0, 0);
        res.y = tvec.at<double>(1, 0);
        res.z = tvec.at<double>(2, 0);
        res.distance = std::sqrt(res.x * res.x + res.y * res.y + res.z * res.z);

        // 从旋转矩阵提取 yaw
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        // 检查 R 是否接近 Rz(yaw) (即 z 列接近 [0,0,1])
        double z_col_deviation = std::abs(R.at<double>(0, 2)) + std::abs(R.at<double>(1, 2));
        if (z_col_deviation < 0.1) {
            // R ≈ Rz(yaw), 直接从 R(0,0) 和 R(1,0) 提取 yaw
            res.yaw = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0)) * 180.0 / CV_PI;
        } else {
            // 使用法向量方法 (兼容 PnP 的完整旋转矩阵)
            cv::Vec3d normal(R.at<double>(0, 2), R.at<double>(1, 2), R.at<double>(2, 2));
            cv::Vec3d pos(res.x, res.y, res.z);
            if (normal.dot(pos) > 0) {
                normal = -normal;
            }
            res.yaw = std::atan2(normal[0], normal[2]) * 180.0 / CV_PI;
        }
        res.pitch = std::atan2(-res.y, res.z) * 180.0 / CV_PI;

        return res;
    }

/**
 * @brief 装甲板物理尺寸常量
 */
namespace ArmorModel {
    // 小装甲板尺寸 (mm): 135 x 55
    constexpr double SMALL_WIDTH = 135.0;
    constexpr double LARGE_WIDTH = 230.0;
    constexpr double HEIGHT = 55.0;
    constexpr double TILT_ANGLE_DEG = 15.0;     // 装甲板后倾角 (度)

    /**
     * @brief 获取装甲板 3D 模型点 (带后倾角补偿)
     * @param is_small 是否为小装甲板
     * @return 4个3D点 (左上、左下、右下、右上)
     */
    inline std::vector<cv::Point3f> getModelPoints(bool is_small) {
        double half_w = is_small ? (SMALL_WIDTH / 2.0) : (LARGE_WIDTH / 2.0);
        double half_h = HEIGHT / 2.0;
        double tilt = TILT_ANGLE_DEG * CV_PI / 180.0;

        double z_top = half_h * std::sin(tilt);
        double z_bottom = -half_h * std::sin(tilt);
        double y_top = -half_h * std::cos(tilt);
        double y_bottom = half_h * std::cos(tilt);

        return {
            cv::Point3f(static_cast<float>(-half_w), static_cast<float>(y_top), static_cast<float>(z_top)),
            cv::Point3f(static_cast<float>(-half_w), static_cast<float>(y_bottom), static_cast<float>(z_bottom)),
            cv::Point3f(static_cast<float>( half_w), static_cast<float>(y_bottom), static_cast<float>(z_bottom)),
            cv::Point3f(static_cast<float>( half_w), static_cast<float>(y_top), static_cast<float>(z_top))
        };
    }

    /**
     * @brief 根据像素宽高比判断大小装甲板
     */
    inline bool isSmallArmor(const std::vector<cv::Point2f>& corners) {
        float pixel_width = cv::norm(corners[0] - corners[3]);
        float pixel_height = cv::norm(corners[0] - corners[1]);
        return (pixel_width / pixel_height) < 3.2f;
    }
}

#endif // ARMOR_TYPES_HPP
