#include "pnp_solver_standalone.hpp"

/**
 * @brief PnP 解算: 使用 IPPE 算法 + 双解验证
 *
 * 流程:
 * 1. 根据像素宽高比判断大小装甲板
 * 2. 构建 3D 模型点 (带 15° 后倾角)
 * 3. 使用 solvePnPGeneric (IPPE) 获取两个解
 * 4. 双解验证 (左右歧义消除 + Pitch 验证)
 * 5. 提取位姿 (tvec, rvec, yaw, pitch, distance)
 */
void PnpSolverStandalone::solve(DetectedArmor& armor, const cv::Mat& K, const cv::Mat& dist) {
    // 角点: 0=左上, 1=左下, 2=右下, 3=右上
    std::vector<cv::Point2f> image_points = armor.pnp_corners;

    // 大小装甲板判断
    float pixel_width = cv::norm(armor.pnp_corners[0] - armor.pnp_corners[3]);
    float pixel_height = cv::norm(armor.pnp_corners[0] - armor.pnp_corners[1]);
    bool is_small = (pixel_width / pixel_height) < 3.2f;

    double half_w = is_small ? (135.0 / 2.0) : (230.0 / 2.0);
    double half_h = 55.0 / 2.0;

    // 装甲板后倾角
    double tilt = 15.0 * CV_PI / 180.0;
    double z_top = half_h * std::sin(tilt);
    double z_bottom = -half_h * std::sin(tilt);
    double y_top = -half_h * std::cos(tilt);
    double y_bottom = half_h * std::cos(tilt);

    // 3D 模型点
    std::vector<cv::Point3f> object_points_cv = {
        cv::Point3f(static_cast<float>(-half_w), static_cast<float>(y_top), static_cast<float>(z_top)),
        cv::Point3f(static_cast<float>(-half_w), static_cast<float>(y_bottom), static_cast<float>(z_bottom)),
        cv::Point3f(static_cast<float>( half_w), static_cast<float>(y_bottom), static_cast<float>(z_bottom)),
        cv::Point3f(static_cast<float>( half_w), static_cast<float>(y_top), static_cast<float>(z_top))
    };

    // IPPE 算法: 返回两个解
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<double> errors;
    int solutions = cv::solvePnPGeneric(
        object_points_cv, image_points, K, dist,
        rvecs, tvecs, false, cv::SOLVEPNP_IPPE, cv::noArray(), cv::noArray(), errors);

    if (solutions <= 0 || rvecs.empty() || tvecs.empty()) {
        armor.distance = 0;
        armor.yaw = 0;
        armor.pitch = 0;
        return;
    }

    // 双解验证
    int best = 0;
    if (rvecs.size() > 1 && tvecs.size() > 1) {
        double err0 = errors[0];
        double err1 = errors[1];

        // 提取 pitch 和 yaw 的 lambda
        auto get_pitch_yaw = [](const cv::Mat& rvec, double& pitch, double& yaw) {
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            yaw = std::atan2(R.at<double>(0, 2), R.at<double>(2, 2)) * 180.0 / CV_PI;
            pitch = std::atan2(-R.at<double>(1, 2),
                              std::sqrt(R.at<double>(0, 2) * R.at<double>(0, 2) +
                                       R.at<double>(2, 2) * R.at<double>(2, 2))) * 180.0 / CV_PI;
        };

        double pitch0, yaw0, pitch1, yaw1;
        get_pitch_yaw(rvecs[0], pitch0, yaw0);
        get_pitch_yaw(rvecs[1], pitch1, yaw1);

        // 左右歧义消除
        cv::Point2f left_axis = armor.pnp_corners[0] - armor.pnp_corners[1];
        cv::Point2f right_axis = armor.pnp_corners[3] - armor.pnp_corners[2];
        double armor_angle_trend = std::atan2(left_axis.x, left_axis.y) +
                                   std::atan2(right_axis.x, right_axis.y);
        bool yaw_match_1 = ((armor_angle_trend > 0 && yaw1 > 0 && yaw0 < 0) ||
                            (armor_angle_trend < 0 && yaw1 < 0 && yaw0 > 0));

        // Pitch 验证
        bool pitch_reasonable_0 = std::abs(pitch0) < 30.0;
        bool pitch_reasonable_1 = std::abs(pitch1) < 30.0;

        if (err1 < err0 * 1.5) {
            if (yaw_match_1 && pitch_reasonable_1 && !pitch_reasonable_0) {
                best = 1;
            } else if (yaw_match_1 && std::abs(pitch1) < std::abs(pitch0)) {
                best = 1;
            }
        }
    }

    // 提取位姿
    armor.tvec = tvecs[best].clone();
    armor.rvec = rvecs[best].clone();

    double x = armor.tvec.at<double>(0, 0);
    double y = armor.tvec.at<double>(1, 0);
    double z = armor.tvec.at<double>(2, 0);

    armor.distance = std::sqrt(x * x + y * y + z * z);

    // 计算 yaw
    cv::Mat R_best;
    cv::Rodrigues(armor.rvec, R_best);
    armor.yaw = std::atan2(R_best.at<double>(0, 2), R_best.at<double>(2, 2)) * 180.0 / CV_PI;
}
