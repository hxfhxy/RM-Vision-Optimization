#ifndef BA_OPTIMIZER_HPP
#define BA_OPTIMIZER_HPP

#include "armor_types.hpp"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

/**
 * @brief 时序 4-DoF LM-BA 优化器
 *
 * 优化变量: [X, Y, Z, Yaw]
 * 残差维度: 11x1 (8 维像素重投影误差 + 3 维 XYZ 时序平滑约束)
 */
class ArmorBAOptimizer {
public:
    ArmorBAOptimizer(const cv::Mat& K, const cv::Mat& dist,
                     int max_iter = 20, double converge_thresh = 1e-6);

    PoseResult optimize(const cv::Mat& pnp_tvec, const cv::Mat& pnp_rvec,
                        const std::vector<cv::Point2f>& observations,
                        const std::vector<cv::Point3f>& model_points);

    double computeReprojError(const Eigen::Vector3d& pos,
                              const Eigen::Vector3d& rvec,
                              const std::vector<cv::Point2f>& observations,
                              const std::vector<cv::Point3f>& model_points);

private:
    cv::Mat K_;                     
    cv::Mat dist_;                  
    double fx_, fy_, cx_, cy_;      

    double k1_ = 0.0;
    double k2_ = 0.0;
    double p1_ = 0.0;
    double p2_ = 0.0;
    double k3_ = 0.0;

    int max_iter_;                  
    double converge_thresh_;        
    double huber_delta_ = 1.0;      

    // ================== 时序平滑新增成员 ==================
    Eigen::Vector3d last_pos_;
    bool has_last_pos_ = false;
    double temporal_weight_ = 0.05; // 惩罚权重，越大越平滑
    // =====================================================

    Eigen::Vector2d project4DoF(const Eigen::Vector3d& pos, double yaw,
                                const cv::Point3f& model_pt) const;

    // 返回 11x1 残差 (包含时序惩罚)
    Eigen::Matrix<double, 11, 1> computeResiduals4DoF(
        const Eigen::Vector3d& pos, double yaw,
        const std::vector<cv::Point2f>& observations,
        const std::vector<cv::Point3f>& model_points) const;

    // 雅可比矩阵升维至 11x4
    Eigen::Matrix<double, 11, 4> computeJacobian4DoF(
        const Eigen::Vector3d& pos, double yaw,
        const std::vector<cv::Point2f>& observations,
        const std::vector<cv::Point3f>& model_points) const;
};

#endif // BA_OPTIMIZER_HPP