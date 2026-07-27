/**
 * @file ba_optimizer.cpp
 * @brief 纯单帧 4-DoF BA 优化器 (已彻底剔除时序惩罚与热启动，保留 IRLS 加权与 Nielsen 更新)
 */

#include "ba_optimizer.hpp"
#include <chrono>
#include <cmath>
#include <Eigen/Dense>

ArmorBAOptimizer::ArmorBAOptimizer(const cv::Mat& K, const cv::Mat& dist,
                                   int max_iter, double converge_thresh)
    : K_(K.clone()), 
      dist_(dist.clone()),
      fx_(K.at<double>(0, 0)),
      fy_(K.at<double>(1, 1)),
      cx_(K.at<double>(0, 2)),
      cy_(K.at<double>(1, 2)),
      max_iter_(max_iter), 
      converge_thresh_(converge_thresh),
      huber_delta_(1.0),           
      last_pos_(Eigen::Vector3d::Zero()), 
      has_last_pos_(false),
      temporal_weight_(0.0) // 权重强制归零
{
    // 从 dist 矩阵安全提取畸变参数，用于手工解卷的高速投影
    if (!dist_.empty()) {
        const double* dptr = dist_.ptr<double>(0);
        k1_ = dist_.cols > 0 ? dptr[0] : 0.0;
        k2_ = dist_.cols > 1 ? dptr[1] : 0.0;
        p1_ = dist_.cols > 2 ? dptr[2] : 0.0;
        p2_ = dist_.cols > 3 ? dptr[3] : 0.0;
        k3_ = dist_.cols > 4 ? dptr[4] : 0.0;
    } else {
        k1_ = k2_ = p1_ = p2_ = k3_ = 0.0;
    }
}

/**
 * @brief 4-DoF 专属极速投影函数 (完全手工展开，0 OpenCV 调用开销)
 */
Eigen::Vector2d ArmorBAOptimizer::project4DoF(const Eigen::Vector3d& pos, double yaw,
                                              const cv::Point3f& model_pt) const {
    double cyaw = std::cos(yaw), syaw = std::sin(yaw);

    Eigen::Matrix3d R_yaw;
    R_yaw <<  cyaw, 0.0,  syaw,
               0.0, 1.0,   0.0,
             -syaw, 0.0,  cyaw;

    bool is_flat = (std::abs(model_pt.z) < 1e-3);
    Eigen::Matrix3d R;
    if (is_flat) {
        double pitch = -15.0 * CV_PI / 180.0;
        double cp = std::cos(pitch), sp = std::sin(pitch);
        Eigen::Matrix3d R_pitch;
        R_pitch << 1.0, 0.0, 0.0,
                   0.0,  cp, -sp,
                   0.0,  sp,  cp;
        R = R_yaw * R_pitch;
    } else {
        R = R_yaw;
    }

    Eigen::Vector3d pt_model(model_pt.x, model_pt.y, model_pt.z);
    Eigen::Vector3d pt_cam = R * pt_model + pos;

    double x = pt_cam.x() / pt_cam.z();
    double y = pt_cam.y() / pt_cam.z();

    // 纯手工底层畸变计算，榨干最后一点 CPU 性能
    double r2 = x * x + y * y;
    double radial = 1.0 + k1_ * r2 + k2_ * r2 * r2 + k3_ * r2 * r2 * r2;
    double x_distorted = x * radial + 2.0 * p1_ * x * y + p2_ * (r2 + 2.0 * x * x);
    double y_distorted = y * radial + p1_ * (r2 + 2.0 * y * y) + 2.0 * p2_ * x * y;
    
    return Eigen::Vector2d(fx_ * x_distorted + cx_, fy_ * y_distorted + cy_);
}

/**
 * @brief 计算残差向量 (为了不改头文件，保留 11 维签名，但后 3 维强制置 0 阉割时序)
 */
Eigen::Matrix<double, 11, 1> ArmorBAOptimizer::computeResiduals4DoF(
    const Eigen::Vector3d& pos, double yaw,
    const std::vector<cv::Point2f>& observations,
    const std::vector<cv::Point3f>& model_points) const {

    Eigen::Matrix<double, 11, 1> residuals;
    residuals.setZero(); // 全局置0，后三维天然为 0

    // 只保留前 8 维的纯像素投影误差
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector2d proj = project4DoF(pos, yaw, model_points[i]);
        residuals(2 * i)     = observations[i].x - proj.x();
        residuals(2 * i + 1) = observations[i].y - proj.y();
    }
    
    // 【已删除】完全剔除时序关联项 residuals(8..10) 的计算
    return residuals;
}

/**
 * @brief 计算数值雅可比矩阵
 */
Eigen::Matrix<double, 11, 4> ArmorBAOptimizer::computeJacobian4DoF(
    const Eigen::Vector3d& pos, double yaw,
    const std::vector<cv::Point2f>& observations,
    const std::vector<cv::Point3f>& model_points) const {

    Eigen::Matrix<double, 11, 4> J;
    J.setZero(); // 强行将矩阵全部初始化为 0，防止内存野值污染时序行

    const double eps_pos = 1e-4; // 精度 0.1um，保障局部线性化不失真    
    const double eps_yaw = 1e-4;    

    for (int j = 0; j < 3; ++j) {
        Eigen::Vector3d pos_p = pos, pos_m = pos;
        pos_p(j) += eps_pos;
        pos_m(j) -= eps_pos;
        auto r_plus = computeResiduals4DoF(pos_p, yaw, observations, model_points);
        auto r_minus = computeResiduals4DoF(pos_m, yaw, observations, model_points);
        J.col(j) = (r_plus - r_minus) / (2.0 * eps_pos);
    }

    auto r_plus = computeResiduals4DoF(pos, yaw + eps_yaw, observations, model_points);
    auto r_minus = computeResiduals4DoF(pos, yaw - eps_yaw, observations, model_points);
    J.col(3) = (r_plus - r_minus) / (2.0 * eps_yaw);

    return J;
}

PoseResult ArmorBAOptimizer::optimize(const cv::Mat& pnp_tvec, const cv::Mat& pnp_rvec,
                                      const std::vector<cv::Point2f>& observations,
                                      const std::vector<cv::Point3f>& model_points) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // 强行使用 PnP 提供的解作为初值
    Eigen::Vector3d pos(pnp_tvec.at<double>(0, 0), pnp_tvec.at<double>(1, 0), pnp_tvec.at<double>(2, 0));

    cv::Mat R_pnp_cv;
    cv::Rodrigues(pnp_rvec, R_pnp_cv);
    // 直接提取 Yaw，绝对不能人为添加 180 度的翻转！
    double yaw = std::atan2(R_pnp_cv.at<double>(0, 2), R_pnp_cv.at<double>(2, 2));

    // 【已删除】：时序热启动 (Warm Start) 已被彻底剔除

    auto residuals = computeResiduals4DoF(pos, yaw, observations, model_points);

    // IRLS 初始参数
    double lambda = 1e-3;
    double v = 2.0;

    // LM 迭代求解主循环
    for (int iter = 0; iter < max_iter_; ++iter) {
        Eigen::Matrix<double, 11, 4> J = computeJacobian4DoF(pos, yaw, observations, model_points);
        
        // 构建极速的 IRLS (迭代重加权) 矩阵 W
        Eigen::DiagonalMatrix<double, 11> W;
        W.setIdentity();
        double current_cost = 0.0;
        
        // 仅计算 4 个角点（前 8 维）的误差与权重
        for (int i = 0; i < 4; ++i) {
            double err_x = residuals(2 * i);
            double err_y = residuals(2 * i + 1);
            double e = std::hypot(err_x, err_y);
            if (e <= huber_delta_) {
                current_cost += 0.5 * e * e;
            } else {
                current_cost += huber_delta_ * (e - 0.5 * huber_delta_);
                double weight = huber_delta_ / e;
                W.diagonal()[2 * i] = weight;
                W.diagonal()[2 * i + 1] = weight;
            }
        }
        
        // 【已删除】剔除了 current_cost 关于后 3 维时序惩罚的累加

        // H = J^T * W * J  ,  g = -J^T * W * r (标准牛顿降维矩阵法)
        Eigen::Matrix<double, 4, 4> H = J.transpose() * W * J;
        Eigen::Matrix<double, 4, 1> g = -J.transpose() * W * residuals;

        if (g.lpNorm<Eigen::Infinity>() < 1e-8) break; // 谷底探测

        // 阻尼叠加
        Eigen::Matrix<double, 4, 4> H_lm = H + lambda * Eigen::Matrix<double, 4, 4>::Identity();
        Eigen::Matrix<double, 4, 1> delta = H_lm.ldlt().solve(g);

        if (delta.norm() < converge_thresh_) break;

        // 步长试探：在标准矩阵论中，g = -J^T W r，所以更新方向是加法 (pos + delta)
        Eigen::Vector3d new_pos = pos + delta.head<3>();
        double new_yaw = yaw + delta(3);
        new_yaw = std::atan2(std::sin(new_yaw), std::cos(new_yaw)); 

        auto new_residuals = computeResiduals4DoF(new_pos, new_yaw, observations, model_points);
        
        // 试算 New Cost (仅纯像素)
        double new_cost = 0.0;
        for (int i = 0; i < 4; ++i) {
            double err_x = new_residuals(2 * i);
            double err_y = new_residuals(2 * i + 1);
            double e = std::hypot(err_x, err_y);
            if (e <= huber_delta_) new_cost += 0.5 * e * e;
            else new_cost += huber_delta_ * (e - 0.5 * huber_delta_);
        }
        // 【已删除】剔除了 new_cost 关于后 3 维时序惩罚的累加

        // Nielsen's 阻尼更新策略
        double actual_drop = current_cost - new_cost;
        double predicted_drop = 0.5 * delta.dot(lambda * delta + g);
        double rho = (predicted_drop > 0.0) ? (actual_drop / predicted_drop) : 0.0;

        if (rho > 0) {
            // 采纳步长
            pos = new_pos;
            yaw = new_yaw;
            residuals = new_residuals;
            // 极其丝滑的降低阻尼，模型越符合预期降得越快
            lambda *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3.0));
            v = 2.0; // 重置惩罚因子
        } else {
            // 拒绝步长：采用倍增惩罚因子，迅速退化回梯度下降
            lambda *= v;
            v *= 2.0;
        }
    }

    // 【已删除】彻底剔除了 last_pos_ 的写入记录

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    PoseResult result;
    result.x = pos(0);
    result.y = pos(1);
    result.z = pos(2);
    result.distance = pos.norm();
    result.compute_time_ms = elapsed_ms;

    bool is_flat = (std::abs(model_points[0].z) < 1e-3);
    double cyaw = std::cos(yaw), syaw = std::sin(yaw);
    Eigen::Matrix3d R_yaw;
    R_yaw <<  cyaw, 0.0,  syaw,
               0.0, 1.0,   0.0,
             -syaw, 0.0,  cyaw;

    Eigen::Matrix3d R_final;
    if (is_flat) {
        double pitch = -15.0 * CV_PI / 180.0;
        double cp = std::cos(pitch), sp = std::sin(pitch);
        Eigen::Matrix3d R_pitch;
        R_pitch << 1.0, 0.0, 0.0,
                   0.0,  cp, -sp,
                   0.0,  sp,  cp;
        R_final = R_yaw * R_pitch;
    } else {
        R_final = R_yaw;
    }

    cv::Mat R_final_cv = (cv::Mat_<double>(3, 3) <<
        R_final(0,0), R_final(0,1), R_final(0,2),
        R_final(1,0), R_final(1,1), R_final(1,2),
        R_final(2,0), R_final(2,1), R_final(2,2));
    cv::Rodrigues(R_final_cv, result.rvec);
    result.tvec = (cv::Mat_<double>(3, 1) << result.x, result.y, result.z);

    result.yaw = yaw * 180.0 / CV_PI;
    result.pitch = std::atan2(-result.y, result.z) * 180.0 / CV_PI;
    
    // 只输出前 8 维的纯像素误差
    result.reprojection_error = std::sqrt(residuals.head<8>().squaredNorm() / 4.0);

    return result;
}

double ArmorBAOptimizer::computeReprojError(const Eigen::Vector3d& pos,
                                            const Eigen::Vector3d& rvec,
                                            const std::vector<cv::Point2f>& observations,
                                            const std::vector<cv::Point3f>& model_points) {
    cv::Mat rvec_cv = (cv::Mat_<double>(3, 1) << rvec(0), rvec(1), rvec(2));
    cv::Mat R_cv;
    cv::Rodrigues(rvec_cv, R_cv);
    Eigen::Matrix3d R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r, c) = R_cv.at<double>(r, c);

    // 直接提取 Yaw，无多余翻转
    double yaw = std::atan2(R(0, 2), R(2, 2));

    Eigen::Matrix<double, 11, 1> residuals = computeResiduals4DoF(pos, yaw, observations, model_points);
    return std::sqrt(residuals.head<8>().squaredNorm() / 4.0);
}