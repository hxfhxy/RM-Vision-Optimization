/**
 * @file corner_refiner.cpp
 * @brief 融合版亚像素角点精调: 完美复刻原版二次正交扫描与端点滑窗搜索
 */

#include "corner_refiner.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

/* 初始化静态时序变量 */
std::pair<cv::Point2f, cv::Point2f> CornerRefiner::last_axes_ = {{0, 1}, {0, 1}};
std::pair<cv::Point2f, cv::Point2f> CornerRefiner::last_centers_ = {{0, 0}, {0, 0}};

namespace {

// ==========================================================
// 严格保留原版超参数
// ==========================================================
constexpr int sample_points_K_ = 20;         // 扫点数：均匀取 15(实为 K-1) 个扫描截面
constexpr float scan_half_width_W_ = 8.0f;   // 扫描线：默认扫描半宽 8 像素
constexpr float endpoint_search_R_ = 12.0f;  // 断点处：端点搜索范围 ±12 像素

/**
 * @brief 原版双线性插值
 */
float BilinearInterpolation(const cv::Mat& img, float x, float y) {
    if (x < 0.0f || x >= (img.cols - 1) || y < 0.0f || y >= (img.rows - 1)) {
        return 0.0f;
    }
    int ix = (int)x;
    int iy = (int)y;
    float dx = x - ix; 
    float dy = y - iy; 

    float v00 = img.at<uchar>(iy, ix);         
    float v10 = img.at<uchar>(iy, ix + 1);     
    float v01 = img.at<uchar>(iy + 1, ix);     
    float v11 = img.at<uchar>(iy + 1, ix + 1); 

    return (1.0f - dx) * (1.0f - dy) * v00 + 
           dx * (1.0f - dy) * v10 + 
           (1.0f - dx) * dy * v01 + 
           dx * dy * v11;
}

/**
 * @brief 原版 ScanAndFit (扫描线寻找双峰 + 拟合)
 */
bool ScanAndFit(const cv::Mat& gray_img, 
                const cv::Point2f& pt_top, const cv::Point2f& pt_bottom,
                cv::Vec4f& exact_line, float dynamic_W) {
    
    cv::Point2f v = pt_bottom - pt_top;
    float length = (float)cv::norm(v);
    if (length < 1.0f) return false;

    cv::Point2f s = v / (sample_points_K_ - 1); 
    cv::Point2f u = v / length;                 
    cv::Point2f n = cv::Point2f(u.y, -u.x);     

    int W_int = (int)dynamic_W;
    int profile_length = W_int * 2 + 1;
    std::vector<cv::Point2f> center_pts; 

    for (int k = 2; k < sample_points_K_; k++) {
        cv::Point2f k_initial_center = pt_top + (k - 1) * s; 
        
        std::vector<float> profile(profile_length, 0.0f);
        float min_val = 255.0f;
        float max_val = 0.0f;
        int max_idx = -1;
        
        for (int offset = -W_int; offset <= W_int; offset++) {
            cv::Point2f p = k_initial_center + offset * n;
            float val = BilinearInterpolation(gray_img, p.x, p.y);
            profile[offset + W_int] = val;

            if (val > max_val) { 
                max_val = val; 
                max_idx = offset + W_int; 
            }
            if (val < min_val) { 
                min_val = val; 
            }
        }

        // 严格保留：对比度足够 (保底 25.0f)
        if (max_val - min_val > 25.0f && (max_idx > 0 && max_idx < profile_length - 1)) {
            float thresh = min_val + (max_val - min_val) * 0.5f;
            float left_idx = -1.0f;
            float right_idx = -1.0f;

            for (int i = max_idx; i > 0; i--) {
                if (profile[i] >= thresh && profile[i-1] < thresh) {
                    float t = (profile[i] - thresh) / (profile[i] - profile[i-1]);
                    left_idx = i - t;   
                    break;
                }
            }

            for (int i = max_idx; i < profile_length - 1; i++) {
                if (profile[i] >= thresh && profile[i+1] < thresh) {
                    float t = (profile[i] - thresh) / (profile[i] - profile[i+1]);
                    right_idx = i + t;
                    break;
                }
            }
        
            if (left_idx >= 0.0f && right_idx >= 0.0f) {
                cv::Point2f pt_l = k_initial_center + (left_idx - W_int) * n;
                cv::Point2f pt_r = k_initial_center + (right_idx - W_int) * n;
                cv::Point2f pt_c = (pt_l + pt_r) * 0.5f; 
                center_pts.push_back(pt_c);
            }
        }
    }

    if (center_pts.size() < 4) {
        return false;
    }

    // 使用系统自带的 Huber 替代不包含在项目内的 fitLineRANSAC
    cv::fitLine(center_pts, exact_line, cv::DIST_HUBER, 0, 0.01, 0.01);
    return true;
}

/**
 * @brief 原版 GetSubPixelCenterLine (迭代二次扫描)
 */
bool GetSubPixelCenterLine(const cv::Mat& gray_img, 
                           const cv::Point2f& pt_top, const cv::Point2f& pt_bottom,
                           cv::Vec4f& exact_line, float dynamic_W) {
    cv::Vec4f L0;
    // 第一次初步扫描
    if (!ScanAndFit(gray_img, pt_top, pt_bottom, L0, dynamic_W)) {
        return false;
    }

    cv::Point2f direction(L0[0], L0[1]);
    if (direction.y < 0) direction = -direction; 
    cv::Point2f p0(L0[2], L0[3]); 

    float t_top = (pt_top - p0).dot(direction); 
    float t_bottom = (pt_bottom - p0).dot(direction);
    cv::Point2f new_top = p0 + t_top * direction;
    cv::Point2f new_bottom = p0 + t_bottom * direction;

    // 【迭代 1】：二次绝对垂直法向扫描
    return ScanAndFit(gray_img, new_top, new_bottom, exact_line, dynamic_W);
}

/**
 * @brief 原版 SearchExactEndpoint (沿中轴线滑窗搜索真正端点)
 */
cv::Point2f SearchExactEndpoint(const cv::Mat& gray_img, 
                                const cv::Vec4f& exact_line, 
                                const cv::Point2f& rough_pt, 
                                bool is_top_point,
                                float bar_length) {
    cv::Point2f v(exact_line[0], exact_line[1]);    
    cv::Point2f p0(exact_line[2], exact_line[3]);   
    
    float t_proj = (rough_pt.x - p0.x) * v.x + (rough_pt.y - p0.y) * v.y;
    cv::Point2f project_pt = p0 + t_proj * v; 

    if (v.y < 0) v = -v; 
    cv::Point2f n(v.y, -v.x); 

    int R = std::min((int)endpoint_search_R_, std::max(4, (int)(bar_length * 0.75f)));

    // 【架构跃迁：定义绝对向内的行进向量】
    cv::Point2f scan_dir = is_top_point ? v : -v;

    // 扫描深度：从外部背景 (-R) 一直深入到灯条的绝对物理正中心
    int inner_dist = std::max((int)(bar_length * 0.5f), R);
    int search_len = R + inner_dist + 1; 
    std::vector<float> signal(search_len, 0.0f); 

    float max_val = 0.0f;
    int max_idx = -1;
    
    for (int i = 0; i < search_len; i++) {
        float step = (i - R) * 1.0f; 
        cv::Point2f current_p = project_pt + step * scan_dir; 

        // 横向取最大值，防细灯条微偏
        float left = BilinearInterpolation(gray_img, current_p.x - 0.5f * n.x, current_p.y - 0.5f * n.y); 
        float middle = BilinearInterpolation(gray_img, current_p.x,              current_p.y);              
        float right = BilinearInterpolation(gray_img, current_p.x + 0.5f * n.x, current_p.y + 0.5f * n.y); 
        
        signal[i] = std::max(left, std::max(middle, right)); 

        if (signal[i] > max_val) {
            max_val = signal[i];
            max_idx = i;
        }
    }

    float sub_offset = 0.0f; 
    bool found = false;

    // 获取绝对真实的外部背景底噪 
    float bg_val = (signal[0] + signal[1] + signal[2]) / 3.0f; 

    // 只有当核心亮度高于背景时才执行，防误切纯黑区域
    if (max_val - bg_val > 25.0f && max_idx > 0) {
        // 核心：背景底噪 + 真实跳变落差的 25% + 25.0f 保底
        float dynamic_thresh = bg_val + (max_val - bg_val) * 0.25f;
        float thresh = std::max(dynamic_thresh, bg_val + 25.0f);

        // 【唯一且无脑的平推】：永远从外部(索引0)出发，保险起见驶向终点
        for (int i = 0; i < search_len - 1; i++) {
            if (signal[i] <= thresh && signal[i+1] > thresh) {
                // 前瞻精修，寻找附近真正的峭壁
                int best_i = i;
                float max_grad = signal[i+1] - signal[i];
                int max_lookahead = 3; 

                for (int k = 1; k <= max_lookahead; k++) {
                    int curr = i + k; 
                    if (curr + 1 >= search_len || curr >= max_idx) {
                        break;
                    }
                    
                    float grad = signal[curr+1] - signal[curr]; 
                    
                    // 一旦梯度不再陡峭（开始变缓），果断踩死刹车！
                    if (grad <= max_grad * 0.9) break;
                    
                    max_grad = grad;
                    best_i = curr;
                }

                // 线性插值精确定位
                float t = (thresh - signal[best_i]) / (signal[best_i+1] - signal[best_i]);
                sub_offset = best_i + t - R; 
                found = true;
                break;
            }
        }
    }

    if (!found) {
        sub_offset = 0.0f; 
    }

    float final_step = sub_offset * 1.0f;
    return project_pt + final_step * scan_dir;
}

} // namespace


/* 暴露给外部流水线的公共接口 */
std::vector<cv::Point2f> CornerRefiner::refine(
    const cv::Mat &gray,
    const std::vector<cv::Point2f> &corners,
    int /*search_radius*/,
    float /*min_grad_threshold*/)
{
    if (corners.size() != 4 || gray.empty()) return corners;

    // 转灰度防御保护
    cv::Mat gray_img;
    if (gray.channels() == 3) cv::cvtColor(gray, gray_img, cv::COLOR_BGR2GRAY);
    else gray_img = gray;

    // 自适应计算扫描线左右半宽 W
    float top_width = (float)cv::norm(corners[3] - corners[0]);
    float bottom_width = (float)cv::norm(corners[2] - corners[1]);
    float armor_average_width = (top_width + bottom_width) / 2.0f;
    float dynamic_W = std::min(scan_half_width_W_, armor_average_width * 0.75f);

    std::vector<cv::Point2f> refined_corners = corners;
    cv::Vec4f left_line, right_line;

    // 对【左灯条】进行精化流水线处理
    if (GetSubPixelCenterLine(gray_img, corners[0], corners[1], left_line, dynamic_W)) {
        float left_length = (float)cv::norm(corners[1] - corners[0]);
        refined_corners[0] = SearchExactEndpoint(gray_img, left_line, corners[0], true, left_length); 
        refined_corners[1] = SearchExactEndpoint(gray_img, left_line, corners[1], false, left_length); 
    }

    // 对【右灯条】进行精化流水线处理
    if (GetSubPixelCenterLine(gray_img, corners[3], corners[2], right_line, dynamic_W)) {
        float right_length = (float)cv::norm(corners[2] - corners[3]);
        refined_corners[3] = SearchExactEndpoint(gray_img, right_line, corners[3], true, right_length); 
        refined_corners[2] = SearchExactEndpoint(gray_img, right_line, corners[2], false, right_length); 
    }

    return refined_corners;
}