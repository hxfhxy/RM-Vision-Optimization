/**
 * @file main.cpp
 * @brief 装甲板位姿优化评估程序 (双流水线 A/B 测试版)
 *
 * 强制并行执行并对比两条流水线:
 * 1. YOLO -> PnP -> BA (无 PCA, 红色曲线)
 * 2. YOLO -> PCA -> PnP -> BA (有 PCA, 青色曲线)
 */

#include "armor_types.hpp"
#include "armor_yolo_standalone.hpp"
#include "pnp_solver_standalone.hpp"
#include "ba_optimizer.hpp"
#include "corner_refiner.hpp"
#include "rosbag_reader.hpp"
#include "evaluator.hpp"
#include "visualizer.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <chrono>

/**
 * @brief 命令行参数结构
 */
struct CommandLineArgs {
    std::string model_path;             // ONNX 模型路径
    std::vector<std::string> videos;    // 视频 rosbag 目录列表
    std::string output_dir = "results"; // 输出目录
    int detect_color = 1;               // 检测颜色 (0=蓝, 1=红)
    bool save_video = true;             // 是否保存可视化视频
    int pca_search_radius = 8;          // PCA 搜索半径
    double pca_max_shift = 0.3;         // PCA 最大偏移
    int max_frames = 0;                 // 最大处理帧数 (0=全部)
};

/**
 * @brief 解析命令行参数
 */
CommandLineArgs parseArgs(int argc, char* argv[]) {
    CommandLineArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--video" && i + 1 < argc) {
            args.videos.push_back(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_dir = argv[++i];
        } else if (arg == "--color" && i + 1 < argc) {
            args.detect_color = std::stoi(argv[++i]);
        } else if (arg == "--no-video") {
            args.save_video = false;
        } else if (arg == "--pca-sr" && i + 1 < argc) {
            args.pca_search_radius = std::stoi(argv[++i]);
        } else if (arg == "--pca-ms" && i + 1 < argc) {
            args.pca_max_shift = std::stod(argv[++i]);
        } else if (arg == "--max-frames" && i + 1 < argc) {
            args.max_frames = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "装甲板位姿优化评估程序 (双流水线 A/B 测试版)" << std::endl;
            std::cout << "用法: ./armor_pose_opt --model <model.onnx> --video <bag_dir> [options]" << std::endl;
            std::cout << "选项:" << std::endl;
            std::cout << "  --model <path>     ONNX 模型路径" << std::endl;
            std::cout << "  --video <dir>      rosbag 目录 (可多次指定)" << std::endl;
            std::cout << "  --output <dir>     输出目录 (默认: results)" << std::endl;
            std::cout << "  --color <0|1>      检测颜色: 0=蓝, 1=红 (默认: 0)" << std::endl;
            std::cout << "  --no-video         不保存可视化视频" << std::endl;
            std::cout << "  --max-frames <n>   最大处理帧数 (0=全部)" << std::endl;
            exit(0);
        }
    }
    return args;
}

/**
 * @brief 获取视频名称 (从路径中提取目录名)
 */
std::string getVideoName(const std::string& path) {
    std::filesystem::path p(path);
    return p.filename().string();
}

/**
 * @brief 计算角点的重投影位置
 */
std::vector<cv::Point2f> computeReprojectedCorners(
    const PoseResult& pose,
    const std::vector<cv::Point3f>& model_points,
    const cv::Mat& K,
    const cv::Mat& dist) {

    std::vector<cv::Point2f> projected;
    cv::projectPoints(model_points, pose.rvec, pose.tvec, K, dist, projected);
    return projected;
}


/**
 * @brief 处理单个视频
 */
bool processVideo(const CommandLineArgs& args, const std::string& video_path,
                  ArmorYoloStandalone& yolo, const cv::Mat& K, const cv::Mat& dist) {

    std::string video_name = getVideoName(video_path);
    std::string output_dir = args.output_dir + "/" + video_name;

    std::cout << "\n========================================" << std::endl;
    std::cout << "处理视频: " << video_name << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. 打开 rosbag
    RosbagReader reader(video_path);
    if (!reader.open()) {
        std::cerr << "无法打开 rosbag: " << video_path << std::endl;
        return false;
    }

    std::cout << "总帧数: " << reader.getTotalFrames()
              << ", 时长: " << std::fixed << std::setprecision(1) << reader.getDuration() << "s" << std::endl;

    // 2. 从 rosbag 读取相机内参
    cv::Mat cam_K, cam_dist;
    RosbagReader::Frame first_frame;
    if (reader.readNextFrame(first_frame)) {
        auto cam_info = reader.getCameraInfo();
        if (cam_info.valid) {
            cam_K = cam_info.K.clone();
            cam_dist = cam_info.dist.clone();
        } else {
            cam_K = K.clone();
            cam_dist = dist.clone();
            std::cout << "[Warning] No camera info in rosbag, using default intrinsics" << std::endl;
        }
    } else {
        cam_K = K.clone();
        cam_dist = dist.clone();
    }

    // 3. 初始化优化器 (为防止时序状态交叉污染，必须独立实例化)
    ArmorBAOptimizer ba_optimizer_yolo(cam_K, cam_dist, 20, 1e-6);
    ArmorBAOptimizer ba_optimizer_pca(cam_K, cam_dist, 20, 1e-6);

    // 4. 初始化评估器和可视化器
    Evaluator evaluator;
    Visualizer visualizer(output_dir, video_name);

    // 5. 逐帧处理
    RosbagReader::Frame frame;
    uint64_t frame_count = 0;
    uint64_t detected_count = 0;

    // 处理第一帧
    bool has_first_frame = true;

    while (true) {
        if (has_first_frame) {
            frame = first_frame;
            has_first_frame = false;
        } else {
            if (!reader.readNextFrame(frame)) break;
        }
        if (args.max_frames > 0 && frame_count >= static_cast<uint64_t>(args.max_frames)) {
            break;
        }

        // YOLO 检测
        auto armors = yolo.detect(frame.image);
        if (armors.empty()) {
            frame_count++;
            continue;
        }

        // 选取最优装甲板
        cv::Point2f imgCenter(frame.image.cols / 2.0f, frame.image.rows / 2.0f);
        std::sort(armors.begin(), armors.end(),
            [&imgCenter](const DetectedArmor& a, const DetectedArmor& b) {
                if (std::abs(a.confidence - b.confidence) > 0.05)
                    return a.confidence > b.confidence;
                return a.dist_to_center < b.dist_to_center;
            });

        auto& best_armor = armors[0];

        // 保存 YOLO 原始角点
        std::vector<cv::Point2f> yolo_corners = best_armor.pnp_corners;

        // 获取 3D 模型点
        bool is_small = ArmorModel::isSmallArmor(yolo_corners);
        double half_w = is_small ? (135.0 / 2.0) : (230.0 / 2.0);
        double half_h = 55.0 / 2.0;
        
        // 保持与 PnP 严格一致的顺序 [左上, 左下, 右下, 右上] 和 15° 倾角
        double tilt = 15.0 * CV_PI / 180.0;
        double z_top = half_h * std::sin(tilt);
        double z_bottom = -half_h * std::sin(tilt);
        double y_top = -half_h * std::cos(tilt);
        double y_bottom = half_h * std::cos(tilt);

        std::vector<cv::Point3f> model_points = {
            cv::Point3f(-half_w, y_top, z_top),       // 0: 左上 (TL)
            cv::Point3f(-half_w, y_bottom, z_bottom), // 1: 左下 (BL)
            cv::Point3f( half_w, y_bottom, z_bottom), // 2: 右下 (BR)
            cv::Point3f( half_w, y_top, z_top)        // 3: 右上 (TR)
        };

        // ==========================================
        // 流程 1: YOLO -> PnP -> BA (无 PCA)
        // ==========================================
        DetectedArmor armor_no_pca = best_armor;
        armor_no_pca.pnp_corners = yolo_corners;
        
        PnpSolverStandalone::solve(armor_no_pca, cam_K, cam_dist);
        
        // 【拦截机制】如果 PnP 失败，直接跳过本帧剩余的所有评估和渲染
        if (armor_no_pca.tvec.empty() || armor_no_pca.tvec.at<double>(2) <= 0) {
            std::cerr << "[Warning] Frame " << frame_count << ": YOLO PnP failed!" << std::endl;
            frame_count++;
            continue; 
        }

        PoseResult ba_result = ba_optimizer_yolo.optimize(armor_no_pca.tvec, armor_no_pca.rvec, yolo_corners, model_points);

        Evaluator::FrameMetrics ba_metrics;
        ba_metrics.frame_index = frame_count;
        ba_metrics.reprojection_error = ba_result.reprojection_error;
        ba_metrics.x = ba_result.x; ba_metrics.y = ba_result.y; ba_metrics.z = ba_result.z;
        ba_metrics.yaw = ba_result.yaw; ba_metrics.distance = ba_result.distance;
        ba_metrics.compute_time_ms = ba_result.compute_time_ms;
        evaluator.addFrame("YOLO_BA", ba_metrics);


        // ==========================================
        // 流程 2: YOLO -> PCA -> PnP -> BA
        // ==========================================
        cv::Mat gray;
        if (frame.image.channels() == 3) cv::cvtColor(frame.image, gray, cv::COLOR_BGR2GRAY);
        else gray = frame.image;

        std::vector<cv::Point2f> pca_corners = CornerRefiner::refine(gray, yolo_corners, args.pca_search_radius, 30.0f);
        double refine_shift = 0.0;
        for (int i = 0; i < 4; ++i) refine_shift += cv::norm(pca_corners[i] - yolo_corners[i]);
        refine_shift /= 4.0;
        
        DetectedArmor armor_pca = best_armor;
        armor_pca.pnp_corners = pca_corners;
        
        PnpSolverStandalone::solve(armor_pca, cam_K, cam_dist);
        
        // 【拦截机制】双重保护
        if (armor_pca.tvec.empty() || armor_pca.tvec.at<double>(2) <= 0) {
            std::cerr << "[Warning] Frame " << frame_count << ": PCA PnP failed!" << std::endl;
            frame_count++;
            continue;
        }

        PoseResult pca_ba_result = ba_optimizer_pca.optimize(armor_pca.tvec, armor_pca.rvec, pca_corners, model_points);

        Evaluator::FrameMetrics pca_ba_metrics;
        pca_ba_metrics.frame_index = frame_count;
        pca_ba_metrics.reprojection_error = pca_ba_result.reprojection_error;
        pca_ba_metrics.x = pca_ba_result.x; pca_ba_metrics.y = pca_ba_result.y; pca_ba_metrics.z = pca_ba_result.z;
        pca_ba_metrics.yaw = pca_ba_result.yaw; pca_ba_metrics.distance = pca_ba_result.distance;
        pca_ba_metrics.compute_time_ms = pca_ba_result.compute_time_ms;
        evaluator.addFrame("PCA_BA", pca_ba_metrics);

        // ==========================================
        // 可视化
        // ==========================================
        auto ba_reproj = computeReprojectedCorners(ba_result, model_points, cam_K, cam_dist);
        auto pca_ba_reproj = computeReprojectedCorners(pca_ba_result, model_points, cam_K, cam_dist);

        const cv::Scalar C_RED(0, 0, 255), C_CYAN(255, 255, 0), C_YELLOW(0, 255, 255), C_GREEN(0, 255, 0), C_WHITE(255, 255, 255);

        cv::Mat vis_image = frame.image.clone();

        // 绘制输入角点
        for (int i = 0; i < 4; ++i) {
            cv::drawMarker(vis_image, yolo_corners[i], C_YELLOW, cv::MARKER_CROSS, 6, 1);
            cv::drawMarker(vis_image, pca_corners[i], C_GREEN, cv::MARKER_CROSS, 6, 1);
        }

        // 绘制 YOLO_BA 重投影 (红色框)
        for (int i = 0; i < 4; ++i) {
            cv::circle(vis_image, ba_reproj[i], 3, C_RED, 1);
            cv::line(vis_image, ba_reproj[i], ba_reproj[(i+1)%4], C_RED, 1, cv::LINE_AA);
        }

        // 绘制 PCA_BA 重投影 (青色框)
        for (int i = 0; i < 4; ++i) {
            cv::circle(vis_image, pca_ba_reproj[i], 3, C_CYAN, 1);
            cv::line(vis_image, pca_ba_reproj[i], pca_ba_reproj[(i+1)%4], C_CYAN, 1, cv::LINE_AA);
        }

        // 图例与信息
        int ly = 30;
        cv::putText(vis_image, cv::format("Frame: %d", (int)frame_count),
                   cv::Point(10, ly), cv::FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 2); ly += 30;
        cv::putText(vis_image, "Yellow=YOLO Green=PCA Red=YOLO_BA Cyan=PCA_BA",
                   cv::Point(10, ly), cv::FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1); ly += 25;

        int info_y = vis_image.rows - 100;
        cv::putText(vis_image, cv::format("YOLO_BA Error: %.3f px", ba_result.reprojection_error),
                   cv::Point(10, info_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, C_RED, 1);
        cv::putText(vis_image, cv::format("PCA_BA Error:  %.3f px", pca_ba_result.reprojection_error),
                   cv::Point(10, info_y + 22), cv::FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1);
        cv::putText(vis_image, cv::format("Z Depth: YOLO_BA=%.0f mm  PCA_BA=%.0f mm", ba_result.z, pca_ba_result.z),
                   cv::Point(10, info_y + 44), cv::FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1);
        cv::putText(vis_image, cv::format("PCA shift: %.3f px", refine_shift),
                   cv::Point(10, info_y + 66), cv::FONT_HERSHEY_SIMPLEX, 0.45, C_GREEN, 1);

        if (args.save_video) {
            visualizer.writeVideoFrame(vis_image);
        }

        // 终端进度打印
        if (frame_count % 50 == 0) {
            std::cout << "  帧 " << frame_count
                      << " | YOLO_BA Z: " << std::fixed << std::setprecision(0) << ba_result.z << "mm"
                      << " | PCA_BA Z: " << pca_ba_result.z << "mm"
                      << std::endl;
        }
        detected_count++;// 仅在成功检测到装甲板时计数
        frame_count++;// 计数所有帧
    }

    reader.close();
    visualizer.releaseVideo();

    // 6. 生成评估报告
    evaluator.printSummary(video_name);
    evaluator.writeSummary(output_dir + "/" + video_name + "_summary.txt", video_name);
    evaluator.writeToCSV(output_dir + "/" + video_name + "_metrics.csv");
    visualizer.generateAllPlots(evaluator);

    std::cout << "处理完成: " << frame_count << " 帧, 检测到 " << detected_count << " 帧" << std::endl;
    std::cout << "输出目录: " << output_dir << std::endl;

    return true;
}

/**
 * @brief 主函数
 */
int main(int argc, char* argv[]) {
    CommandLineArgs args = parseArgs(argc, argv);

    if (args.model_path.empty()) {
        std::cerr << "错误: 必须指定 ONNX 模型路径 (--model)" << std::endl;
        return 1;
    }

    if (args.videos.empty()) {
        std::string video_dir = "video";
        if (std::filesystem::exists(video_dir)) {
            for (const auto& entry : std::filesystem::directory_iterator(video_dir)) {
                if (entry.is_directory()) {
                    std::string metadata = entry.path().string() + "/metadata.yaml";
                    if (std::filesystem::exists(metadata)) {
                        args.videos.push_back(entry.path().string());
                    }
                }
            }
            std::sort(args.videos.begin(), args.videos.end());
        }
        if (args.videos.empty()) {
            std::cerr << "错误: 未找到视频目录, 请使用 --video 指定" << std::endl;
            return 1;
        }
    }

    std::cout << "============================================" << std::endl;
    std::cout << "  装甲板位姿优化评估程序 (双流水线 A/B 测试版)" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "模型: " << args.model_path << std::endl;
    std::cout << "视频数: " << args.videos.size() << std::endl;
    std::cout << "输出目录: " << args.output_dir << std::endl;
    std::cout << "检测颜色: " << (args.detect_color == 0 ? "蓝色" : "红色") << std::endl;

    std::filesystem::create_directories(args.output_dir);

    ArmorYoloStandalone yolo;
    yolo.setDetectColor(args.detect_color);
    if (!yolo.loadModel(args.model_path)) {
        std::cerr << "无法加载模型: " << args.model_path << std::endl;
        return 1;
    }

    cv::Mat default_K = (cv::Mat_<double>(3, 3) <<
        1750.34699, 0.0, 733.30737,
        0.0, 1749.40469, 549.1027,
        0.0, 0.0, 1.0);
    cv::Mat default_dist = (cv::Mat_<double>(1, 5) <<
        -0.056721, 0.044929, -0.000139, -0.000374, 0.0);

    int success_count = 0;
    for (const auto& video_path : args.videos) {
        if (processVideo(args, video_path, yolo, default_K, default_dist)) {
            success_count++;
        }
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "  全部完成! 成功处理 " << success_count << "/" << args.videos.size() << " 个视频" << std::endl;
    std::cout << "  结果保存在: " << args.output_dir << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}