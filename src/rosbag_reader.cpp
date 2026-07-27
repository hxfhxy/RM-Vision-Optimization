#include "rosbag_reader.hpp"

// ROS2 rosbag2
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_cpp/converter_options.hpp>

// ROS2 messages
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <iostream>
#include <filesystem>

/**
 * @brief Pimpl 实现: 隐藏 rosbag2 依赖细节
 */
struct RosbagReader::Impl {
    rosbag2_cpp::Reader reader;
    bool camera_info_read = false;
};

/**
 * @brief 构造函数
 */
RosbagReader::RosbagReader(const std::string& bag_path)
    : bag_path_(bag_path), impl_(new Impl()) {}

/**
 * @brief 析构函数
 */
RosbagReader::~RosbagReader() {
    close();
    delete impl_;
}

/**
 * @brief 打开 rosbag 文件
 */
bool RosbagReader::open() {
    try {
        // 检查 metadata.yaml 是否存在
        std::string metadata_path = bag_path_ + "/metadata.yaml";
        if (!std::filesystem::exists(metadata_path)) {
            // 如果传入的是 .db3 文件所在目录, 检查 .db3 是否存在
            // 也支持直接传入包含 metadata.yaml 的目录
            std::cerr << "[RosbagReader] metadata.yaml not found at: " << metadata_path << std::endl;
            return false;
        }

        // 配置存储选项
        rosbag2_storage::StorageOptions storage_options;
        storage_options.uri = bag_path_;
        storage_options.storage_id = "sqlite3";

        // 打开 rosbag
        impl_->reader.open(storage_options, rosbag2_cpp::ConverterOptions{});

        // 获取元数据
        const auto& metadata = impl_->reader.get_metadata();
        total_image_msgs_ = 0;
        for (const auto& topic : metadata.topics_with_message_count) {
            if (topic.topic_metadata.name == "/image_raw") {
                total_image_msgs_ = topic.message_count;
            }
        }

        // 计算时长
        auto duration = metadata.duration;
        duration_sec_ = std::chrono::duration<double>(duration).count();

        is_open_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[RosbagReader] Failed to open: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 关闭 rosbag 文件
 */
void RosbagReader::close() {
    if (is_open_ && impl_) {
        try {
            impl_->reader.close();
        } catch (...) {}
        is_open_ = false;
    }
}

/**
 * @brief 读取下一帧图像
 *
 * 遍历 rosbag 消息, 跳过非图像消息, 解析 Image 消息为 cv::Mat
 * 同时解析第一条 CameraInfo 消息获取相机内参
 */
bool RosbagReader::readNextFrame(Frame& frame) {
    if (!is_open_) return false;

    while (impl_->reader.has_next()) {
        auto bag_message = impl_->reader.read_next();
        const std::string& topic = bag_message->topic_name;

        // 处理 CameraInfo 消息 (只需读取一次)
        if (topic == "/camera_info" && !impl_->camera_info_read) {
            rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
            rclcpp::Serialization<sensor_msgs::msg::CameraInfo> serialization;
            sensor_msgs::msg::CameraInfo camera_info_msg;
            serialization.deserialize_message(&serialized_msg, &camera_info_msg);

            // 提取内参矩阵
            camera_info_.K = cv::Mat::eye(3, 3, CV_64F);
            camera_info_.K.at<double>(0, 0) = camera_info_msg.k[0];  // fx
            camera_info_.K.at<double>(0, 2) = camera_info_msg.k[2];  // cx
            camera_info_.K.at<double>(1, 1) = camera_info_msg.k[4];  // fy
            camera_info_.K.at<double>(1, 2) = camera_info_msg.k[5];  // cy

            // 提取畸变系数
            camera_info_.dist = cv::Mat(1, camera_info_msg.d.size(), CV_64F);
            for (size_t i = 0; i < camera_info_msg.d.size(); ++i) {
                camera_info_.dist.at<double>(0, i) = camera_info_msg.d[i];
            }

            camera_info_.width = camera_info_msg.width;
            camera_info_.height = camera_info_msg.height;
            camera_info_.valid = true;
            impl_->camera_info_read = true;

            std::cout << "[RosbagReader] Camera intrinsics loaded:" << std::endl;
            std::cout << "  fx=" << camera_info_.K.at<double>(0, 0)
                      << " fy=" << camera_info_.K.at<double>(1, 1)
                      << " cx=" << camera_info_.K.at<double>(0, 2)
                      << " cy=" << camera_info_.K.at<double>(1, 2) << std::endl;
            std::cout << "  Resolution: " << camera_info_.width << "x" << camera_info_.height << std::endl;
            std::cout << "  Distortion coeffs: " << camera_info_msg.d.size() << std::endl;

            continue;
        }

        // 处理 Image 消息
        if (topic == "/image_raw") {
            rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
            rclcpp::Serialization<sensor_msgs::msg::Image> serialization;
            sensor_msgs::msg::Image image_msg;
            serialization.deserialize_message(&serialized_msg, &image_msg);

            // 将 ROS Image 转换为 OpenCV Mat
            cv::Mat image;
            const uint8_t* raw_data = image_msg.data.data();
            if (image_msg.encoding == "bgr8") {
                cv::Mat tmp(image_msg.height, image_msg.width, CV_8UC3,
                           const_cast<uint8_t*>(raw_data));
                image = tmp.clone();  // clone 因为 data 可能在下次读取时被覆盖
            } else if (image_msg.encoding == "rgb8") {
                cv::Mat rgb(image_msg.height, image_msg.width, CV_8UC3,
                           const_cast<uint8_t*>(raw_data));
                cv::Mat bgr;
                cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
                image = bgr.clone();
            } else if (image_msg.encoding == "mono8") {
                cv::Mat gray(image_msg.height, image_msg.width, CV_8UC1,
                            const_cast<uint8_t*>(raw_data));
                cv::Mat bgr;
                cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
                image = bgr.clone();
            } else {
                // 尝试通用转换
                std::cerr << "[RosbagReader] Unsupported encoding: " << image_msg.encoding
                          << ", attempting passthrough" << std::endl;
                cv::Mat tmp(image_msg.height, image_msg.width, CV_8UC3,
                           const_cast<uint8_t*>(raw_data));
                image = tmp.clone();
            }

            if (image.empty()) {
                continue;
            }

            // 填充帧数据
            frame.image = image;
            frame.timestamp_sec = bag_message->time_stamp / 1e9;  // 纳秒 → 秒
            frame.frame_index = frame_count_++;

            return true;
        }
    }

    return false;  // 已到末尾
}

/**
 * @brief 获取相机内参
 */
RosbagReader::CameraInfo RosbagReader::getCameraInfo() const {
    return camera_info_;
}

/**
 * @brief 获取总帧数
 */
uint64_t RosbagReader::getTotalFrames() const {
    return total_image_msgs_;
}

/**
 * @brief 获取总时长
 */
double RosbagReader::getDuration() const {
    return duration_sec_;
}

/**
 * @brief 是否已打开
 */
bool RosbagReader::isOpen() const {
    return is_open_;
}
