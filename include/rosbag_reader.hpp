#ifndef ROSBAG_READER_HPP
#define ROSBAG_READER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <functional>

/**
 * @brief Rosbag 数据读取器
 *
 * 从 ROS2 rosbag 文件中读取图像帧和相机内参
 * 支持 sensor_msgs/msg/Image 和 sensor_msgs/msg/CameraInfo
 */
class RosbagReader {
public:
    /**
     * @brief 帧数据结构
     */
    struct Frame {
        cv::Mat image;              // BGR 图像
        double timestamp_sec;       // 时间戳 (秒)
        uint64_t frame_index;       // 帧序号
    };

    /**
     * @brief 相机内参数据
     */
    struct CameraInfo {
        cv::Mat K;                  // 3x3 内参矩阵
        cv::Mat dist;               // 畸变系数
        int width = 0;              // 图像宽度
        int height = 0;             // 图像高度
        bool valid = false;         // 是否已获取有效内参
    };

    /**
     * @brief 构造函数
     * @param bag_path rosbag 文件夹路径 (包含 metadata.yaml)
     */
    explicit RosbagReader(const std::string& bag_path);

    /**
     * @brief 析构函数
     */
    ~RosbagReader();

    /**
     * @brief 打开 rosbag 文件
     * @return 是否成功打开
     */
    bool open();

    /**
     * @brief 关闭 rosbag 文件
     */
    void close();

    /**
     * @brief 读取下一帧
     * @param frame 输出的帧数据
     * @return 是否成功读取 (false 表示已到末尾)
     */
    bool readNextFrame(Frame& frame);

    /**
     * @brief 获取相机内参
     * @note 需要在 readNextFrame 之后调用, 因为内参从 bag 中解析
     */
    CameraInfo getCameraInfo() const;

    /**
     * @brief 获取总帧数 (从 metadata 读取)
     */
    uint64_t getTotalFrames() const;

    /**
     * @brief 获取总时长 (秒)
     */
    double getDuration() const;

    /**
     * @brief 是否已打开
     */
    bool isOpen() const;

private:
    std::string bag_path_;          // rosbag 路径
    CameraInfo camera_info_;        // 相机内参
    bool is_open_ = false;          // 是否已打开
    uint64_t frame_count_ = 0;      // 帧计数
    uint64_t total_image_msgs_ = 0; // 图像消息总数
    double duration_sec_ = 0.0;     // 总时长

    // 使用 pimpl 隐藏 rosbag2 实现细节
    struct Impl;
    Impl* impl_;
};

#endif // ROSBAG_READER_HPP
