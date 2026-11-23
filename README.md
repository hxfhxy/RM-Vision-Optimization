# 装甲板位姿优化-程序设计报告

## 一、 整体流程设计
主要包含以下五个处理阶段：

1. **目标初筛**：读取视频帧，利用 YOLO 目标检测网络获取装甲板的初始 2D 包围框（ROI）和四个粗略角点。
2. **亚像素精调**：因为 YOLO 粗角点定位存在波动且远距离误差较大，系统通过局部灰度采样、中心线 Huber 拟合与端点寻优，获取高精度亚像素角点。
3. **初始姿态估计**：结合装甲板 15° 固定后倾角的 3D 模型，将精调后的角点输入 IPPE 算法，获取包含歧义校验的初始 3D 位置与偏航角（Yaw），作为非线性优化的初值。
4. **4-DoF 位姿优化**：将初始位姿送入手写的 LM-BA 优化器，将原本 6 自由度模型降维约束为 4 自由度（X, Y, Z, Yaw），通过迭代微调状态量，使 3D 模型重投影至 2D 图像的残差降至最低。
5. **评估与可视化**：将计算出的位姿数据入库，计算深度标准差、平滑度等指标，并渲染出 X-Z 鸟瞰轨迹图和深度时序图，用于直观评估算法性能。

## 二，函数结构组织
### 1. 前端粗检测
* **主要职责**：负责从原始图像中快速筛选目标，提取装甲板 ROI 与粗 2D 角点。
* **核心函数**：
  * `YoloDetector::detect()`：加载神经网络模型，完成图像推理与 NMS 非极大值抑制。
  * `YoloDetector::extractCoarseCorners()`：解析 Bounding Box，输出 4 个装甲板角点。

### 2. 亚像素角点精炼
* **主要职责**：抛弃全局二值化与纯 PCA，利用物理边缘与鲁棒拟合，将 2D 角点精度提升到亚像素级。
* **核心函数**：
  * `CornerRefiner::refine()`：角点精炼
  * `CornerRefiner::sampleNormalProfiles()`：沿灯条法线建立多截面采样，提取中心梯度点。
  * `CornerRefiner::fitCenterAxisHuber()`：利用 Huber 鲁棒算法拟合亚像素中心轴线（剔除发光晕染与噪点）。
  * `CornerRefiner::searchEndpoints1D()`：将粗角点投影至轴线，沿轴向进行一维梯度搜寻与端界约束，锁定最终高精度亚像素角点。

### 3. 位姿初值解算与消歧
* **主要职责**：求解 3D-2D 对应关系，生成初值，并消除平面 PnP 的镜像二义性。
* **核心函数**：
  * `PnPInitializer::solveIPPE()`：利用 IPPE 算法快速求解平面装甲板模型的候选 3D 姿态。
  * `PnPInitializer::disambiguateDualSolutions()`：双解校验函数：
    * `checkYawTrend()`：对比 2D 灯条倾斜趋势与 3D Yaw 正负号，消除左右镜像歧义；
    * `filterPitchBound()`：硬编码俯仰角阀值；
    * **返回值**：返回姿态送入后端。

### 4. 后端 4-DoF LM-BA 优化器
* **主要职责**：构建非线性最小二乘问题，切断深度-姿态耦合，输出平滑稳定的 3D 位姿。
* **核心函数**：
  * `LMBASolver::solve()`：BA 优化器求解总入口。
  * `LMBASolver::buildResidual()`：构建 2D 亚像素角点与 3D 装甲板模型的重投影残差，硬编码 Pitch/Roll 倾角，降维为 X,Y,Z,Yaw 4 自由度优化。
  * `LMBASolver::computeJacobian()`：手写与解析 4-DoF 残差对状态量的雅可比矩阵 与 Hessian 矩阵。
  * `LMBASolver::updateNielsenDamping()`：根据增益比自适应更新 Nielsen 阻尼因子。

### 5. 主流程调度与后处理层 (Pipeline & Gate Control)
* **主要职责**：串联整体数据流
* **核心函数**：
  * `ArmorPoseEstimator::processFrame()`：主调度入口，依次调用上述 1~4 模块。


## 三、 具体重要算法

1. YOLO目标检测与 ROI 粗选
    输入：原始相机 RGB 图像
    输出：装甲板 2D 目标框（ROI）及 4 个初始粗角点

2. 多行梯度与抛物线拟合的亚像素边缘精炼
    输入：原始灰度图、YOLO 粗角点 
    输出：较精准的角点
  2.1. **法线采样 得到灰度曲线**
     基于初始角点计算灯条几何方向，沿法线方向建立采样区域，在局部灰度图中采集多条平行采样线，并对各采样线灰度进行加权平均，生成一维灰度响应曲线，以抑制高光噪声、色散及随机噪点，为后续边缘检测提供稳定的灰度输入
  2.2. **边缘检测 得到离散边缘**
     基于 2.1 生成的一维灰度响应曲线，以灰度峰值为中心向两侧搜索，通过计算亮度数值来跨越动态阈值的亚像素位置，将边缘精度从整数像素提升成亚像素，从而在不均匀光照下精准锁定装甲板边缘，在阈值穿越的相邻像素间通过线性插值计算亚像素偏移，得到亚像素级的左右边缘坐标，再取两者中点作为该扫描截面的灯条中心
  2.3. **多截面拟合 得到亚像素中心线**
     沿灯条长度方向均匀取多个扫描截面，重复 2.2 的步骤得到一系列中心点点集；采用 Huber 鲁棒拟合对所有中心点进行直线拟合，抑制离群点干扰，最终得到灯条的亚像素中心轴线。
  2.4. **基于中心轴线的端点搜索 得到四个精确顶点**
     基于 2.3 获得的灯条亚像素中心线，将 YOLO 的角点投影到拟合好的中心轴线上，作为纵向搜索的起点。随后沿中心轴线分别向灯条顶部和底部方向采集灰度信号，根据背景与灯条之间的亮度跳变检测真实物理端点，并采用线性插值计算端点的亚像素位置。每条灯条分别执行顶部和底部端点搜索，获得两个高精度端点坐标；左右灯条共得到四个精确角点，按照 TL、BL、BR、TR 的顺序组成装甲板最终角点，为后续位姿解算提供高精度二维观测数据。

3. 4-DoF 的LM-ba位姿优化
   输入：PnP 输出的初始位姿、亚像素角点观测、相机内参矩阵与畸变系数
   输出：3D 位姿 
  3.1. **降维与几何约束**
      将原本 6-DoF 刚体位姿中的俯仰角（Pitch）与翻滚角（Roll）去除，仅保留 xyz与 偏航角（Yaw），构建 4 自由度状态向量；装甲板 15° 后倾角作为固定参数硬编码
  3.2. **手动展开重投影模型**
      先通过偏航角构造旋转矩阵，将 3D 模型点变换至相机坐标系；再展开针孔成像与径向 / 切向畸变计算，最终构建 8 维重投影残差向量（4 个角点 × 2 个坐标）
  3.3. **中心差分数值雅可比计算**
      采用双端中心差分法（对位姿施加微小正负扰动，利用误差变化率来数值近似雅可比矩阵）计算残差对 4 维状态量的雅可比矩阵：对每个状态量分别施加正负微小扰动，通过两次重投影计算差分近似偏导.
  3.4. **IRLS 鲁棒核函数加权**
      引入 Huber 损失函数构建对角权重矩阵，当单个角点的重投影误差小于阈值时保持权重为 1，超过阈值时按比例降低权重，抑制反光、遮挡等离群点对优化结果的拉扯。
  3.5. **Nielsen 阻尼自适应 LM 迭代**
      构建阻尼正规方程组：
      采用 Nielsen 策略自适应更新阻尼因子：
     - 若步长使代价下降：接受更新，按增益比平滑降低阻尼，向牛顿法收敛
     - 若步长使代价上升：拒绝更新，倍增阻尼，退化为梯度下降
     迭代收敛后输出最终位姿与重投影误差。

## 四，数据组织
1. 2D 装甲板观测对象 (`ArmorObject`)
封装从 YOLO 粗检测到亚像素精炼全过程的 2D 图像特征数据。

```cpp
struct ArmorObject {
    int id;                           // 装甲板类别/数字 ID
    int color;                        // 颜色 (RED / BLUE)
    float confidence;                 // YOLO 检测置信度
    
    cv::Rect2f bounding_box;          // 2D 粗检测框
    std::vector<cv::Point2f> coarse_corners;  // 4 个 YOLO 粗像素角点
    std::vector<cv::Point2f> refined_corners; // 4 个 亚像素精炼角点 (0.1px 级)
    
    cv::Point2f center;               // 装甲板 2D 中心点
    float armor_angle_trend;          // 2D 左右灯条倾斜趋势 (用于 IPPE 双解消歧)
};

2. 物理 3D 模型定义 (Armor3DModel)
struct Armor3DModel {
    ArmorType type;                   // 目标类型 (SMALL / LARGE)
    float width;                      // 灯条间距 (小装甲板 135mm / 大装甲板 230mm)
    float height;                     // 灯条高度 (55mm)
    float pitch_angle_offset;         // 物理装甲板固有后倾角 (硬编码 15°)
    
    std::vector<cv::Point3f> points_3d; // 局部坐标系下的 4 个 3D 特征角点坐标
};

3. 位姿状态向量 (Pose3D)

struct Pose3D {
    // 4-DoF 待优化状态量
    double x;                         // X 轴平移 (mm)
    double y;                         // Y 轴平移 (mm)
    double z;                         // Z 轴深度 (mm)
    double yaw;                       // 偏航角 (rad)
    
    // 常数/先验约束 (锁死)
    double pitch = -15.0 * M_PI / 180.0; // 俯仰角 (固定后倾 15°)
    double roll = 0.0;                   // 翻滚角 (硬编码 0)
    
    // 求解质量指标
    double reprojection_error;        // 平均重投影残差 (px)
    bool is_converged;                // LM 算法收敛状态
    Eigen::Matrix4d T_camera_armor;   // 最终 4x4 变换矩阵
};
```


## 五、 遇到的难点及解决思路

1. ：z和yaw互相干扰
      发现这两玩意好像耦合关系比较强，本来是打算优化成distance yaw两个自由度的，但最后搞来稿去还是退回了4自由度
      通过在优化器中锁死 Pitch 和 Roll，降维成4自由度，强迫算法只能在一个水平面上调整偏航角和深度,使得解算出的 Z 轴深度非常稳定地保持在车辆真实的物理尺寸范围内。

2. 如何优化角点精度
      毕竟pca受光线影响比较大，如果单纯用pca肯定是不够的，会有跳变，当灯光闪烁也会导致位置不准
      所以前端优化角点精度部分搞了很多除了pca其他的，亚像素中心轴线拟合 + 端界约束的思路。首先沿法线方向多截面采样，利用 Huber 鲁棒算法拟合出亚像素中心轴线；随后将粗角点垂直投影至该轴线上作为搜索起点，沿轴向进行梯度搜寻。利用线的约束住点的横向位置，消除了曝光变化引发的端点漂移。


## 六，前后精度表现
个人觉得以神经网络为基准来看重投影误差其实作用不是很大吧，毕竟你的yolo本身就不是说很精准，虽然你误差大，但其实你才是对的
我主要就是从稳定性方面来看吧，也就是数据抖动程度，相比与yolo，数据收敛很多，而且主要是在z方面的稳定程度提升很多


![alt text](results/rosbag2_2026_06_27-15_57_58/rosbag2_2026_06_27-15_57_58_depth_comparison.png)
![alt text](results/rosbag2_2026_06_27-15_57_58/rosbag2_2026_06_27-15_57_58_pose_azimuth.png)
![alt text](results/rosbag2_2026_06_27-15_57_58/rosbag2_2026_06_27-15_57_58_pose_x.png)
![alt text](results/rosbag2_2026_06_27-15_57_58/rosbag2_2026_06_27-15_57_58_pose_y.png)
![alt text](results/rosbag2_2026_06_27-15_57_58/rosbag2_2026_06_27-15_57_58_reproj_error.png)
![alt text](results/rosbag2_2026_06_27-15_57_58/rosbag2_2026_06_27-15_57_58_trajectory_xz.png)
![alt text](results/rosbag2_2026_06_27-15_57_58/rosbag2_2026_06_27-15_57_58_timing.png)
静态时，可以看到对比单纯yolo后ba，稳定性大幅度提升，标准差从约 150 mm 压缩至 20 mm，增强了位姿在静止状态下的输出稳定


![alt text](results/rosbag2_2026_06_27-16_00_45/rosbag2_2026_06_27-16_00_45_pose_azimuth.png)
![alt text](results/rosbag2_2026_06_27-16_00_45/rosbag2_2026_06_27-16_00_45_depth_comparison.png)
![alt text](results/rosbag2_2026_06_27-16_00_45/rosbag2_2026_06_27-16_00_45_pose_x.png)
![alt text](results/rosbag2_2026_06_27-16_00_45/rosbag2_2026_06_27-16_00_45_pose_y.png)
![alt text](results/rosbag2_2026_06_27-16_00_45/rosbag2_2026_06_27-16_00_45_reproj_error.png)
![alt text](results/rosbag2_2026_06_27-16_00_45/rosbag2_2026_06_27-16_00_45_trajectory_xz.png)
![alt text](results/rosbag2_2026_06_27-16_00_45/rosbag2_2026_06_27-16_00_45_timing.png)
小陀螺平移：虽然会有偶发单帧位姿跳变，但是下一帧立即重回稳定状态，青色曲线（PCA_BA）在大部分时间里比红色曲线（YOLO_BA）更加平滑、


![alt text](results/rosbag2_2026_06_27-15_59_59/rosbag2_2026_06_27-15_59_59_trajectory_xz.png)
![alt text](results/rosbag2_2026_06_27-15_59_59/rosbag2_2026_06_27-15_59_59_pose_azimuth.png)
![alt text](results/rosbag2_2026_06_27-15_59_59/rosbag2_2026_06_27-15_59_59_pose_x.png)
![alt text](results/rosbag2_2026_06_27-15_59_59/rosbag2_2026_06_27-15_59_59_pose_y.png)
![alt text](results/rosbag2_2026_06_27-15_59_59/rosbag2_2026_06_27-15_59_59_depth_comparison.png)
![alt text](results/rosbag2_2026_06_27-15_59_59/rosbag2_2026_06_27-15_59_59_reproj_error.png)
![alt text](results/rosbag2_2026_06_27-15_59_59/rosbag2_2026_06_27-15_59_59_timing.png)
在左右平移的场景里也可以看出，我这套方案还是比较有效的压制了抖动的情况，更为稳定一点

![alt text](results/rosbag2_2026_06_27-15_54_16/rosbag2_2026_06_27-15_54_16_depth_comparison.png)
![alt text](results/rosbag2_2026_06_27-15_54_16/rosbag2_2026_06_27-15_54_16_pose_azimuth.png)
![alt text](results/rosbag2_2026_06_27-15_54_16/rosbag2_2026_06_27-15_54_16_pose_x.png)
![alt text](results/rosbag2_2026_06_27-15_54_16/rosbag2_2026_06_27-15_54_16_pose_y.png)
![alt text](results/rosbag2_2026_06_27-15_54_16/rosbag2_2026_06_27-15_54_16_reproj_error.png)
![alt text](results/rosbag2_2026_06_27-15_54_16/rosbag2_2026_06_27-15_54_16_trajectory_xz.png)
原地小陀螺部分主要就是重投影误差部分，会有旋转到极限角度出现尖峰，但总体还是抖动减少了很多

![alt text](results/rosbag2_2026_06_27-16_05_23/rosbag2_2026_06_27-16_05_23_depth_comparison.png)
![alt text](results/rosbag2_2026_06_27-16_05_23/rosbag2_2026_06_27-16_05_23_pose_azimuth.png)
![alt text](results/rosbag2_2026_06_27-16_05_23/rosbag2_2026_06_27-16_05_23_pose_x.png)
![alt text](results/rosbag2_2026_06_27-16_05_23/rosbag2_2026_06_27-16_05_23_pose_y.png)
![alt text](results/rosbag2_2026_06_27-16_05_23/rosbag2_2026_06_27-16_05_23_reproj_error.png)
![alt text](results/rosbag2_2026_06_27-16_05_23/rosbag2_2026_06_27-16_05_23_trajectory_xz.png)
在前后平移的时候也是一样，重投影误差抖动减少了很多

## 七，总结（一些个人想法）
总体的简单流程就是yolo，优化角点精度，pnp，ba，但是在这个流程里，优化角点精度是比较主要的，一开始我也只是比较简单的加了个pca，但是pca受光照影响比较大，其实会导致角点不够精准，抖动大，所以不能只是加个pca这么简单，使用pca的主要目的其实是来找中轴线的，所以我就想，那就换个方法找中轴线，所以就增加了上面提到过的自己的一些方法，中轴线通过沿灯条法线方向采集灰度，得到比较准确的左右边缘，中点连成的线就是中轴线，再把yolo角点投影上去进行搜索，所以老实说感觉自己好像不算完全用上了pca，主要还是Huber拟合之类的来精炼角点，最后总体来讲我的流程就差不多这些吧。 

在此之前，ba方面我也尝试了很多不同的ba优化，比如多帧ba和2自由度ba，多帧ba我觉得会有明显的滞后性，和污染性，我使用了多帧ba后其实数据会有明显的滞后，尽管可以调参来减小，但还是会有轻微的滞后，我觉得这是比较致命的，而且数据曲线图来看，并不比单帧ba好，最后还是选择了单帧ba
视频	BA	Multiframe	Temporal
video1	0.156	0.156	0.192
video2	0.122	0.122	4.382
video3	0.217	0.226	1.152
video4	0.109	0.109	17.150
video5	0.101	0.101	11.019
这是当时学习时记录的数据，重投影误差可以看到还是单帧ba感觉最优，不过也可能是我的其他ba没有优化好

然后是降为2自由度的方案，虽然我看主流的方案都是四自由度，但是我当时就想xyz其实是等价换成distance的，能不能来变成2自由度方案，而且在ai的帮助下，发现z和yaw其实有比较强的耦合关系，这样子的话其实就反而会更好进行优化，因为他们的耦合，优化好distance或者yaw，另一个也就会优化好，但最后我还是放弃了2自由度方案，因为这个是四自由度做到一半想到的，不过没那么多时间来研究了，想着还是用回了四自由度，而且在这种rosbag的情况下好像二自由度方案没有四自由度好 
