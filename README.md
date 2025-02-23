# Digital-Video-Course-Experiment
这是**北京林业大学数字视频处理**课程的实验相关的源码**（c++实现）**，其中源码部分必做和选做部分均已完成。

### 使用说明

------

本项目是利用了vs2022进行开发，同时需要下载opencv3.4.3。

运行这个项目需要新建一个控制台程序项目。

具体的安装步骤见实验提示中的**VS+opencv环境配置**.pptx。



### 文件夹说明

------

**实验说明**文件夹，内涵三个docx文档中，包含了对于三个实验的实现步骤以及对于源码的详细说明。

**实验所需视频文件夹**，其中包含了测试效果所包含的三个测试视频。

**实验提示文件夹**，包含了老师对实验的说明，以及相关提示文档等。

**源码**文件夹，则包含了三个实验的源码**（c++实现）**

**EnvironmentTest**，之前实验时在本地端部署的项目源码，包含一些项目属性的配置，仅供参考。

### **实验内容**

------

#### **实验一**：关键帧的提取与分割

提取视频关键帧（镜头的第一帧为关键帧），然后分别利用分水岭算法和阈值分割算法对关键帧进行图像分割。 其中，至少任选两种不同的阈值分割算法进行实验。

我实现了如下的算法：

1.  直方图法
2. 最大熵算法
3. 经典分水岭算法
4. 矩不变法阈值分割
5. 迭代法阈值分割
6. 自适应阈值分割

------

#### 实验二：运动目标检测

任选一种方法（背景减法、帧差法等）编程实现视频中运动目标检测，并分析实验结果。

**我实现了如下的算法：**

1. 帧差法
2. 相邻三帧帧差法
3. 背景减法—KNN
4.  背景减法—GMM,混合高斯模型
5. 相邻三帧帧差法和边缘检测相结合

------

#### **实验三：视频增强与播放控制**

编程实现具有视频增强及滤波功能的视频播放器，包括以下功能：

1. 控制视频的播放与暂停
2. 控制视频播放进度
3. 可调节当前视频的对比度与亮度
4. 可对视频进行增强 (如直方图均衡化)
5. 可对视频在HSI模型上通过色度进行色彩增强处理。
6. 可对视频进行滤波(任意实现一种滤波器，多做加分)。

其中有特色的包括了增强视频对比度，调整色度等相关实现。

我实现的**滤波**如下：

1. 均值滤波
2. 中值滤波
3. 高斯滤波
4. 理想低通滤波
5. 高斯低通滤波
6. 巴特沃斯低通滤波
