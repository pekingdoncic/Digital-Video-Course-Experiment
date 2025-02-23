////////#include "stdio.h"
////////#include <iostream>  
////////#include <opencv2/core/core.hpp>  
////////#include <opencv2/highgui/highgui.hpp>  
////////
////////using namespace cv;
////////
////////int main()
////////{
////////    Mat img = imread("E:\\test.jpg");
////////    namedWindow("画面");
////////    imshow("画面", img);
////////    waitKey(6000);
////////}
//////
////////#include <opencv2/opencv.hpp>
////////#include <opencv2/highgui/highgui.hpp>
////////#include <opencv2/imgproc/imgproc.hpp>
////////
////////using namespace cv;
////////
////////int main() {
////////    // 读取视频
////////    VideoCapture cap("E:\\数字视频实验视频\\1实验一视频\\exp1(1).avi");
////////
////////    if (!cap.isOpened()) {
////////        std::cerr << "Error: Unable to open the video file." << std::endl;
////////        return -1;
////////    }
////////
////////    Mat prevFrame, currentFrame, diffFrame, keyFrame, segmentedFrame, markers;
////////
////////    // 读取第一帧作为初始帧
////////    cap >> prevFrame;
////////
////////    // 设定阈值
////////    int thresholdValue = 30;
////////
////////    while (true) {
////////        // 读取当前帧
////////        cap >> currentFrame;
////////
////////        if (currentFrame.empty()) {
////////            std::cerr << "End of video stream." << std::endl;
////////            break;
////////        }
////////
////////        // 计算前后两帧灰度值之差
////////        absdiff(prevFrame, currentFrame, diffFrame);
////////
////////        // 转换为灰度图
////////        cvtColor(diffFrame, diffFrame, COLOR_BGR2GRAY);
////////
////////        // 二值化处理
////////        threshold(diffFrame, keyFrame, thresholdValue, 255, THRESH_BINARY);
////////
////////        // 更新前一帧
////////        prevFrame = currentFrame;
////////
////////        // 使用分水岭算法进行图像分割
////////        Mat dist;
////////        distanceTransform(keyFrame, dist, DIST_L2, 3);
////////        normalize(dist, dist, 0, 1.0, NORM_MINMAX);
////////        threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
////////
////////        Mat kernel = Mat::ones(3, 3, CV_8U);
////////        dilate(dist, dist, kernel);
////////
////////        // 创建标记(markers)图像
////////        markers = Mat::zeros(dist.size(), CV_32S);
////////        markers += 1;  // 所有区域初始化为背景
////////
////////        markers.setTo(2, keyFrame);  // 关键帧区域标记为2
////////
////////        // 应用分水岭算法
////////        watershed(currentFrame, markers);
////////
////////        // 将分割区域着色
////////        Mat segmented;
////////        markers.convertTo(segmented, CV_8U);
////////        bitwise_and(currentFrame, currentFrame, segmented, segmented);
////////
////////        // 显示原视频、关键帧和分割后结果
////////        imshow("Original Video", currentFrame);
////////        imshow("Key Frame", keyFrame);
////////        imshow("Segmented Frame", segmented);
////////
////////        char key = waitKey(30);
////////        if (key == 27)  // 按下ESC键退出循环
////////            break;
////////    }
////////
////////    return 0;
////////}
//////
//////#include <opencv2/opencv.hpp>
//////#include <opencv2/highgui/highgui.hpp>
//////#include <opencv2/imgproc/imgproc.hpp>
//////
//////using namespace cv;
//////
//////int main() {
//////    // 读取视频
//////    VideoCapture cap("E:\\数字视频实验视频\\1实验一视频\\exp1.avi");
//////
//////    if (!cap.isOpened()) {
//////        std::cerr << "Error: Unable to open the video file." << std::endl;
//////        return -1;
//////    }
//////
//////    Mat prevFrame, currentFrame, diffFrame, keyFrame, segmentedFrame, markers;
//////
//////    // 读取第一帧作为初始帧
//////    cap >> prevFrame;
//////
//////    // 设定阈值
//////    int thresholdValue = 30;
//////
//////    while (true) {
//////        // 读取当前帧
//////        cap >> currentFrame;
//////
//////        if (currentFrame.empty()) {
//////            std::cerr << "End of video stream." << std::endl;
//////            break;
//////        }
//////
//////        // 计算前后两帧灰度值之差
//////        absdiff(prevFrame, currentFrame, diffFrame);
//////
//////        // 转换为灰度图
//////        cvtColor(diffFrame, diffFrame, COLOR_BGR2GRAY);
//////
//////        // 二值化处理
//////        threshold(diffFrame, keyFrame, thresholdValue, 255, THRESH_BINARY);
//////
//////        // 更新前一帧
//////        prevFrame = currentFrame;
//////
//////        // 使用分水岭算法进行图像分割
//////        Mat dist;
//////        distanceTransform(keyFrame, dist, DIST_L2, 3);
//////        normalize(dist, dist, 0, 1.0, NORM_MINMAX);
//////        threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
//////
//////        Mat kernel = Mat::ones(3, 3, CV_8U);
//////        dilate(dist, dist, kernel);
//////
//////        // 创建标记(markers)图像
//////        markers = Mat::zeros(dist.size(), CV_32S);
//////        markers += 1;  // 所有区域初始化为背景
//////
//////        markers.setTo(2, keyFrame);  // 关键帧区域标记为2
//////
//////        // 应用分水岭算法
//////        watershed(currentFrame, markers);
//////
//////        // 将分割区域着色
//////        Mat segmented;
//////        markers.convertTo(segmented, CV_8U);
//////        bitwise_and(currentFrame, currentFrame, segmented, segmented);
//////
//////        // 显示原视频、关键帧和分割后结果
//////        imshow("Original Video", currentFrame);
//////        imshow("Key Frame", keyFrame);
//////        imshow("Segmented Frame", segmented);
//////
//////        char key = waitKey(30);
//////        if (key == 27)  // 按下ESC键退出循环
//////            break;
//////    }
//////
//////    return 0;
//////}
////
////#include <opencv2/opencv.hpp>
////#include <opencv2/highgui/highgui.hpp>
////#include <opencv2/imgproc/imgproc.hpp>
////
////using namespace cv;
////
////int main() {
////    // 读取视频
////    VideoCapture cap("E:\\数字视频实验视频\\1实验一视频\\exp1.avi");
////
////    if (!cap.isOpened()) {
////        std::cerr << "Error: Unable to open the video file." << std::endl;
////        return -1;
////    }
////
////    Mat prevFrame, currentFrame, diffFrame, keyFrame, segmentedFrame;
////
////    // 读取第一帧作为初始帧
////    cap >> prevFrame;
////
////    // 设定阈值
////    int thresholdValue = 30;
////
////    while (true) {
////        // 读取当前帧
////        cap >> currentFrame;
////
////        if (currentFrame.empty()) {
////            std::cerr << "End of video stream." << std::endl;
////            break;
////        }
////
////        // 计算前后两帧灰度值之差
////        absdiff(prevFrame, currentFrame, diffFrame);
////
////        // 转换为灰度图
////        cvtColor(diffFrame, diffFrame, COLOR_BGR2GRAY);
////
////        // 二值化处理
////        threshold(diffFrame, keyFrame, thresholdValue, 255, THRESH_BINARY);
////
////        // 更新前一帧
////        prevFrame = currentFrame;
////
////        // 显示原视频、关键帧和分割后结果
////        imshow("Original Video", currentFrame);
////        imshow("Key Frame", keyFrame);
////
////        char key = waitKey(30);
////        if (key == 27)  // 按下ESC键退出循环
////            break;
////    }
////
////    return 0;
////}
////
//
////#include <opencv2/opencv.hpp>
////#include <opencv2/highgui/highgui.hpp>
////#include <opencv2/imgproc/imgproc.hpp>
////
////using namespace cv;
////
////int main() {
////    // 读取视频
////    VideoCapture cap("E:\\数字视频实验视频\\1实验一视频\\exp1.avi");
////
////    if (!cap.isOpened()) {
////        std::cerr << "Error: Unable to open the video file." << std::endl;
////        return -1;
////    }
////
////    Mat prevFrame, currentFrame, diffFrame, keyFrame, segmentedFrame;
////
////    // 读取第一帧作为初始帧
////    cap >> prevFrame;
////
////    // 设定阈值
////    int thresholdValue = 30;
////
////    while (true) {
////        // 读取当前帧
////        cap >> currentFrame;
////
////        if (currentFrame.empty()) {
////            std::cerr << "End of video stream." << std::endl;
////            break;
////        }
////
////        // 计算前后两帧灰度值之差
////        absdiff(prevFrame, currentFrame, diffFrame);
////
////        // 转换为灰度图
////        cvtColor(diffFrame, diffFrame, COLOR_BGR2GRAY);
////
////        // 二值化处理
////        threshold(diffFrame, keyFrame, thresholdValue, 255, THRESH_BINARY);
////
////        // 更新前一帧
////        prevFrame = currentFrame;
////
////        // 显示原视频、关键帧和分割后结果
////        imshow("Original Video", currentFrame);
////        imshow("Key Frame", keyFrame);
////
////        // 阈值分割显示
////        cv::Mat segmented;
////        keyFrame.copyTo(segmented);
////        cv::cvtColor(segmented, segmented, COLOR_GRAY2BGR);
////        std::vector<std::vector<cv::Point>> contours;
////        cv::findContours(keyFrame, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
////        cv::drawContours(segmented, contours, -1, Scalar(0, 255, 0), 2);
////
////        imshow("Segmented Frame", segmented);
////
////        char key = waitKey(30);
////        if (key == 27)  // 按下ESC键退出循环
////            break;
////    }
////
////    return 0;
////}
//
////#include <opencv2/opencv.hpp>
////#include <opencv2/highgui/highgui.hpp>
////#include <opencv2/imgproc/imgproc.hpp>
////
////using namespace cv;
////using namespace std;
////int main() {
////    // 读取视频
////    VideoCapture cap("E:\\数字视频实验视频\\1实验一视频\\exp1.avi");
////
////    if (!cap.isOpened()) {
////        cout << "Error opening video file" << endl;
////        return -1;
////    }
////
////    int frame_num = 0;
////    Mat frame, prev_frame, gray_frame, thresh_frame;
////    while (cap.read(frame)) {
////        // 计算前后两帧灰度值之差  
////        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
////        absdiff(prev_frame, gray_frame, thresh_frame);
////        threshold(thresh_frame, thresh_frame, 25, 255, THRESH_BINARY);
////
////        // 显示原视频、关键帧和分割后结果  
////        imshow("Original", frame);
////        imshow("Key Frame", gray_frame);
////        imshow("Segmented", thresh_frame);
////        imshow("Difference", thresh_frame);
////
////        // 如果当前帧是关键帧，则将其保存为图像文件  
////        if (countNonZero(thresh_frame) > 2000) {
////            char filename[100];
////            sprintf(filename, "key_frame_%03d.png", frame_num++);
////            imwrite(filename, gray_frame);
////        }
////
////        // 将当前帧保存为前一帧，以便下一帧进行比较  
////        prev_frame = gray_frame.clone();
////
////        // 等待按键或 30 毫秒后自动进入下一帧  
////        int key = waitKey(30);
////        if (key == 'q' || key == 27) break;
////    }
////    return 0;
////}
//
//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//int main() {
//    // 读取视频
//    VideoCapture cap("E:\\数字视频实验视频\\1实验一视频\\exp1.avi");
//    if (!cap.isOpened()) {
//        cout << "Error opening video file" << endl;
//        return -1;
//    }
//
//    Mat currentFrame, previousFrame, diffFrame, grayDiff, segmentedFrame;
//
//    int thresholdValue = 30; // 灰度差阈值
//    int keyFrameThreshold = 500; // 关键帧灰度差总阈值
//
//    int frameCount = 0;
//    int keyFrameCount = 0;
//
//    // 视频帧处理循环
//    while (true) {
//        cap >> currentFrame;
//
//        if (currentFrame.empty())
//            break;
//
//        cvtColor(currentFrame, currentFrame, COLOR_BGR2GRAY);
//
//        if (!previousFrame.empty()) {
//            // 计算前后两帧灰度值之差
//            absdiff(previousFrame, currentFrame, diffFrame);
//            threshold(diffFrame, grayDiff, thresholdValue, 255, THRESH_BINARY);
//
//            // 计算灰度差总和
//            int diffSum = sum(grayDiff)[0];
//
//            // 判断是否为关键帧
//            if (diffSum > keyFrameThreshold) {
//                keyFrameCount++;
//
//                // 分割关键帧
//                // 这里可以选择阈值分割算法或分水岭算法
//                // 以下是阈值分割算法的示例
//                threshold(currentFrame, segmentedFrame, 128, 255, THRESH_BINARY);
//
//                // 显示关键帧和分割结果
//                imshow("Key Frame", currentFrame);
//                imshow("Segmented Frame", segmentedFrame);
//            }
//        }
//
//        // 显示原视频
//        imshow("Original Video", currentFrame);
//
//        // 更新前一帧
//        currentFrame.copyTo(previousFrame);
//
//        frameCount++;
//
//        // 按ESC键退出循环
//        if (waitKey(30) == 27)
//            break;
//    }
//
//    cout << "Total frames: " << frameCount << endl;
//    cout << "Key frames: " << keyFrameCount << endl;
//
//    return 0;
//}

//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//
//
//using namespace cv;
//
//int main() {
//    // 打开视频文件
//    VideoCapture cap("E:\\数字视频实验视频\\1实验一视频\\exp1.avi");
//
//    if (!cap.isOpened()) {
//        std::cerr << "Error opening video file." << std::endl;
//        return -1;
//    }
//
//    Mat frame, prevFrame, diff, gray, segmented;
//
//    // 读取第一帧
//    cap >> prevFrame;
//
//    while (cap.read(frame)) {
//        // 转换为灰度图像
//        cvtColor(frame, gray, COLOR_BGR2GRAY);
//        cvtColor(prevFrame, prevFrame, COLOR_BGR2GRAY);
//
//        // 计算前后两帧的灰度差异
//        absdiff(gray, prevFrame, diff);
//
//        // 设置阈值，找到关键帧
//        double thrshold = 30.0;
//        threshold(diff, diff, thrshold, 255, THRESH_BINARY);
//
//        // 在这里可以添加分水岭算法或其他分割算法
//
//        // 显示原视频、关键帧和分割结果
//        imshow("Original Video", frame);
//        imshow("Key Frame", diff);
//        imshow("Segmented Result", segmented);
//
//        // 等待按键，如果按下ESC键则退出循环
//        char key = waitKey(30);
//        if (key == 27)
//            break;
//
//        // 更新前一帧
//        prevFrame = frame.clone();
//    }
//
//    // 释放VideoCapture和关闭窗口
//    cap.release();
//    destroyAllWindows();
//
//    return 0;
//}

//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//
//using namespace cv;
//using namespace std;
//
//int main() {
//    VideoCapture capture("E:\\数字视频实验视频\\1实验一视频\\exp1.avi"); // 替换成你的视频文件路径
//
//    if (!capture.isOpened()) {
//        cout << "Error: Could not open video file." << endl;
//        return -1;
//    }
//
//    Mat frame, prevFrame, diff, keyFrame, segmented;
//
//    double thresholdValue = 50.0; // 可调整的阈值
//    int keyFrameCount = 0;
//
//    while (true) {
//        capture >> frame;
//
//        if (frame.empty()) {
//            break; // 视频结束
//        }
//
//        cvtColor(frame, frame, COLOR_BGR2GRAY);
//
//        if (!prevFrame.empty()) {
//            absdiff(prevFrame, frame, diff);
//            threshold(diff, diff, thresholdValue, 255, THRESH_BINARY);
//
//            if (countNonZero(diff) > 5000) { // 可调整的阈值
//                keyFrame = frame.clone();
//                keyFrameCount++;
//
//                // 阈值分割算法
//                threshold(keyFrame, segmented, 128, 255, THRESH_BINARY);
//
//                // 显示原视频、关键帧和分割后结果
//                imshow("Original Video", frame);
//                imshow("Key Frame", keyFrame);
//                imshow("Segmented Result", segmented);
//
//                int key = waitKey(30); // 设置每帧之间的等待时间，单位为毫秒，控制视频播放速度
//                if (key == 27) // 按下ESC键退出循环
//                    break;
//
//                destroyAllWindows(); // 关闭所有窗口
//            }
//        }
//
//        prevFrame = frame.clone();
//    }
//
//    cout << "Total Key Frames: " << keyFrameCount << endl;
//
//    return 0;
//}

#include <opencv2/opencv.hpp>
#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <opencv2/imgproc/types_c.h>
using namespace std;
using namespace cv;

/**
* @brief 计算直方图，两个峰值，然后波谷阈值法
* @param currentframe 输入图像
*/

void threshBasic(const Mat& currentframe)
{
	// 计算灰度直方图，zero函数用于创建一个大小为256的Mat对象，类型为CV_32F，全部初始化为0
	Mat histogram = Mat::zeros(Size(256, 1), CV_32SC1);
	// 获取行高
	int rows = currentframe.rows;
	int cols = currentframe.cols;
	// 遍历图像，计算灰度直方图
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int index = int(currentframe.at<uchar>(i, j));
			histogram.at<int>(0, index) += 1;
		}
	}
	// 找到灰度直方图最大峰值对应的灰度值
	Point peak1;
	// 在数组中找到全局最小和最大值
	minMaxLoc(histogram, NULL, NULL, NULL, &peak1);
	int p1 = peak1.x;
	Mat gray_2 = Mat::zeros(Size(256, 1), CV_32FC1);
	for (int k = 0; k < 256; k++)
	{
		int hist_k = histogram.at<int>(0, k);
		gray_2.at<float>(0, k) = pow(float(k - p1), 2) * hist_k;
	}
	Point peak2;
	minMaxLoc(gray_2, NULL, NULL, NULL, &peak2);
	int p2 = peak2.x;
	//找到两个峰值之间的最小值对应的灰度值，作为阈值
	Point threshLoc;
	int thresh = 0;
	if (p1 < p2) {
		minMaxLoc(histogram.colRange(p1, p2), NULL, NULL, &threshLoc);
		thresh = p1 + threshLoc.x + 1;
	}
	else {
		minMaxLoc(histogram.colRange(p2, p1), NULL, NULL, &threshLoc);
		thresh = p2 + threshLoc.x + 1;
	}
	/**
	* CV_EXPORTS_W double threshold( InputArray src, OutputArray dst,
								   double thresh, double maxval, int type );
	* src:输入图像
	* dst:输出图像
	* thresh:阈值
	* maxval:输出图像中最大值
	* type:阈值类型，THRESH_BINARY 指二值化
	*/
	Mat threshImage;
	threshold(currentframe, threshImage, thresh, 255, THRESH_BINARY);
	imshow("直方图阈值分割结果", threshImage);
}

/**
* @brief 最大熵法
* @param currentframe 输入图像
* @param threshImage 输出图像
*/

void threshEntroy(Mat currentframe)
{
	int hist_t[256] = { 0 }; //每个像素值个数
	int index = 0;//最大熵对应的灰度
	double Property = 0.0;// 像素占有比率
	double maxEntropy = -1.0;//最大熵
	double frontEntropy = 0.0;//前景熵
	double backEntropy = 0.0;//背景熵
	int sum_t = 0;	//像素总数
	int nCol = currentframe.cols;//每行的像素数
	// 计算每个像素值的个数
	for (int i = 0; i < currentframe.rows; i++)
	{
		uchar* pData = currentframe.ptr<uchar>(i);
		for (int j = 0; j < nCol; ++j)
		{
			++sum_t;
			hist_t[pData[j]] += 1;
		}
	}
	// 计算最大熵
	for (int i = 0; i < 256; i++)
	{
		// 背景像素数
		double sum_back = 0;
		for (int j = 0; j < i; j++)
		{
			sum_back += hist_t[j];
		}

		//背景熵
		for (int j = 0; j < i; j++)
		{
			if (hist_t[j] != 0)
			{
				Property = hist_t[j] / sum_back;
				backEntropy += -Property * logf((float)Property);
			}
		}
		//前景熵
		for (int k = i; k < 256; k++)
		{
			if (hist_t[k] != 0)
			{
				Property = hist_t[k] / (sum_t - sum_back);
				frontEntropy += -Property * logf((float)Property);
			}
		}

		if (frontEntropy + backEntropy > maxEntropy)// 求最大熵
		{
			maxEntropy = frontEntropy + backEntropy;
			index = i;
		}
		frontEntropy = 0.0;	// 前景熵清零
		backEntropy = 0.0;	// 背景熵清零
	}
	Mat threshImage;
	threshold(currentframe, threshImage, index, 255, 0); //进行阈值分割
	imshow("最大熵法", threshImage);
}

/**
* @brief 分水岭算法
* @param currentframe 输入图像
*/


void watershed(Mat& _img)
{
	Mat image = _img;
	Mat grayImage;
	// 转换成灰度图像
	cvtColor(image, grayImage, CV_BGR2GRAY);
	// 高斯滤波和边缘检测
	GaussianBlur(grayImage, grayImage, Size(3, 3), 0, 0);
	Canny(grayImage, grayImage, 50, 150, 3);
	// 轮廓  
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	
	// 找到轮廓
	findContours(grayImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	
	// 生成分水岭图像
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);
	Mat biaoji(image.size(), CV_32S);
	biaoji = Scalar::all(0);
	int index = 0, n1 = 0;
	// 绘制轮廓
	for (; index >= 0; index = hierarchy[index][0], n1++)
	{
		drawContours(biaoji, contours, index, Scalar::all(n1 + 1), 1, 8, hierarchy);	// 绘制轮廓
		drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);	// 绘制轮廓
	}

	Mat biaojiShows;
	convertScaleAbs(biaoji, biaojiShows);
	watershed(image, biaoji);

	Mat w1;	// 分水岭后的图像
	convertScaleAbs(biaoji, w1);	// 转换成8位图像
	imshow("分水岭算法", w1);
	waitKey(1);
}

//void watershedSegmentation(const Mat& inputImage, Mat& markers) {
//	// 步骤1：图像灰度化
//	Mat grayImage;
//	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
//
//	// 步骤2：计算图像梯度
//	Mat gradient;
//	Sobel(grayImage, gradient, CV_32F, 1, 1);
//
//	// 步骤3：排序梯度值
//	Mat absGradient;
//	convertScaleAbs(gradient, absGradient);
//	threshold(absGradient, absGradient, 50, 255, THRESH_BINARY);
//
//	// 步骤4-7：分水岭算法
//	markers = Mat::zeros(gradient.size(), CV_32S);
//	watershed(inputImage, markers);
//
//	// 将标记的区域可视化
//	Mat segmentedImage = Mat::zeros(inputImage.size(), inputImage.type());
//	for (int i = 0; i < markers.rows; ++i) {
//		for (int j = 0; j < markers.cols; ++j) {
//			int index = markers.at<int>(i, j);
//			if (index == -1) {
//				segmentedImage.at<Vec3b>(i, j) = Vec3b(0, 0, 255);  // 设置为红色
//			}
//			else {
//				segmentedImage.at<Vec3b>(i, j) = inputImage.at<Vec3b>(i, j);
//			}
//		}
//	}
//
//	// 显示结果（可选）
//	Mat w1;	// 分水岭后的图像
//	convertScaleAbs(segmentedImage, w1);	// 转换成8位图像
//	imshow("Segmented Image", w1);
//	waitKey(100);  // 等待1毫秒，确保图像显示
//}

//菜单
void menu(Mat currentframe, Mat threshImage, int in,Mat markers) {
	switch (in) {
	case 1: {
		cvtColor(currentframe, currentframe, CV_BGR2GRAY);
		threshBasic(currentframe);
		break;
	}
	case 2: {
		cvtColor(currentframe, currentframe, CV_BGR2GRAY);
		threshEntroy(currentframe);
		break;
	}
	case 3: {
		watershed(currentframe);
		/*watershedSegmentation(currentframe,markers);*/
		break;
	}
	default: {
		cout << "输入错误" << endl;
		break;
	}
	}
}

int main() {
	int in;
	int flag = 1;
	while (flag) {
		cout << "选择值分割算法：" << endl;
		cout << "1.直方图法" << endl;
		cout << "2.最大熵算法" << endl;
		cout << "3.经典分水岭算法" << endl;
		cin >> in;
		flag = 0;
	}
	VideoCapture cap;
	cap.open("E:\\数字视频实验视频\\1实验一视频\\exp1.avi");
	// 打开失败
	if (!cap.isOpened()) {
		cout << "无法读取视频" << endl;
		system("pause");
		return -1;
	}
	int k = -1;

	Mat tempframe, currentframe, keyframe, previousframe, diff;// 当前帧，前一帧，关键帧
	Mat frame;
	int framenum = 0;
	int dur_time = 1;
	int thresholdValue = 50;
	while (true) {
		//显示视频
		Mat frame1;
		// 读取所有帧
		bool ok = cap.read(frame);
		if (!ok) break;
		imshow("视频", frame);
		k = waitKey(50);
		Mat markers;
		cap >> frame1;	//读取视频的每帧
		tempframe = frame1;
		framenum++;
		// 第一帧，如果是第一帧，就将当前帧赋值给前一帧
		if (framenum == 1) {
			keyframe = tempframe;//第一帧作为一个关键帧
			imshow("关键帧", keyframe);//关键帧1
			cvtColor(keyframe, currentframe, CV_BGR2GRAY);//转换为灰度图
			Mat threshImage;
		//	imshow("灰度图", currentframe);//显示灰度图
			menu(keyframe, threshImage, in, markers);
		}
		//第一帧之后的所有帧
		if (framenum >= 2) {
			cvtColor(tempframe, currentframe, CV_BGR2GRAY);//当前帧转灰度图
			cvtColor(keyframe, previousframe, CV_BGR2GRAY);//参考帧转灰度图
			absdiff(currentframe, previousframe, diff);// 两帧差分
			
			float sum = 0.0;
			// 计算两帧差分的累计
			for (int i = 0; i < diff.rows; i++) {
				uchar* data = diff.ptr<uchar>(i);
				for (int j = 0; j < diff.cols; j++)
				{
					sum += data[j];
				}
			}
			if (sum > 7000000) {
				// 阈值
				imshow("关键帧", tempframe);
				keyframe = tempframe;
				Mat threshImage;
				//imshow("灰度图", currentframe);
				menu(keyframe, threshImage, in, markers);
			}
		}
	}
	return 0;
}
//int main() {
//	// 打开视频文件
//	cv::VideoCapture cap("E:\\数字视频实验视频\\1实验一视频\\exp1.avi");
//
//	if (!cap.isOpened()) {
//		std::cerr << "Error opening video file\n";
//		return -1;
//	}
//
//	cv::Mat previous_frame, current_frame, diff;
//
//	// 读取第一帧作为参考帧
//	cap >> previous_frame;
//
//	while (true) {
//		// 读取当前帧
//		cap >> current_frame;
//
//		if (current_frame.empty()) {
//			break;  // 视频结束时退出循环
//		}
//
//		// 将帧转换为灰度图
//		cv::Mat current_gray, previous_gray;
//		cv::cvtColor(current_frame, current_gray, cv::COLOR_BGR2GRAY);
//		cv::cvtColor(previous_frame, previous_gray, cv::COLOR_BGR2GRAY);
//
//		// 计算两帧之间的差异
//		cv::absdiff(current_gray, previous_gray, diff);
//
//		// 设置差异阈值
//		double threshold = 60;
//		cv::threshold(diff, diff, threshold, 255, cv::THRESH_BINARY);
//
//
//		// 显示当前帧和差异图
//		cv::imshow("Current Frame", current_frame);
//		cv::imshow("Difference", diff);
//
//		// 更新参考帧
//		previous_frame = current_frame.clone();
//
//		// 按ESC键退出循环
//		if (cv::waitKey(30) == 27) {
//			break;
//		}
//	}
//
//	// 关闭窗口
//	cv::destroyAllWindows();
//
//	return 0;
//}
//int main() {
//	cv::VideoCapture cap("E:\\数字视频实验视频\\1实验一视频\\exp1.avi");
//
//	if (!cap.isOpened()) {
//		std::cerr << "Error opening video file\n";
//		return -1;
//	}
//
//	cv::Mat previous_frame, current_frame, diff;
//
//	// Read the first frame as a reference frame
//	cap >> previous_frame;
//
//	int frame_count = 0;  // Variable to keep track of frame count
//	while (true) {
//		// Read the current frame
//		cap >> current_frame;
//
//		if (current_frame.empty()) {
//			break;  // Exit the loop when the video ends
//		}
//
//		// Convert frames to grayscale
//		cv::Mat current_gray, previous_gray;
//		cv::cvtColor(current_frame, current_gray, cv::COLOR_BGR2GRAY);
//		cv::cvtColor(previous_frame, previous_gray, cv::COLOR_BGR2GRAY);
//
//		// Compute the absolute difference between frames
//		cv::absdiff(current_gray, previous_gray, diff);
//
//		// Set the difference threshold
//		double threshold = 60;
//		cv::threshold(diff, diff, threshold, 255, cv::THRESH_BINARY);
//
//		// Calculate the percentage of non-zero pixels in the difference image
//		double non_zero_percentage = cv::countNonZero(diff) / (double)(diff.rows * diff.cols);
//
//		// Save the frame as a keyframe if the difference is significant
//		if (non_zero_percentage > 0.001) {  // Adjust this threshold as needed
//			std::string keyframe_filename = "keyframe" + std::to_string(frame_count) + ".png";
//			cv::imwrite(keyframe_filename, current_frame);
//			std::cout << "Keyframe saved: " << keyframe_filename << std::endl;
//
//			// Display the original frame and keyframe side by side
//			cv::Mat combined_frame;
//			cv::hconcat(current_frame, cv::imread(keyframe_filename), combined_frame);
//			cv::imshow("Original vs. Keyframe", combined_frame);
//			cv::waitKey(1);  // Wait until a key is pressed to continue
//		}
//
//		// Show the current frame and the difference image
//		cv::imshow("Current Frame", current_frame);
//		//cv::imshow("Difference", diff);
//
//		// Update the reference frame
//		previous_frame = current_frame.clone();
//
//		// Increment the frame count
//		frame_count++;
//
//		// Press ESC to exit the loop
//		if (cv::waitKey(30) == 27) {
//			break;
//		}
//	}
//
//	// Close windows
//	cv::destroyAllWindows();
//
//	return 0;
//}


