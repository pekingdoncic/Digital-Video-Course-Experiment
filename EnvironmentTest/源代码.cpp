#include <iostream>  
#include "opencv2/opencv.hpp"  
#include <opencv2\imgproc\types_c.h>

using namespace cv;
using namespace std;

// 帧差法运动目标检测
/*
 @ temp 临时帧
 @ frame 当前帧
 return result 帧差法运动目标检测结果，为每个运动目标加上绿色的矩形框
*/

Mat frameDiff(Mat temp, Mat frame)
{
	Mat result;
	// 将当前帧转换为灰度图像
	cvtColor(frame, frame, COLOR_BGR2GRAY);
	// 将临时帧转换为灰度图像
	cvtColor(temp, temp, COLOR_BGR2GRAY);
	// 将当前帧与临时帧进行帧差运算
	absdiff(frame, temp, result);
	// 将帧差运算结果二值化
	threshold(result, result, 50, 255, THRESH_BINARY);
	// 对二值化结果进行腐蚀膨胀运算
	//erode(result, result, Mat());
	//dilate(result, result, Mat());

	// 对二值化结果进行形态学开运算
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(result, result, MORPH_OPEN, kernel);
	// 对二值化结果进行形态学闭运算
	morphologyEx(result, result, MORPH_CLOSE, kernel);
	imshow("biyunsuan", result);
	// 查找二值化结果中的轮廓
	vector<vector<Point>> contours;
	findContours(result, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// 遍历轮廓
	for (size_t t = 0; t < contours.size(); t++)
	{
		// 绘制轮廓
		drawContours(frame, contours, t, Scalar(0, 0, 255), 2);
		
		
		// 获取当前轮廓的矩形边界框
		Rect rect = boundingRect(contours[t]);
		// 如果在图形的上半部分，忽略小于  的框
		//if (rect.y < frame.rows / 4 && rect.area() < 500)
		//	continue;
		//// 如果在图形的下半部分，忽略小于  的框
		//if (rect.y > frame.rows / 4 && rect.area() < 900)
		//	continue;
		if (rect.area() < 400)
				continue;
		// 在当前帧中绘制矩形边界框
		rectangle(frame, rect, Scalar(0, 255, 0), 2, 8, 0);
	}
	return frame;
}

// 三帧差法

Mat frameDiff3(Mat temp, Mat frame)
{
	Mat result;
	Mat gray1, gray2, gray3;
	Mat diff1, diff2, diff3;
	Mat diff;
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int thresh = 100;
	int max_thresh = 255;
	RNG rng(12345);
	// 将当前帧转换为灰度图像
	cvtColor(temp, gray1, CV_BGR2GRAY);
	cvtColor(frame, gray2, CV_BGR2GRAY);
	absdiff(gray1, gray2, diff1);	// 第一帧与第二帧的差
	cvtColor(frame, gray3, CV_BGR2GRAY);
	absdiff(gray2, gray3, diff2);	// 第二帧与第三帧的差
	absdiff(diff1, diff2, diff3);	// 两帧差的差
	threshold(diff3, threshold_output, thresh, 255, THRESH_BINARY);
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	imshow("threshold_output", threshold_output);
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	// 遍历轮廓
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);	// 多边形逼近
		boundRect[i] = boundingRect(Mat(contours_poly[i]));	// 获取当前轮廓的矩形边界框
	}
	result = frame.clone();
	for (int i = 0; i < contours.size(); i++)
	{
		// 绘制轮廓-红褐色
		drawContours(result, contours_poly, i, Scalar(0, 0, 255), 1, 8, vector<Vec4i>(), 0, Point());

		//如果在图形的上1/3部分，忽略面积小于 400 的框
		if (boundRect[i].y < frame.rows / 4 && boundRect[i].area() < 200)
			continue;
		//如果在图形的下2/3部分，忽略面积小于 900 的框
		if (boundRect[i].y > frame.rows / 4 && boundRect[i].area() < 600)
			continue;	
		//if ( boundRect[i].area() < 200)
		//	continue;
		
		rectangle(result, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0);
	}
	return result;
}

// 边缘检测法 运动目标检测函数,原理是利用边缘检测算法，检测出图像中的运动目标
/*
 @ temp 临时帧
 @ frame 当前帧
 return result 检测结果，为每个运动目标加上绿色的矩形框,使用 canny 算法
*/

Mat MotionDetectByCanny(Mat temp, Mat frame)
{
	Mat result;
	frame.copyTo(result);
	// 1. 将当前帧和临时帧转换为灰度图
	Mat gray1, gray2;
	cvtColor(temp, gray1, COLOR_BGR2GRAY);
	cvtColor(frame, gray2, COLOR_BGR2GRAY);
	// 2. 对当前帧和临时帧进行高斯滤波
	GaussianBlur(gray1, gray1, Size(3, 3), 0, 0);
	GaussianBlur(gray2, gray2, Size(3, 3), 0, 0);
	// 3. 对当前帧和临时帧进行运动目标检测
	Mat motion = gray1 - gray2;
	// 4. 对运动目标进行二值化处理
	threshold(motion, motion, 50, 255, THRESH_BINARY);
	// 5. 对二值化图像进行形态学操作
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(motion, motion, MORPH_OPEN, element);
	// 6. 对二值化图像进行边缘检测
	// 后面两个参数分别代表 threshold1 和 threshold2 的值以及 apertureSize
	Canny(motion, motion, 3, 9, 3);
	imshow("motion", motion);
	// 7. 查找轮廓
	vector<vector<Point>> contours;
	findContours(motion, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	// 8. 绘制轮廓
	for (size_t t = 0; t < contours.size(); t++)
	{
		// 绘制轮廓-红褐色
		drawContours(result, contours, t, Scalar(0, 0, 255), 2);
		//如果在图形的上1/3部分，忽略面积小于 400 的框
		Rect rect = boundingRect(contours[t]);
		if (rect.y < frame.rows / 4 && rect.area() < 200)
			continue;
		// 如果在图形的下2/3部分，忽略面积小于 1000 的框
		if (rect.y > frame.rows * 1 / 4 && rect.area() < 900)
			continue;
		
		rectangle(result, rect, Scalar(0, 255, 0), 2, 8, 0);

	}
	return result;
}

// 使用 BackgroundSubtractorMOG() 背景减法实现运动目标检测
/*
 @ temp 临时帧
 @ frame 当前帧
 return result 检测结果，为每个运动目标加上绿色的矩形框,使用 BackgroundSubtractorMOG 函数
*/

Mat detectMotion(Mat frame) {
	Mat result;
	frame.copyTo(result);
	//创建背景减法器
	Ptr<BackgroundSubtractorMOG2> bgModel = createBackgroundSubtractorMOG2();
	//背景减法
	Mat fgMask;
	bgModel->apply(frame, fgMask);
	//膨胀
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(fgMask, fgMask, kernel);
	imshow("fgmask", fgMask);
	//查找轮廓
	vector<vector<Point>> contours;
	findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//绘制边缘
	for (size_t t = 0; t < contours.size(); t++)
	{
		// 绘制轮廓-红褐色
		drawContours(result, contours, t, Scalar(0, 0, 255), 2);
		//如果在图形的上1/3部分，忽略面积小于 400 的框
		Rect rect = boundingRect(contours[t]);
		if (rect.y < frame.rows / 4 && rect.area() < 200)
			continue;
		// 如果在图形的下2/3部分，忽略面积小于 1000 的框
		if (rect.y > frame.rows * 1 / 4 && rect.area() < 900)
			continue;

		rectangle(result, rect, Scalar(0, 255, 0), 2, 8, 0);

	}
	return result;
}



int main()
{
	VideoCapture video("F:\\学习\\数字视频处理\\EnvironmentTest\\Videos\\exp2.avi");
	if (!video.isOpened())
	{
		cout << "video open error!" << endl;
		return 0;
	}
	int frameCount = video.get(CAP_PROP_FRAME_COUNT); 	// 获取帧数
	double FPS = video.get(CAP_PROP_FPS); 	// 获取 FPS
	Mat frame;
	Mat temp;// 存储前一帧图像
	Mat fst;
	Mat result;// 存储结果图像
	for (int i = 0; i < frameCount; i++)
	{
		video >> frame;
		imshow("frame", frame);
		if (frame.empty())
		{
			cout << "frame is empty!" << endl;
			break;
		}
		
		int framePosition = video.get(CAP_PROP_POS_FRAMES);
		if (i == 0)  
		{
			// 记录第一帧
			fst = frame.clone();

			// 帧差法
			//result = frameDiff(frame, frame); 
			
			// 三帧差法
			// result = frameDiff3(frame, frame); 
			
			// 边缘检测法
			result = MotionDetectByCanny(frame, frame);
		}
		else
		{
			// 帧差法
			//result  = frameDiff(temp, frame);   
			
			// 三帧差法
			// result  = frameDiff3(temp, frame);   

			// 边缘检测法
			result = MotionDetectByCanny(temp, frame);
		}
		imshow("result", result);
		if (waitKey(1000.0 / FPS) == 27)
		{
			cout << "ESC退出!" << endl;
			break;
		}
		temp = frame.clone();
	}
	return 0;
}