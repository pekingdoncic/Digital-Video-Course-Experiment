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
////////    namedWindow("����");
////////    imshow("����", img);
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
////////    // ��ȡ��Ƶ
////////    VideoCapture cap("E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1(1).avi");
////////
////////    if (!cap.isOpened()) {
////////        std::cerr << "Error: Unable to open the video file." << std::endl;
////////        return -1;
////////    }
////////
////////    Mat prevFrame, currentFrame, diffFrame, keyFrame, segmentedFrame, markers;
////////
////////    // ��ȡ��һ֡��Ϊ��ʼ֡
////////    cap >> prevFrame;
////////
////////    // �趨��ֵ
////////    int thresholdValue = 30;
////////
////////    while (true) {
////////        // ��ȡ��ǰ֡
////////        cap >> currentFrame;
////////
////////        if (currentFrame.empty()) {
////////            std::cerr << "End of video stream." << std::endl;
////////            break;
////////        }
////////
////////        // ����ǰ����֡�Ҷ�ֵ֮��
////////        absdiff(prevFrame, currentFrame, diffFrame);
////////
////////        // ת��Ϊ�Ҷ�ͼ
////////        cvtColor(diffFrame, diffFrame, COLOR_BGR2GRAY);
////////
////////        // ��ֵ������
////////        threshold(diffFrame, keyFrame, thresholdValue, 255, THRESH_BINARY);
////////
////////        // ����ǰһ֡
////////        prevFrame = currentFrame;
////////
////////        // ʹ�÷�ˮ���㷨����ͼ��ָ�
////////        Mat dist;
////////        distanceTransform(keyFrame, dist, DIST_L2, 3);
////////        normalize(dist, dist, 0, 1.0, NORM_MINMAX);
////////        threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
////////
////////        Mat kernel = Mat::ones(3, 3, CV_8U);
////////        dilate(dist, dist, kernel);
////////
////////        // �������(markers)ͼ��
////////        markers = Mat::zeros(dist.size(), CV_32S);
////////        markers += 1;  // ���������ʼ��Ϊ����
////////
////////        markers.setTo(2, keyFrame);  // �ؼ�֡������Ϊ2
////////
////////        // Ӧ�÷�ˮ���㷨
////////        watershed(currentFrame, markers);
////////
////////        // ���ָ�������ɫ
////////        Mat segmented;
////////        markers.convertTo(segmented, CV_8U);
////////        bitwise_and(currentFrame, currentFrame, segmented, segmented);
////////
////////        // ��ʾԭ��Ƶ���ؼ�֡�ͷָ����
////////        imshow("Original Video", currentFrame);
////////        imshow("Key Frame", keyFrame);
////////        imshow("Segmented Frame", segmented);
////////
////////        char key = waitKey(30);
////////        if (key == 27)  // ����ESC���˳�ѭ��
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
//////    // ��ȡ��Ƶ
//////    VideoCapture cap("E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1.avi");
//////
//////    if (!cap.isOpened()) {
//////        std::cerr << "Error: Unable to open the video file." << std::endl;
//////        return -1;
//////    }
//////
//////    Mat prevFrame, currentFrame, diffFrame, keyFrame, segmentedFrame, markers;
//////
//////    // ��ȡ��һ֡��Ϊ��ʼ֡
//////    cap >> prevFrame;
//////
//////    // �趨��ֵ
//////    int thresholdValue = 30;
//////
//////    while (true) {
//////        // ��ȡ��ǰ֡
//////        cap >> currentFrame;
//////
//////        if (currentFrame.empty()) {
//////            std::cerr << "End of video stream." << std::endl;
//////            break;
//////        }
//////
//////        // ����ǰ����֡�Ҷ�ֵ֮��
//////        absdiff(prevFrame, currentFrame, diffFrame);
//////
//////        // ת��Ϊ�Ҷ�ͼ
//////        cvtColor(diffFrame, diffFrame, COLOR_BGR2GRAY);
//////
//////        // ��ֵ������
//////        threshold(diffFrame, keyFrame, thresholdValue, 255, THRESH_BINARY);
//////
//////        // ����ǰһ֡
//////        prevFrame = currentFrame;
//////
//////        // ʹ�÷�ˮ���㷨����ͼ��ָ�
//////        Mat dist;
//////        distanceTransform(keyFrame, dist, DIST_L2, 3);
//////        normalize(dist, dist, 0, 1.0, NORM_MINMAX);
//////        threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
//////
//////        Mat kernel = Mat::ones(3, 3, CV_8U);
//////        dilate(dist, dist, kernel);
//////
//////        // �������(markers)ͼ��
//////        markers = Mat::zeros(dist.size(), CV_32S);
//////        markers += 1;  // ���������ʼ��Ϊ����
//////
//////        markers.setTo(2, keyFrame);  // �ؼ�֡������Ϊ2
//////
//////        // Ӧ�÷�ˮ���㷨
//////        watershed(currentFrame, markers);
//////
//////        // ���ָ�������ɫ
//////        Mat segmented;
//////        markers.convertTo(segmented, CV_8U);
//////        bitwise_and(currentFrame, currentFrame, segmented, segmented);
//////
//////        // ��ʾԭ��Ƶ���ؼ�֡�ͷָ����
//////        imshow("Original Video", currentFrame);
//////        imshow("Key Frame", keyFrame);
//////        imshow("Segmented Frame", segmented);
//////
//////        char key = waitKey(30);
//////        if (key == 27)  // ����ESC���˳�ѭ��
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
////    // ��ȡ��Ƶ
////    VideoCapture cap("E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1.avi");
////
////    if (!cap.isOpened()) {
////        std::cerr << "Error: Unable to open the video file." << std::endl;
////        return -1;
////    }
////
////    Mat prevFrame, currentFrame, diffFrame, keyFrame, segmentedFrame;
////
////    // ��ȡ��һ֡��Ϊ��ʼ֡
////    cap >> prevFrame;
////
////    // �趨��ֵ
////    int thresholdValue = 30;
////
////    while (true) {
////        // ��ȡ��ǰ֡
////        cap >> currentFrame;
////
////        if (currentFrame.empty()) {
////            std::cerr << "End of video stream." << std::endl;
////            break;
////        }
////
////        // ����ǰ����֡�Ҷ�ֵ֮��
////        absdiff(prevFrame, currentFrame, diffFrame);
////
////        // ת��Ϊ�Ҷ�ͼ
////        cvtColor(diffFrame, diffFrame, COLOR_BGR2GRAY);
////
////        // ��ֵ������
////        threshold(diffFrame, keyFrame, thresholdValue, 255, THRESH_BINARY);
////
////        // ����ǰһ֡
////        prevFrame = currentFrame;
////
////        // ��ʾԭ��Ƶ���ؼ�֡�ͷָ����
////        imshow("Original Video", currentFrame);
////        imshow("Key Frame", keyFrame);
////
////        char key = waitKey(30);
////        if (key == 27)  // ����ESC���˳�ѭ��
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
////    // ��ȡ��Ƶ
////    VideoCapture cap("E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1.avi");
////
////    if (!cap.isOpened()) {
////        std::cerr << "Error: Unable to open the video file." << std::endl;
////        return -1;
////    }
////
////    Mat prevFrame, currentFrame, diffFrame, keyFrame, segmentedFrame;
////
////    // ��ȡ��һ֡��Ϊ��ʼ֡
////    cap >> prevFrame;
////
////    // �趨��ֵ
////    int thresholdValue = 30;
////
////    while (true) {
////        // ��ȡ��ǰ֡
////        cap >> currentFrame;
////
////        if (currentFrame.empty()) {
////            std::cerr << "End of video stream." << std::endl;
////            break;
////        }
////
////        // ����ǰ����֡�Ҷ�ֵ֮��
////        absdiff(prevFrame, currentFrame, diffFrame);
////
////        // ת��Ϊ�Ҷ�ͼ
////        cvtColor(diffFrame, diffFrame, COLOR_BGR2GRAY);
////
////        // ��ֵ������
////        threshold(diffFrame, keyFrame, thresholdValue, 255, THRESH_BINARY);
////
////        // ����ǰһ֡
////        prevFrame = currentFrame;
////
////        // ��ʾԭ��Ƶ���ؼ�֡�ͷָ����
////        imshow("Original Video", currentFrame);
////        imshow("Key Frame", keyFrame);
////
////        // ��ֵ�ָ���ʾ
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
////        if (key == 27)  // ����ESC���˳�ѭ��
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
////    // ��ȡ��Ƶ
////    VideoCapture cap("E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1.avi");
////
////    if (!cap.isOpened()) {
////        cout << "Error opening video file" << endl;
////        return -1;
////    }
////
////    int frame_num = 0;
////    Mat frame, prev_frame, gray_frame, thresh_frame;
////    while (cap.read(frame)) {
////        // ����ǰ����֡�Ҷ�ֵ֮��  
////        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
////        absdiff(prev_frame, gray_frame, thresh_frame);
////        threshold(thresh_frame, thresh_frame, 25, 255, THRESH_BINARY);
////
////        // ��ʾԭ��Ƶ���ؼ�֡�ͷָ����  
////        imshow("Original", frame);
////        imshow("Key Frame", gray_frame);
////        imshow("Segmented", thresh_frame);
////        imshow("Difference", thresh_frame);
////
////        // �����ǰ֡�ǹؼ�֡�����䱣��Ϊͼ���ļ�  
////        if (countNonZero(thresh_frame) > 2000) {
////            char filename[100];
////            sprintf(filename, "key_frame_%03d.png", frame_num++);
////            imwrite(filename, gray_frame);
////        }
////
////        // ����ǰ֡����Ϊǰһ֡���Ա���һ֡���бȽ�  
////        prev_frame = gray_frame.clone();
////
////        // �ȴ������� 30 ������Զ�������һ֡  
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
//    // ��ȡ��Ƶ
//    VideoCapture cap("E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1.avi");
//    if (!cap.isOpened()) {
//        cout << "Error opening video file" << endl;
//        return -1;
//    }
//
//    Mat currentFrame, previousFrame, diffFrame, grayDiff, segmentedFrame;
//
//    int thresholdValue = 30; // �ҶȲ���ֵ
//    int keyFrameThreshold = 500; // �ؼ�֡�ҶȲ�����ֵ
//
//    int frameCount = 0;
//    int keyFrameCount = 0;
//
//    // ��Ƶ֡����ѭ��
//    while (true) {
//        cap >> currentFrame;
//
//        if (currentFrame.empty())
//            break;
//
//        cvtColor(currentFrame, currentFrame, COLOR_BGR2GRAY);
//
//        if (!previousFrame.empty()) {
//            // ����ǰ����֡�Ҷ�ֵ֮��
//            absdiff(previousFrame, currentFrame, diffFrame);
//            threshold(diffFrame, grayDiff, thresholdValue, 255, THRESH_BINARY);
//
//            // ����ҶȲ��ܺ�
//            int diffSum = sum(grayDiff)[0];
//
//            // �ж��Ƿ�Ϊ�ؼ�֡
//            if (diffSum > keyFrameThreshold) {
//                keyFrameCount++;
//
//                // �ָ�ؼ�֡
//                // �������ѡ����ֵ�ָ��㷨���ˮ���㷨
//                // ��������ֵ�ָ��㷨��ʾ��
//                threshold(currentFrame, segmentedFrame, 128, 255, THRESH_BINARY);
//
//                // ��ʾ�ؼ�֡�ͷָ���
//                imshow("Key Frame", currentFrame);
//                imshow("Segmented Frame", segmentedFrame);
//            }
//        }
//
//        // ��ʾԭ��Ƶ
//        imshow("Original Video", currentFrame);
//
//        // ����ǰһ֡
//        currentFrame.copyTo(previousFrame);
//
//        frameCount++;
//
//        // ��ESC���˳�ѭ��
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
//    // ����Ƶ�ļ�
//    VideoCapture cap("E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1.avi");
//
//    if (!cap.isOpened()) {
//        std::cerr << "Error opening video file." << std::endl;
//        return -1;
//    }
//
//    Mat frame, prevFrame, diff, gray, segmented;
//
//    // ��ȡ��һ֡
//    cap >> prevFrame;
//
//    while (cap.read(frame)) {
//        // ת��Ϊ�Ҷ�ͼ��
//        cvtColor(frame, gray, COLOR_BGR2GRAY);
//        cvtColor(prevFrame, prevFrame, COLOR_BGR2GRAY);
//
//        // ����ǰ����֡�ĻҶȲ���
//        absdiff(gray, prevFrame, diff);
//
//        // ������ֵ���ҵ��ؼ�֡
//        double thrshold = 30.0;
//        threshold(diff, diff, thrshold, 255, THRESH_BINARY);
//
//        // �����������ӷ�ˮ���㷨�������ָ��㷨
//
//        // ��ʾԭ��Ƶ���ؼ�֡�ͷָ���
//        imshow("Original Video", frame);
//        imshow("Key Frame", diff);
//        imshow("Segmented Result", segmented);
//
//        // �ȴ��������������ESC�����˳�ѭ��
//        char key = waitKey(30);
//        if (key == 27)
//            break;
//
//        // ����ǰһ֡
//        prevFrame = frame.clone();
//    }
//
//    // �ͷ�VideoCapture�͹رմ���
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
//    VideoCapture capture("E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1.avi"); // �滻�������Ƶ�ļ�·��
//
//    if (!capture.isOpened()) {
//        cout << "Error: Could not open video file." << endl;
//        return -1;
//    }
//
//    Mat frame, prevFrame, diff, keyFrame, segmented;
//
//    double thresholdValue = 50.0; // �ɵ�������ֵ
//    int keyFrameCount = 0;
//
//    while (true) {
//        capture >> frame;
//
//        if (frame.empty()) {
//            break; // ��Ƶ����
//        }
//
//        cvtColor(frame, frame, COLOR_BGR2GRAY);
//
//        if (!prevFrame.empty()) {
//            absdiff(prevFrame, frame, diff);
//            threshold(diff, diff, thresholdValue, 255, THRESH_BINARY);
//
//            if (countNonZero(diff) > 5000) { // �ɵ�������ֵ
//                keyFrame = frame.clone();
//                keyFrameCount++;
//
//                // ��ֵ�ָ��㷨
//                threshold(keyFrame, segmented, 128, 255, THRESH_BINARY);
//
//                // ��ʾԭ��Ƶ���ؼ�֡�ͷָ����
//                imshow("Original Video", frame);
//                imshow("Key Frame", keyFrame);
//                imshow("Segmented Result", segmented);
//
//                int key = waitKey(30); // ����ÿ֮֡��ĵȴ�ʱ�䣬��λΪ���룬������Ƶ�����ٶ�
//                if (key == 27) // ����ESC���˳�ѭ��
//                    break;
//
//                destroyAllWindows(); // �ر����д���
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
* @brief ����ֱ��ͼ��������ֵ��Ȼ�󲨹���ֵ��
* @param currentframe ����ͼ��
*/

void threshBasic(const Mat& currentframe)
{
	// ����Ҷ�ֱ��ͼ��zero�������ڴ���һ����СΪ256��Mat��������ΪCV_32F��ȫ����ʼ��Ϊ0
	Mat histogram = Mat::zeros(Size(256, 1), CV_32SC1);
	// ��ȡ�и�
	int rows = currentframe.rows;
	int cols = currentframe.cols;
	// ����ͼ�񣬼���Ҷ�ֱ��ͼ
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int index = int(currentframe.at<uchar>(i, j));
			histogram.at<int>(0, index) += 1;
		}
	}
	// �ҵ��Ҷ�ֱ��ͼ����ֵ��Ӧ�ĻҶ�ֵ
	Point peak1;
	// ���������ҵ�ȫ����С�����ֵ
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
	//�ҵ�������ֵ֮�����Сֵ��Ӧ�ĻҶ�ֵ����Ϊ��ֵ
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
	* src:����ͼ��
	* dst:���ͼ��
	* thresh:��ֵ
	* maxval:���ͼ�������ֵ
	* type:��ֵ���ͣ�THRESH_BINARY ָ��ֵ��
	*/
	Mat threshImage;
	threshold(currentframe, threshImage, thresh, 255, THRESH_BINARY);
	imshow("ֱ��ͼ��ֵ�ָ���", threshImage);
}

/**
* @brief ����ط�
* @param currentframe ����ͼ��
* @param threshImage ���ͼ��
*/

void threshEntroy(Mat currentframe)
{
	int hist_t[256] = { 0 }; //ÿ������ֵ����
	int index = 0;//����ض�Ӧ�ĻҶ�
	double Property = 0.0;// ����ռ�б���
	double maxEntropy = -1.0;//�����
	double frontEntropy = 0.0;//ǰ����
	double backEntropy = 0.0;//������
	int sum_t = 0;	//��������
	int nCol = currentframe.cols;//ÿ�е�������
	// ����ÿ������ֵ�ĸ���
	for (int i = 0; i < currentframe.rows; i++)
	{
		uchar* pData = currentframe.ptr<uchar>(i);
		for (int j = 0; j < nCol; ++j)
		{
			++sum_t;
			hist_t[pData[j]] += 1;
		}
	}
	// ���������
	for (int i = 0; i < 256; i++)
	{
		// ����������
		double sum_back = 0;
		for (int j = 0; j < i; j++)
		{
			sum_back += hist_t[j];
		}

		//������
		for (int j = 0; j < i; j++)
		{
			if (hist_t[j] != 0)
			{
				Property = hist_t[j] / sum_back;
				backEntropy += -Property * logf((float)Property);
			}
		}
		//ǰ����
		for (int k = i; k < 256; k++)
		{
			if (hist_t[k] != 0)
			{
				Property = hist_t[k] / (sum_t - sum_back);
				frontEntropy += -Property * logf((float)Property);
			}
		}

		if (frontEntropy + backEntropy > maxEntropy)// �������
		{
			maxEntropy = frontEntropy + backEntropy;
			index = i;
		}
		frontEntropy = 0.0;	// ǰ��������
		backEntropy = 0.0;	// ����������
	}
	Mat threshImage;
	threshold(currentframe, threshImage, index, 255, 0); //������ֵ�ָ�
	imshow("����ط�", threshImage);
}

/**
* @brief ��ˮ���㷨
* @param currentframe ����ͼ��
*/


void watershed(Mat& _img)
{
	Mat image = _img;
	Mat grayImage;
	// ת���ɻҶ�ͼ��
	cvtColor(image, grayImage, CV_BGR2GRAY);
	// ��˹�˲��ͱ�Ե���
	GaussianBlur(grayImage, grayImage, Size(3, 3), 0, 0);
	Canny(grayImage, grayImage, 50, 150, 3);
	// ����  
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	
	// �ҵ�����
	findContours(grayImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	
	// ���ɷ�ˮ��ͼ��
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);
	Mat biaoji(image.size(), CV_32S);
	biaoji = Scalar::all(0);
	int index = 0, n1 = 0;
	// ��������
	for (; index >= 0; index = hierarchy[index][0], n1++)
	{
		drawContours(biaoji, contours, index, Scalar::all(n1 + 1), 1, 8, hierarchy);	// ��������
		drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);	// ��������
	}

	Mat biaojiShows;
	convertScaleAbs(biaoji, biaojiShows);
	watershed(image, biaoji);

	Mat w1;	// ��ˮ����ͼ��
	convertScaleAbs(biaoji, w1);	// ת����8λͼ��
	imshow("��ˮ���㷨", w1);
	waitKey(1);
}

//void watershedSegmentation(const Mat& inputImage, Mat& markers) {
//	// ����1��ͼ��ҶȻ�
//	Mat grayImage;
//	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
//
//	// ����2������ͼ���ݶ�
//	Mat gradient;
//	Sobel(grayImage, gradient, CV_32F, 1, 1);
//
//	// ����3�������ݶ�ֵ
//	Mat absGradient;
//	convertScaleAbs(gradient, absGradient);
//	threshold(absGradient, absGradient, 50, 255, THRESH_BINARY);
//
//	// ����4-7����ˮ���㷨
//	markers = Mat::zeros(gradient.size(), CV_32S);
//	watershed(inputImage, markers);
//
//	// ����ǵ�������ӻ�
//	Mat segmentedImage = Mat::zeros(inputImage.size(), inputImage.type());
//	for (int i = 0; i < markers.rows; ++i) {
//		for (int j = 0; j < markers.cols; ++j) {
//			int index = markers.at<int>(i, j);
//			if (index == -1) {
//				segmentedImage.at<Vec3b>(i, j) = Vec3b(0, 0, 255);  // ����Ϊ��ɫ
//			}
//			else {
//				segmentedImage.at<Vec3b>(i, j) = inputImage.at<Vec3b>(i, j);
//			}
//		}
//	}
//
//	// ��ʾ�������ѡ��
//	Mat w1;	// ��ˮ����ͼ��
//	convertScaleAbs(segmentedImage, w1);	// ת����8λͼ��
//	imshow("Segmented Image", w1);
//	waitKey(100);  // �ȴ�1���룬ȷ��ͼ����ʾ
//}

//�˵�
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
		cout << "�������" << endl;
		break;
	}
	}
}

int main() {
	int in;
	int flag = 1;
	while (flag) {
		cout << "ѡ��ֵ�ָ��㷨��" << endl;
		cout << "1.ֱ��ͼ��" << endl;
		cout << "2.������㷨" << endl;
		cout << "3.�����ˮ���㷨" << endl;
		cin >> in;
		flag = 0;
	}
	VideoCapture cap;
	cap.open("E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1.avi");
	// ��ʧ��
	if (!cap.isOpened()) {
		cout << "�޷���ȡ��Ƶ" << endl;
		system("pause");
		return -1;
	}
	int k = -1;

	Mat tempframe, currentframe, keyframe, previousframe, diff;// ��ǰ֡��ǰһ֡���ؼ�֡
	Mat frame;
	int framenum = 0;
	int dur_time = 1;
	int thresholdValue = 50;
	while (true) {
		//��ʾ��Ƶ
		Mat frame1;
		// ��ȡ����֡
		bool ok = cap.read(frame);
		if (!ok) break;
		imshow("��Ƶ", frame);
		k = waitKey(50);
		Mat markers;
		cap >> frame1;	//��ȡ��Ƶ��ÿ֡
		tempframe = frame1;
		framenum++;
		// ��һ֡������ǵ�һ֡���ͽ���ǰ֡��ֵ��ǰһ֡
		if (framenum == 1) {
			keyframe = tempframe;//��һ֡��Ϊһ���ؼ�֡
			imshow("�ؼ�֡", keyframe);//�ؼ�֡1
			cvtColor(keyframe, currentframe, CV_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
			Mat threshImage;
		//	imshow("�Ҷ�ͼ", currentframe);//��ʾ�Ҷ�ͼ
			menu(keyframe, threshImage, in, markers);
		}
		//��һ֮֡�������֡
		if (framenum >= 2) {
			cvtColor(tempframe, currentframe, CV_BGR2GRAY);//��ǰ֡ת�Ҷ�ͼ
			cvtColor(keyframe, previousframe, CV_BGR2GRAY);//�ο�֡ת�Ҷ�ͼ
			absdiff(currentframe, previousframe, diff);// ��֡���
			
			float sum = 0.0;
			// ������֡��ֵ��ۼ�
			for (int i = 0; i < diff.rows; i++) {
				uchar* data = diff.ptr<uchar>(i);
				for (int j = 0; j < diff.cols; j++)
				{
					sum += data[j];
				}
			}
			if (sum > 7000000) {
				// ��ֵ
				imshow("�ؼ�֡", tempframe);
				keyframe = tempframe;
				Mat threshImage;
				//imshow("�Ҷ�ͼ", currentframe);
				menu(keyframe, threshImage, in, markers);
			}
		}
	}
	return 0;
}
//int main() {
//	// ����Ƶ�ļ�
//	cv::VideoCapture cap("E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1.avi");
//
//	if (!cap.isOpened()) {
//		std::cerr << "Error opening video file\n";
//		return -1;
//	}
//
//	cv::Mat previous_frame, current_frame, diff;
//
//	// ��ȡ��һ֡��Ϊ�ο�֡
//	cap >> previous_frame;
//
//	while (true) {
//		// ��ȡ��ǰ֡
//		cap >> current_frame;
//
//		if (current_frame.empty()) {
//			break;  // ��Ƶ����ʱ�˳�ѭ��
//		}
//
//		// ��֡ת��Ϊ�Ҷ�ͼ
//		cv::Mat current_gray, previous_gray;
//		cv::cvtColor(current_frame, current_gray, cv::COLOR_BGR2GRAY);
//		cv::cvtColor(previous_frame, previous_gray, cv::COLOR_BGR2GRAY);
//
//		// ������֮֡��Ĳ���
//		cv::absdiff(current_gray, previous_gray, diff);
//
//		// ���ò�����ֵ
//		double threshold = 60;
//		cv::threshold(diff, diff, threshold, 255, cv::THRESH_BINARY);
//
//
//		// ��ʾ��ǰ֡�Ͳ���ͼ
//		cv::imshow("Current Frame", current_frame);
//		cv::imshow("Difference", diff);
//
//		// ���²ο�֡
//		previous_frame = current_frame.clone();
//
//		// ��ESC���˳�ѭ��
//		if (cv::waitKey(30) == 27) {
//			break;
//		}
//	}
//
//	// �رմ���
//	cv::destroyAllWindows();
//
//	return 0;
//}
//int main() {
//	cv::VideoCapture cap("E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1.avi");
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


