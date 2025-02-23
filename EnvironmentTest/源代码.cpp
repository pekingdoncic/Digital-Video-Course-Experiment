#include <iostream>  
#include "opencv2/opencv.hpp"  
#include <opencv2\imgproc\types_c.h>

using namespace cv;
using namespace std;

// ֡��˶�Ŀ����
/*
 @ temp ��ʱ֡
 @ frame ��ǰ֡
 return result ֡��˶�Ŀ��������Ϊÿ���˶�Ŀ�������ɫ�ľ��ο�
*/

Mat frameDiff(Mat temp, Mat frame)
{
	Mat result;
	// ����ǰ֡ת��Ϊ�Ҷ�ͼ��
	cvtColor(frame, frame, COLOR_BGR2GRAY);
	// ����ʱ֡ת��Ϊ�Ҷ�ͼ��
	cvtColor(temp, temp, COLOR_BGR2GRAY);
	// ����ǰ֡����ʱ֡����֡������
	absdiff(frame, temp, result);
	// ��֡����������ֵ��
	threshold(result, result, 50, 255, THRESH_BINARY);
	// �Զ�ֵ��������и�ʴ��������
	//erode(result, result, Mat());
	//dilate(result, result, Mat());

	// �Զ�ֵ�����������̬ѧ������
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(result, result, MORPH_OPEN, kernel);
	// �Զ�ֵ�����������̬ѧ������
	morphologyEx(result, result, MORPH_CLOSE, kernel);
	imshow("biyunsuan", result);
	// ���Ҷ�ֵ������е�����
	vector<vector<Point>> contours;
	findContours(result, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// ��������
	for (size_t t = 0; t < contours.size(); t++)
	{
		// ��������
		drawContours(frame, contours, t, Scalar(0, 0, 255), 2);
		
		
		// ��ȡ��ǰ�����ľ��α߽��
		Rect rect = boundingRect(contours[t]);
		// �����ͼ�ε��ϰ벿�֣�����С��  �Ŀ�
		//if (rect.y < frame.rows / 4 && rect.area() < 500)
		//	continue;
		//// �����ͼ�ε��°벿�֣�����С��  �Ŀ�
		//if (rect.y > frame.rows / 4 && rect.area() < 900)
		//	continue;
		if (rect.area() < 400)
				continue;
		// �ڵ�ǰ֡�л��ƾ��α߽��
		rectangle(frame, rect, Scalar(0, 255, 0), 2, 8, 0);
	}
	return frame;
}

// ��֡�

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
	// ����ǰ֡ת��Ϊ�Ҷ�ͼ��
	cvtColor(temp, gray1, CV_BGR2GRAY);
	cvtColor(frame, gray2, CV_BGR2GRAY);
	absdiff(gray1, gray2, diff1);	// ��һ֡��ڶ�֡�Ĳ�
	cvtColor(frame, gray3, CV_BGR2GRAY);
	absdiff(gray2, gray3, diff2);	// �ڶ�֡�����֡�Ĳ�
	absdiff(diff1, diff2, diff3);	// ��֡��Ĳ�
	threshold(diff3, threshold_output, thresh, 255, THRESH_BINARY);
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	imshow("threshold_output", threshold_output);
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	// ��������
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);	// ����αƽ�
		boundRect[i] = boundingRect(Mat(contours_poly[i]));	// ��ȡ��ǰ�����ľ��α߽��
	}
	result = frame.clone();
	for (int i = 0; i < contours.size(); i++)
	{
		// ��������-���ɫ
		drawContours(result, contours_poly, i, Scalar(0, 0, 255), 1, 8, vector<Vec4i>(), 0, Point());

		//�����ͼ�ε���1/3���֣��������С�� 400 �Ŀ�
		if (boundRect[i].y < frame.rows / 4 && boundRect[i].area() < 200)
			continue;
		//�����ͼ�ε���2/3���֣��������С�� 900 �Ŀ�
		if (boundRect[i].y > frame.rows / 4 && boundRect[i].area() < 600)
			continue;	
		//if ( boundRect[i].area() < 200)
		//	continue;
		
		rectangle(result, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0);
	}
	return result;
}

// ��Ե��ⷨ �˶�Ŀ���⺯��,ԭ�������ñ�Ե����㷨������ͼ���е��˶�Ŀ��
/*
 @ temp ��ʱ֡
 @ frame ��ǰ֡
 return result �������Ϊÿ���˶�Ŀ�������ɫ�ľ��ο�,ʹ�� canny �㷨
*/

Mat MotionDetectByCanny(Mat temp, Mat frame)
{
	Mat result;
	frame.copyTo(result);
	// 1. ����ǰ֡����ʱ֡ת��Ϊ�Ҷ�ͼ
	Mat gray1, gray2;
	cvtColor(temp, gray1, COLOR_BGR2GRAY);
	cvtColor(frame, gray2, COLOR_BGR2GRAY);
	// 2. �Ե�ǰ֡����ʱ֡���и�˹�˲�
	GaussianBlur(gray1, gray1, Size(3, 3), 0, 0);
	GaussianBlur(gray2, gray2, Size(3, 3), 0, 0);
	// 3. �Ե�ǰ֡����ʱ֡�����˶�Ŀ����
	Mat motion = gray1 - gray2;
	// 4. ���˶�Ŀ����ж�ֵ������
	threshold(motion, motion, 50, 255, THRESH_BINARY);
	// 5. �Զ�ֵ��ͼ�������̬ѧ����
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(motion, motion, MORPH_OPEN, element);
	// 6. �Զ�ֵ��ͼ����б�Ե���
	// �������������ֱ���� threshold1 �� threshold2 ��ֵ�Լ� apertureSize
	Canny(motion, motion, 3, 9, 3);
	imshow("motion", motion);
	// 7. ��������
	vector<vector<Point>> contours;
	findContours(motion, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	// 8. ��������
	for (size_t t = 0; t < contours.size(); t++)
	{
		// ��������-���ɫ
		drawContours(result, contours, t, Scalar(0, 0, 255), 2);
		//�����ͼ�ε���1/3���֣��������С�� 400 �Ŀ�
		Rect rect = boundingRect(contours[t]);
		if (rect.y < frame.rows / 4 && rect.area() < 200)
			continue;
		// �����ͼ�ε���2/3���֣��������С�� 1000 �Ŀ�
		if (rect.y > frame.rows * 1 / 4 && rect.area() < 900)
			continue;
		
		rectangle(result, rect, Scalar(0, 255, 0), 2, 8, 0);

	}
	return result;
}

// ʹ�� BackgroundSubtractorMOG() ��������ʵ���˶�Ŀ����
/*
 @ temp ��ʱ֡
 @ frame ��ǰ֡
 return result �������Ϊÿ���˶�Ŀ�������ɫ�ľ��ο�,ʹ�� BackgroundSubtractorMOG ����
*/

Mat detectMotion(Mat frame) {
	Mat result;
	frame.copyTo(result);
	//��������������
	Ptr<BackgroundSubtractorMOG2> bgModel = createBackgroundSubtractorMOG2();
	//��������
	Mat fgMask;
	bgModel->apply(frame, fgMask);
	//����
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(fgMask, fgMask, kernel);
	imshow("fgmask", fgMask);
	//��������
	vector<vector<Point>> contours;
	findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//���Ʊ�Ե
	for (size_t t = 0; t < contours.size(); t++)
	{
		// ��������-���ɫ
		drawContours(result, contours, t, Scalar(0, 0, 255), 2);
		//�����ͼ�ε���1/3���֣��������С�� 400 �Ŀ�
		Rect rect = boundingRect(contours[t]);
		if (rect.y < frame.rows / 4 && rect.area() < 200)
			continue;
		// �����ͼ�ε���2/3���֣��������С�� 1000 �Ŀ�
		if (rect.y > frame.rows * 1 / 4 && rect.area() < 900)
			continue;

		rectangle(result, rect, Scalar(0, 255, 0), 2, 8, 0);

	}
	return result;
}



int main()
{
	VideoCapture video("F:\\ѧϰ\\������Ƶ����\\EnvironmentTest\\Videos\\exp2.avi");
	if (!video.isOpened())
	{
		cout << "video open error!" << endl;
		return 0;
	}
	int frameCount = video.get(CAP_PROP_FRAME_COUNT); 	// ��ȡ֡��
	double FPS = video.get(CAP_PROP_FPS); 	// ��ȡ FPS
	Mat frame;
	Mat temp;// �洢ǰһ֡ͼ��
	Mat fst;
	Mat result;// �洢���ͼ��
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
			// ��¼��һ֡
			fst = frame.clone();

			// ֡�
			//result = frameDiff(frame, frame); 
			
			// ��֡�
			// result = frameDiff3(frame, frame); 
			
			// ��Ե��ⷨ
			result = MotionDetectByCanny(frame, frame);
		}
		else
		{
			// ֡�
			//result  = frameDiff(temp, frame);   
			
			// ��֡�
			// result  = frameDiff3(temp, frame);   

			// ��Ե��ⷨ
			result = MotionDetectByCanny(temp, frame);
		}
		imshow("result", result);
		if (waitKey(1000.0 / FPS) == 27)
		{
			cout << "ESC�˳�!" << endl;
			break;
		}
		temp = frame.clone();
	}
	return 0;
}