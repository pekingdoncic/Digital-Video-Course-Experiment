//实现目标检测
#include "stdio.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat frameDiff(Mat old, Mat currentframe, Mat afterThresold)
{
    Mat frame_gray, old_gray;
    //前后两帧图像进行灰度化
    cvtColor(currentframe, frame_gray, COLOR_BGR2GRAY);
    cvtColor(old, old_gray, COLOR_BGR2GRAY);
    Size size; size.height = size.width = 7;

    //对前后两帧的灰度图像进行预处理
    GaussianBlur(frame_gray, frame_gray, size, 1);
    GaussianBlur(old_gray, old_gray, size, 1);

    //帧差法
    Mat diff;
    absdiff(frame_gray, old_gray, diff);

    //差值图像的二值化
    threshold(diff, afterThresold, 0, 255, THRESH_OTSU);

    dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)));//膨胀
    dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(8, 8)));
    erode(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//腐蚀
    //// 腐蚀操作
    //erode(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

    ////// 或者开运算
    //morphologyEx(afterThresold, afterThresold, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    imshow("相邻两帧帧差法目标检测", afterThresold);
    return afterThresold;
}

Mat frameDiff3(Mat old, Mat currnetframe,Mat next_frame,Mat afterThresold)
{
    Mat frame_gray, old_gray, next_gray;
    //三帧图像进行灰度化
    cvtColor(currnetframe, frame_gray, COLOR_BGR2GRAY);
    cvtColor(old, old_gray, COLOR_BGR2GRAY);
    cvtColor(next_frame, next_gray, COLOR_BGR2GRAY);

    //三帧的灰度图像进行预处理
    GaussianBlur(frame_gray, frame_gray, Size(5, 5), 1);
    GaussianBlur(old_gray, old_gray, Size(5, 5), 1);
    GaussianBlur(next_gray, next_gray, Size(5, 5), 1);

    //帧差法
    Mat diff1, diff2;
    absdiff(frame_gray, old_gray, diff1);
    absdiff(frame_gray, next_gray, diff2);
    Mat afterThresold1, afterThresold2;

    //差值图像的二值化
    threshold(diff1, afterThresold1, 0, 255, THRESH_OTSU);
    threshold(diff2, afterThresold2, 0, 255, THRESH_OTSU);
    /* threshold(diff1, afterThresold1,60, 255, THRESH_BINARY);
     threshold(diff2, afterThresold2,60, 255, THRESH_BINARY);*/

     //两帧相加
    add(afterThresold1, afterThresold2, afterThresold);
    //blur(afterThresold, afterThresold, Size(5, 5));
    //medianBlur(afterThresold, afterThresold, 3);
    // GaussianBlur(afterThresold, afterThresold, Size(5, 5), 0);
    dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(3, 5)));//膨胀
    erode(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//腐蚀
    //dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));//膨胀
    morphologyEx(afterThresold, afterThresold, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1)));     // 形态学处理--闭运算
    imshow("相邻三帧帧差法目标检测", afterThresold);
    return afterThresold;
}

Mat KNN(Mat currentframe, Mat afterThresold, Ptr<BackgroundSubtractor> pKNN)
{
    pKNN->apply(currentframe, afterThresold);
    threshold(afterThresold, afterThresold, 200, 255, THRESH_BINARY);//阈值分割进行二值化
    morphologyEx(afterThresold, afterThresold, MORPH_OPEN, getStructuringElement(1, Size(3, 5)));//形态学处理--开运算
    dilate(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//膨胀
    imshow("背景减法--KNN", afterThresold);
    return afterThresold;
}

Mat GMM(Mat currentframe, Mat afterThresold, Ptr<BackgroundSubtractorMOG2> bgsubtractor)
{
    bgsubtractor->apply(currentframe, afterThresold, 0.01);
    threshold(afterThresold, afterThresold, 200, 255, THRESH_BINARY);//阈值分割进行二值化

    morphologyEx(afterThresold, afterThresold, MORPH_OPEN, getStructuringElement(1, Size(3, 5)));//形态学处理--开运算
    dilate(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//膨胀
    imshow("背景减法--混合高斯模型", afterThresold);
    return afterThresold;
}

Mat EdgeDetection(Mat old,Mat currentframe, Mat next_frame, Mat afterThresold )
{
    Mat frame_gray, old_gray, next_gray;
    //三帧图像进行灰度化
    cvtColor(currentframe, frame_gray, COLOR_BGR2GRAY);
    cvtColor(old, old_gray, COLOR_BGR2GRAY);
    cvtColor(next_frame, next_gray, COLOR_BGR2GRAY);
    //三帧的灰度图像进行预处理
    GaussianBlur(frame_gray, frame_gray, Size(5, 5), 1);
    GaussianBlur(old_gray, old_gray, Size(5, 5), 1);
    GaussianBlur(next_gray, next_gray, Size(5, 5), 1);

     //帧差法
    Mat diff1, diff2;
    absdiff(frame_gray, old_gray, diff1);
    absdiff(frame_gray, next_gray, diff2);
    Mat afterThresold1, afterThresold2;

    //差值图像的二值化
    threshold(diff1, afterThresold1, 0, 255, THRESH_OTSU);
    threshold(diff2, afterThresold2, 0, 255, THRESH_OTSU);

    Mat Canny_afterThresold;
    //两帧相加
    add(afterThresold1, afterThresold2, afterThresold);
    Canny(afterThresold, Canny_afterThresold, 100, 200, 3, false);//Canny算子处理三帧帧差法得到的图像
    bitwise_or(afterThresold, Canny_afterThresold, afterThresold);//边缘检测结果与原图像进行逻辑或操作

    dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));//膨胀
    erode(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//腐蚀
    morphologyEx(afterThresold, afterThresold, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1)));     // 形态学处理--闭运算
    imshow("三帧差分和边缘检测结合算法", afterThresold);
    return afterThresold;
}

void menu()
{
    cout << "-------------------------------------------" << endl;
    cout << "             选择目标检测的方式            " << endl;
    cout << "-------------------------------------------" << endl;
    cout << "            1.相邻两帧帧差法" << endl;
    cout << "            2.相邻三帧帧差法" << endl;
    cout << "            3.背景减法--KNN" << endl;
    cout << "            4.背景减法--GMM,混合高斯模型" << endl;
    cout << "            5.相邻三帧帧差法和边缘检测" << endl;
    cout << "请输入您的选择：（如：1）" << endl;
}

Mat choice(int choice,Mat afterThresold,Mat currentframe,Mat old,Mat nextframe, Ptr<BackgroundSubtractor> pKNN, Ptr<BackgroundSubtractorMOG2> bgsubtractor)
{
    switch (choice)
    {
        case 1:
        {
            afterThresold = frameDiff(old, currentframe, afterThresold);
            break;
        }
        case 2:
        {
            afterThresold = frameDiff3(old, currentframe, nextframe, afterThresold);
            break;
        }
        case 3:
        {
            afterThresold = KNN(currentframe, afterThresold, pKNN);
            break;
        }
        case 4:
        {
            afterThresold = GMM(currentframe, afterThresold, bgsubtractor);
            break;
        }
        case 5:
        {
            afterThresold = EdgeDetection(old, currentframe, nextframe, afterThresold);
            break;
        }
        default:
            break;
    }
    return afterThresold;
}

int main()
{
    VideoCapture video;
    String filename = "F:\\学习\\数字视频处理\\EnvironmentTest\\Videos\\exp2.avi";
    video.open(filename);
    if (!video.isOpened())
    {
        cout << "无法打开视频文件！" << endl;
        return -1;
    }
    long totalFrameNumber = video.get(CAP_PROP_FRAME_COUNT);//求总的视频帧数
    cout << totalFrameNumber << endl;
    menu();
    int m;
    cin >> m;

 
    Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
    bgsubtractor->setVarThreshold(20);
    Ptr<BackgroundSubtractor> pKNN = createBackgroundSubtractorKNN(false);

    int flag = 0;

    Mat currentframe;  //储存一帧图
    Mat old; //前一帧图像
    Mat next_frame;//获取视频下一帧

    Mat frame_gray;  //储存一帧灰度图
    Mat old_gray; //前一帧灰度图像
    Mat next_gray;//视频下一帧的灰度图像

    string num;
    while (true)
    {
        video >> currentframe; //从视频中读入一帧图像

        Mat frame2;
        currentframe.copyTo(frame2);

        if (currentframe.empty())  //视频播放完成退出
        {
            break;
        }
        if (flag != 0)
        {
            Mat afterThresold;
            video >> next_frame;
            if (next_frame.empty())  //视频播放完成退出
            {
                break;
            }
            afterThresold = choice(m, afterThresold, currentframe, old, next_frame, pKNN , bgsubtractor);

            //寻找边框
            vector<vector<Point>> contours;
            //contours：一个双重向量，向量内每个元素保存了一组由连续的Point点构成的点的集合向量，每一组Point点集就是一个轮廓
            //有多少轮廓，向量contours就有多少元素
            vector<Vec4i> hierarchy;
            //向量内每一个元素包含了4个int型变量
            //向量hierarchy内的元素和contours内的元素是一一对应的
            findContours(afterThresold, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            //afterThresold:经过处理过的二值图像
            //RETR_TREE:轮廓的检索模式
            //CHAIN_APPROX_SIMPLE:轮廓的近似方法

            //画边框
            drawContours(frame2, contours, -1, Scalar::all(0.7), 1, 8);
            //frame2:要绘制轮廓的图像
            //contours:所有输入的轮廓，即findContours()函数找到的所有轮廓，每个轮廓被保存成一个point向量
            //contourldx：指定要绘制轮廓的编号，如果是负数，则绘制所有的轮廓，一般默认-1就可
            //color：要绘制轮廓的颜色
            //thickness：要绘制轮廓的粗细
            //lineType：要绘制的轮廓的线的类型


            RNG rngs = { 012345 };//构造方法设定一个具体值，表示下面代码每次生成的结果都是一样的
            //Scalar colors = Scalar(rngs.uniform(0, 255), rngs.uniform(0, 255), rngs.uniform(0, 255));//从[0,255)范围内随机一个值

            Scalar colors = Scalar(0, 0, 255);
            int index = 0;
            int compCount = 0;
            for (; index >= 0; index = hierarchy[index][0], compCount++)//每次找后一个轮廓
            {//4个int型向量分别表示该轮廓的后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号
                Rect rec = boundingRect(contours[index]);//计算轮廓的垂直边界最小矩形，矩形是与图像上下边界平行的
                if (rec.area() <= 500)//区域小于500，则不绘画
                    continue;
                rectangle(frame2, rec, colors, 2);
                //frame2:要做处理的图片
                //rec:矩形区域
                //colors:线条颜色
                //2：线条宽度
            }
        }
        //putText(frame2,num, Point(50, 60), FONT_HERSHEY_SIMPLEX, 200, Scalar(0, 0, 255), 4, 8);
        imshow("播放视频", frame2);  //显示当前读入的一帧图像
        waitKey(30);    //延时10ms
        currentframe.copyTo(old);
  //      frame_gray.copyTo(old_gray);
        flag = 1;
        char key = waitKey(1);
        if (key == 27 || key == 113 || key == 81)
        {
            break;
        }
    }
    video.release();
}



