#include "stdio.h"
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int twopeak=1;
int otsu=1;
int iter = 1;
int max_entropy = 1;

//产生随机颜色
Vec3b ranColor(int value)//生成随机颜色
{
    value = value % 255;  //生成0~255的随机数
    RNG rng;//随机数产生器
    int r = rng.uniform(0, value);
    int g = rng.uniform(0, value);
    int b = rng.uniform(0, value);
    return Vec3b(r, g, b);
}

//迭代法进行阈值分割
Mat iterative(Mat frame)
{
    int iMaxGrayValue = 0, iMinGrayValue = 255;//获取最大灰度值与最小灰度值
    long hist[256] = { 0 };//直方图数组
    //获得直方图
    for (int i = 0; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
        for (int j = 0; j < frame.cols; j++)
        {
            hist[int(data[j])]++;
            if (iMinGrayValue > data[j])
                iMinGrayValue = data[j];
            if (iMaxGrayValue < data[j])
                iMaxGrayValue = data[j];
        }
    }
    int iThreshold = 0;///阈值
    int iNewThreshold= (iMinGrayValue + iMaxGrayValue) / 2;//初始阈值

    long totalValue_f = 0;//计算前一区域的灰度值和
    long meanValue_f;//计算前一区域的灰度平均

    long totalValue_b = 0;//计算后一区域的灰度值和
    long meanValue_b;//计算后一区域的灰度平均
  
    long lp1 = 0,lp2=0;//用于计算区域灰度平均值的中间变量
  
    for (int iIterationTimes = 0; abs(iThreshold - iNewThreshold) > 2 && iIterationTimes < 100; iIterationTimes++)//最多迭代100次
    {
        iThreshold = iNewThreshold;
        totalValue_f = 0; totalValue_b = 0; lp1 = 0; lp2 = 0;
        for (int j = iMinGrayValue; j < iNewThreshold; j++)
        {
            lp1 += hist[j] * j;
            totalValue_f += hist[j];
        }
        meanValue_f = lp1 / totalValue_f;
      
        for (int k = iNewThreshold; k < iMaxGrayValue; k++)
        {
            lp2 += hist[k] * k;
            totalValue_b += hist[k];
        }
        meanValue_b = lp2 / totalValue_b;
        iNewThreshold = (meanValue_f + meanValue_b) / 2;//新阈值
    }
    cout << "第 "<<iter << " 个关键帧的迭代法的阈值为  " << iNewThreshold << endl;
    iter++;
    Mat after1;
    threshold(frame, after1, iNewThreshold, 255, THRESH_BINARY);//阈值分割
    return after1;
}

//OTSU法阈值分割
Mat OTSU(Mat frame)
{
    //循环变量
    long i, j, t;
    //阈值，最大灰度值与最小灰度值，两个区域的平均灰度值
    int iThreshold, iNewThreshold, iMaxGrayValue, iMinGrayValue, iMean1GrayValue, iMean2GrayValue;
    iMaxGrayValue = 0;
    iMinGrayValue = 255;
    //前景点数占图像比例，背景点数占图像比例
    double w0, w1;
    //方差
    double G = 0, tempG = 0;
    //用于计算区域灰度平均值的中间变量
    long lP1, lP2, lS1, lS2;
    long lHistogram[256] = { 0 };
    //获得直方图
    for (int i = 0; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
        for (int j = 0; j < frame.cols; j++)
        {
            lHistogram[int(data[j])]++;
            if (iMinGrayValue > data[j])
                iMinGrayValue = data[j];
            if (iMaxGrayValue < data[j])
                iMaxGrayValue = data[j];
        }
    }
    int n = frame.rows * frame.cols;

    for (t = iMinGrayValue; t < iMaxGrayValue; t++)
    {
        iNewThreshold = t;lP1 = 0;lP2 = 0;lS1 = 0;lS2 = 0;
        //求两个区域的灰度平均值
        for (i = iMinGrayValue; i <= iNewThreshold; i++)
        {
            lP1 += lHistogram[i] * i;
            lS1 += lHistogram[i];
        }
        iMean1GrayValue = (unsigned char)(lP1 / lS1);
        w0 = (double)(lS1) / n;
        for (i = iNewThreshold + 1; i <= iMaxGrayValue; i++)
        {
            lP2 += lHistogram[i] * i;
            lS2 += lHistogram[i];
        }
        iMean2GrayValue = (unsigned char)(lP2 / lS2);
        w1 = 1 - w0;
        double mean = w0 * iMean1GrayValue + w1 * iMean2GrayValue;
        //G = (double)w0 * w1 * (iMean1GrayValue - iMean2GrayValue) * (iMean1GrayValue - iMean2GrayValue);
        G = (double)w0 * (iMean1GrayValue - mean) * (iMean1GrayValue - mean) + w1 * (iMean2GrayValue - mean) * (iMean2GrayValue - mean);
        if (G > tempG)//求出使G值最大的t
        {
            tempG = G;
            iThreshold = iNewThreshold;
        }
    }
    cout << "第 " << otsu << " 个关键帧的OTSU法的阈值为  " << iThreshold << endl;
    otsu++;
    Mat after2 ;
    threshold(frame, after2, iThreshold, 255, THRESH_BINARY);////阈值分割
    return after2;
}

//双峰波谷法阈值分割
Mat TwoPeak(Mat frame) {
    typedef struct Peak
    {
        int grey;
        int num;
    }Peak;
    unsigned char* p;
    int histarr[256];
    //获得直方图
    for (int i = 0; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
        for (int j = 0; j < frame.cols; j++)
        {
            histarr[int(data[j])]++;
        }
    }

    Peak peaks[256];//用于统计波峰
    int cnt = 0;//计数
    peaks[cnt].grey = 0;//peaks[i]的灰度值，先存0
    peaks[cnt].num = histarr[0];//peaks[i]的统计数
    cnt++;//cnt=1
    for (int i = 1; i < 255; i++)
    {
        if (histarr[i] > histarr[i + 1] && histarr[i] > histarr[i - 1])
        {
            //波峰
            peaks[cnt].grey = i;
            peaks[cnt].num = histarr[i];
            cnt++;
        }
    }
    peaks[cnt].grey = 255;
    peaks[cnt].num = histarr[255];
    cnt++;//波峰数
    Peak max = peaks[0], snd = peaks[0];//寻找最大和第二大的波峰
    for (int i = 0; i < cnt; i++)
    {
        if (peaks[i].num > max.num)
        {
            snd = max;
            max = peaks[i];
        }
        else if (peaks[i].num > snd.num)
        {
            snd = peaks[i];
        }
    }

    int i = (max.grey > snd.grey ? snd.grey : max.grey);//从小的开始
    int mnum = (max.grey > snd.grey ? max.grey : snd.grey);
    Peak K = max;
    for (; i < mnum; i++)//找两个波峰之间的波谷
    {
        if (histarr[i] < K.num)
        {
            K.num = histarr[i];
            K.grey = i;
        }
    }
    cout << "第 " << twopeak << " 个关键帧的双峰波谷法的阈值为  " << K.grey << endl;
    twopeak++;
    //波谷的灰度值(阈值)为 K.grey	
    Mat after3;
    threshold(frame, after3, K.grey, 255, THRESH_BINARY);////阈值分割
    return after3;
}

//分水岭算法
Mat WatershedSegmentation(Mat frame, Mat frame_gray)
{
    //----1.处理图像
    GaussianBlur(frame_gray, frame_gray, Size(5, 5), 2);   //将灰度图进行高斯滤波，避免过度分割
    Canny(frame_gray, frame_gray, 80, 150);//将灰度图进行canny的边缘检测，确定种子的大概边界

    //----2.查找轮廓
    //调用findCounter函数来查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(frame_gray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());//查找处理后的灰度图像的轮廓，hierarchy是区域号
    //CV_CHAIN_APPROX_SIMPLE 仅保存轮廓的拐点信息，把所有轮廓拐点处的点保存入contours向量内，拐点与拐点之间直线段上的信息点不予保留

    //---3.标记轮廓
    Mat mask(frame.rows, frame.cols, CV_32SC1);//初始化创建标记
    mask = Scalar::all(0);//给每个通道都赋值0
    int index = 0;//画第几个轮廓
    int compCount = 0;//用于改变颜色
    for (; index >= 0; index = hierarchy[index][0], compCount++)//每次找后一个轮廓
    {
        //对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点
        drawContours(mask, contours, index, Scalar::all(compCount + 80), 1, 8, hierarchy);//标记轮廓
        //1为轮廓的线宽
    }
    //imshow("markers", mask * 1000);//展示边界轮廓

    //---4.watershed
    watershed(frame, mask);//使用opencv的库函数，对先前得到的轮廓图进行注水
    //---5.颜色填充
    //对每一个区域进行颜色填充
    Mat afterFilled = Mat::zeros(frame.size(), CV_8UC3);
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            int index = mask.at<int>(i, j);//区域的标记符号
            if (mask.at<int>(i, j) == -1)//区域与区域之间的分界处的值被置为“-1”，以做区分
            {
                afterFilled.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            }
            else
            {
                afterFilled.at<Vec3b>(i, j) = ranColor(index);
            }
        }
    }
    return afterFilled;
}

//最大熵进行阈值分割
Mat Max_Entropy(Mat frame)
{
    long lHistogram[256] = { 0 };
    //获得直方图
    for (int i = 0; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
        for (int j = 0; j < frame.cols; j++)
        {
            if(data[j]==0)
                continue; //排除掉黑色的像素点
            lHistogram[int(data[j])]++;
        }
    }
    
     float probability = 0.0; //概率
     float max_Entropy = 0.0; //最大熵
     int totalpix= frame.rows * frame.cols;//总的像素数

     int Max_threshold = 0;
     for (int i = 0; i < 256; ++i){//逐个像素进行判断
        float HO = 0.0; //前景熵
        float HB = 0.0; //背景熵
         //计算前景像素数
         int frontpix = 0;
         for (int j = 0; j < i; ++j){
             frontpix += lHistogram[j];
         }
        //计算前景熵
        for (int j = 0; j < i; ++j){
            if (lHistogram[j] != 0){
                 probability = (float)lHistogram[j] / frontpix;
                HO = HO + probability*log(1/probability);
             }
        }
         //计算背景熵
         for (int k = i; k < 256; ++k){
            if (lHistogram[k] != 0){
                 probability = (float)lHistogram[k] / (totalpix - frontpix);
                 HB = HB + probability*log(1/probability);
             }
         }
         //计算最大熵
        if(HO + HB > max_Entropy){
             max_Entropy = HO + HB;
             Max_threshold = i + 8;
         }
     }
     cout << "第 " << max_entropy << " 个关键帧的最大熵法的阈值为  " << Max_threshold << endl;
     max_entropy++;
     //波谷的灰度值(阈值)为 K.grey	
     Mat after3;
     threshold(frame, after3, Max_threshold, 255, THRESH_BINARY);////阈值分割
     return after3;

}

//自适应阈值分割
Mat RegionSegAdaptive(Mat frame)
{
    int nLocAvg;
    //左上，分成4部分
    nLocAvg = 0;
    for (int i = 0; i < frame.rows/2; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
        for (int j = 0; j < frame.cols/2; j++)
        {
            nLocAvg += data[j];
        }
    }
    nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));//求得每一个区域的平均阈值
    for (int i = 0; i < frame.rows/2; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
        for (int j = 0; j < frame.cols/2; j++)
        {
            if (data[j] < nLocAvg)//阈值分割
                data[j] = 0;
            else 
                data[j] = 255;
        }
    }
//左下
    nLocAvg = 0;
    for (int i = frame.rows/2; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
        for (int j = 0; j < frame.cols/2; j++)
        {
            nLocAvg += data[j];
        }
    }
    nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));
    for (int i = frame.rows/2; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
        for (int j = 0; j < frame.cols/2; j++)
        {
            if (data[j] < nLocAvg)
                data[j] = 0;
            else
                data[j] = 255;
        }
    }

    //右上
    nLocAvg = 0;
    for (int i = 0; i < frame.rows/2; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
        for (int j = frame.cols / 2; j < frame.cols ; j++)
        {
            nLocAvg += data[j];
        }
    }
    nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));
    for (int i = 0; i < frame.rows/2; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
        for (int j = frame.cols / 2; j < frame.cols; j++)
        {
            if (data[j] < nLocAvg)
                data[j] = 0;
            else
                data[j] = 255;
        }
    }

    //右下
    nLocAvg = 0;
    for (int i = frame.rows / 2; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
        for (int j = frame.cols / 2; j < frame.cols; j++)
        {
            nLocAvg += data[j];
        }
    }
    nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));
    for (int i = frame.rows / 2; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
        for (int j = frame.cols / 2; j < frame.cols; j++)
        {
            if (data[j] < nLocAvg)
                data[j] = 0;
            else
                data[j] = 255;
        }
    }
    return frame;
}

//主函数
int main()
{
    cout << "-------------------------------------------" << endl;
    cout << "             选择阈值分割的方式            " << endl;
    cout << "-------------------------------------------" << endl;
    cout << "            1.分水岭法" << endl;
    cout << "            2.OTSU法" << endl;
    cout << "            3.迭代法" << endl;
    cout << "            4.双峰波谷法" << endl;
    cout << "            5.最大熵法" << endl;
    cout << "            6.自适应阈值分割法" << endl;
    cout << "请输入您的选择：（如：1）" << endl;
    int m;
    cin >> m;
    VideoCapture capture;
    String filename = "E:\\数字视频实验视频\\1实验一视频\\exp1(2).avi";
    capture.open(filename);//读取视频
    if (!capture.isOpened())//
    {
        cout << "无法打开视频文件！" << endl;
        return -1;
    }
    long totalFrameNumber = capture.get(CAP_PROP_FRAME_COUNT);//求总的视频帧数
    Mat frame_key;//关键帧
    capture >> frame_key;//视频的第一帧首先作为关键帧
    imshow("关键帧", frame_key);//展示关键帧
    waitKey(10);
    Mat frame_key_gray;
    cvtColor(frame_key, frame_key_gray, COLOR_BGR2GRAY);//将关键帧灰度化
    cout << "整个视频共" << totalFrameNumber << "帧" << endl;
    cout << "正在写第1帧" << endl;
    /*stringstream str;
    str << "D:\\1.jpg";
    imwrite(str.str(), frame_key);*/
    switch (m)
    {
    case 1: 
        {  
            Mat Segmentation3 = WatershedSegmentation(frame_key, frame_key_gray);
            imshow("分水岭阈值分割", Segmentation3);
            waitKey(10); 
        }
        break;
    case 2: 
        {
            Mat Segmentation2 = OTSU(frame_key_gray);
            imshow("OTSU阈值分割", Segmentation2);
            waitKey(10); 
        }
        break;
    case 3: 
        {
            Mat Segmentation1 = iterative(frame_key_gray);
            imshow("迭代法阈值分割", Segmentation1);
            waitKey(10);  
        }
        break;
    case 4:  
        { 
            Mat Segmentation4 = TwoPeak(frame_key_gray);
            imshow("双峰波谷法阈值分割", Segmentation4);
            waitKey(10); 
        }
        break;
    case 5:
        { 
            Mat Segmentation5 = Max_Entropy(frame_key_gray);
            imshow("最大熵法阈值分割", Segmentation5);
            waitKey(10); 
        }
        break;
    case 6:
        {
            Mat Segmentation6 = RegionSegAdaptive(frame_key_gray);
            imshow("自适应阈值分割", Segmentation6);
            waitKey(10); 
        }
        break;
    default:
        break;
    }
   // int threhold=5800000;//阈值，经验值
    int framecount = 0;//记录当前为第几帧
    capture.set(CAP_PROP_FRAME_COUNT, 2);//设置帧从第二帧开始
    while (1)
    {
        Mat frame;//当前帧，每次都需要初始化
        capture >> frame;//从视频中读入一帧图像
        if (frame.empty())  //视频播放完成退出
        {
            break;
        }
        imshow("视频", frame);
        waitKey(10);
        framecount++;
        Mat current_frame=frame;//当前帧
        Mat previous_frame = frame_key;//上一关键帧
        Mat diff;//两帧之间的差值图像
        Mat current_framegray;//当前帧灰度化的结果
        Mat previous_framegray;//上一关键帧灰度化的结果
        cvtColor(current_frame, current_framegray, COLOR_BGR2GRAY);//将当前帧灰度化
        cvtColor(previous_frame, previous_framegray, COLOR_BGR2GRAY);//将前一关键帧灰度化
        absdiff(current_framegray, previous_framegray, diff);//帧差法求两帧之间的差值图像
        int delta= 0;
        float num = 0;
        float counter = 0;//计数差值满足要求(>15)的像素个数
        for (int i = 0; i < diff.rows; i++)//遍历求得差值图像的灰度差值之和为delta
        {
            uchar* data = diff.ptr<uchar>(i); //获取每一行的指针
            for (int j = 0; j < diff.cols; j++)
            {
                num++;
                if (data[j] > 15)
                {
                    counter = counter + 1;
                }
            }
        }
        float p = counter / num;
        //cout << "两帧之差的结果为" << delta << endl;
        //if (delta > threhold)
        if(p>=0.55)
        {   
            frame_key = frame;//将当前帧作为关键帧
            Mat frame_key_gray;
            cvtColor(frame_key, frame_key_gray, COLOR_BGR2GRAY);//将关键帧灰度化
            imshow("关键帧", frame_key);
            waitKey(10);

            cout << "正在写第" << framecount << "帧" << endl;
            /*stringstream str;
            str << "D:\\" << framecount << ".jpg";
            imwrite(str.str(), frame_key);*/
            if (m == 1)
            {
                Mat Segmentation3 = WatershedSegmentation(frame_key, frame_key_gray);
                imshow("分水岭阈值分割", Segmentation3);
                waitKey(10);
               
            }
            if (m == 2)
            {
                Mat Segmentation2 = OTSU(frame_key_gray);
                imshow("OTSU阈值分割", Segmentation2);
                waitKey(10);
            }

            if (m == 3)
            {
                Mat Segmentation1 = iterative(frame_key_gray);
                imshow("迭代法阈值分割", Segmentation1);
                waitKey(10);
            } 

            if (m == 4)
            {
                Mat Segmentation4 = TwoPeak(frame_key_gray);
                imshow("双峰波谷法阈值分割", Segmentation4);
                waitKey(10);
            }

            if (m == 5)
            {
                Mat Segmentation5 = Max_Entropy(frame_key_gray);
                imshow("最大熵法阈值分割", Segmentation5);
                waitKey(10);
            }

            if (m == 6)
            {
                Mat Segmentation6 = RegionSegAdaptive(frame_key_gray);
                imshow("自适应阈值分割", Segmentation6);
                waitKey(10);
            }
        }
    }

}



