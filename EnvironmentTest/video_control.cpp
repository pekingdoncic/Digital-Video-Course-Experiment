//实验二 控制视频播放和滤波
#include "stdio.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>


using namespace std;
using namespace cv;


Mat frame;//当前帧

void ChangeBrightness(int brightnessValue, Mat frame);//改变图像的亮度
void AdjustContrastRatio(int contrastRatioValue, Mat frame);//提高对比度

void HistEqual(Mat frame);//直方图均衡化

void MeanFilter(Mat frame, int filterN);//均值滤波
void GaussFilter(Mat frame, int filterN);//高斯滤波
void MidFilter(Mat frame, int filterN);//中值滤波

void lixiangditong(Mat frame);//理想低通滤波
void BLPF(Mat frame);//巴特沃斯低通滤波
void GausslowFliter(Mat image);//高斯低通滤波

int trackValue = 0; // 记录滑块的位置，即控制条数值
int brightnessValue = 255; // 亮度变化值,初始化为255时为原来图像
int contrastRatioValue = 10; // 对比度，初始化为10时为原来图像
int equalHistValue = 0; // 直方图是否均衡化，为1表示进行直方图均衡化
int meanFilterValue = 1;//是否进行均值滤波
int fangkuangFilterValue = 1;//方框滤波初始值
int gaussFilterValue = 0;//
int paused = 0;  // 是否暂停
int framePos = 0;//记录视频帧位置
int MidFilterValue = 1;//中值滤波
int lixiangtditongFilterValue = 0;//理想低通滤波初始值
int BLPFValue = 0;//是否进行巴特沃斯低通滤波
int GausslowValue = 0;//是否进行高斯低通滤波

 // 控制进度条位置的回调函数
void ControlVideo(int, void* param)
{
    VideoCapture cap = *(VideoCapture*)param;//获取回调函数定义的对象
    cap.set(CAP_PROP_POS_FRAMES, trackValue);//通过滑动块位置，来设置视频帧位置
}


int main()
{
    VideoCapture video;
    String filename = "F:\\学习\\数字视频处理\\EnvironmentTest\\Videos\\exp3.avi";
    video.open(filename);
    if (!video.isOpened())
    {
        cout << "无法打开视频文件！" << endl;
        return -1;
    }
    namedWindow("video", 0);//创建窗口
    resizeWindow("video", 500, 500); //创建一个500*500大小的窗口
    int frameallNums = video.get(CAP_PROP_FRAME_COUNT); //视频总帧数
    createTrackbar("进度条", "video", &trackValue, frameallNums, ControlVideo, &video); //视频进度条控制
    createTrackbar("pause", "video", &paused, 1); //视频播放暂停控制
    createTrackbar("亮度", "video", &brightnessValue, 510); //视频亮度
    createTrackbar("对比度", "video", &contrastRatioValue, 20); //视频对比度
    createTrackbar("直方图均衡化", "video", &equalHistValue, 1); //视频直方图均衡化
    createTrackbar("均值滤波", "video", &meanFilterValue, 5); //视频均值滤波
    createTrackbar("高斯滤波", "video", &gaussFilterValue, 1); //视频高斯滤波
    createTrackbar("中值滤波", "video", &MidFilterValue, 5); //视频中值滤波
    createTrackbar("方框滤波", "video", &fangkuangFilterValue, 3); //视频方框滤波
    createTrackbar("理想低通滤波", "video", &lixiangtditongFilterValue, 1); //理想低通滤波
    createTrackbar("高斯低通滤波", "video", &GausslowValue, 1); //高斯低通滤波
    createTrackbar("巴特沃斯低通滤波", "video", &BLPFValue, 1); //巴特沃斯低通滤波
    
    while (1)
    {
        framePos = video.get(CAP_PROP_POS_FRAMES);//获取视频帧位置
        setTrackbarPos("进度条", "video", framePos);//通过视频帧位置，来更新滑动条位置
        if (paused == 0)//播放
        {
            //实现循环播放
            if (framePos == int(video.get(CAP_PROP_FRAME_COUNT)))//如果读到的帧是最后一帧
            {
                framePos = 0;//帧数置为0
                video.set(CAP_PROP_POS_FRAMES, 0);//将视频从0开始读
            }
            video >> frame;//取帧给frame
            if (frame.empty())//取帧失败
            {
                break;
            }   
        }
        if (contrastRatioValue != 10)//如果移动了对比度值
            AdjustContrastRatio(contrastRatioValue, frame);
        if (brightnessValue != 255)//如果移动了亮度值      
            ChangeBrightness(brightnessValue, frame);
        if (meanFilterValue != 1)
            MeanFilter(frame, meanFilterValue);//均值滤波
        if (gaussFilterValue != 0)
            GaussFilter(frame, 3);//高斯滤波
        if (fangkuangFilterValue != 1)//方框滤波
        {
            if (fangkuangFilterValue == 0)
                return 0;
            boxFilter(frame, frame, -1, Size(fangkuangFilterValue, fangkuangFilterValue), Point(-1, -1), 0, 4);
        
        }
            
        if (MidFilterValue != 1)//如果选择了中值滤波
        {
            if (MidFilterValue % 2 == 0)//如果是偶数，则加1变为奇数
                MidFilterValue += 1;//2,3->3  4,5->5
            medianBlur(frame, frame, MidFilterValue);
        }
        if (BLPFValue != 0)//如果选择了巴特沃斯低通滤波
            BLPF(frame);
        if (GausslowValue != 0)//如果选择了高斯低通滤波
            GausslowFliter(frame);
        if (lixiangtditongFilterValue == 1)//如果选择了理想低通滤波
            lixiangditong(frame);
        if (equalHistValue !=0)//如果选择了直方图均衡化
            HistEqual(frame);//进行均衡化
        else
            imshow("视频", frame);
        waitKey(10);
        char c = waitKey(3);
        if (c == 27)
            break;//按esc键退出
    }
    video.release();
}

//改变帧图像的亮度
void ChangeBrightness(int brightnessValue, Mat frame)
{
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            float b = frame.at<Vec3b>(i, j)[0];
            float g = frame.at<Vec3b>(i, j)[1];
            float r = frame.at<Vec3b>(i, j)[2];
            frame.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(b +     brightnessValue-255);
            frame.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(g + brightnessValue-255);
            frame.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(r + brightnessValue-255);
        }
    }
}

//改变图片的对比度
void AdjustContrastRatio(int contrastRatioValue, Mat frame)//改变图像的对比度
{
    for (int i = 0; i < frame.rows; i++)//彩色图像要对RGB三个通道的值进行改变
    {
        for (int j = 0; j < frame.cols; j++)
        {
            float b = frame.at<Vec3b>(i, j)[0];//获取三个通道的值
            float g = frame.at<Vec3b>(i, j)[1];
            float r = frame.at<Vec3b>(i, j)[2];
            if (contrastRatioValue >= 10)//对比度的值大于10，则修改像素值，b,g,r同×一个系数,系数值为（x-9)
            {
                frame.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(b * (contrastRatioValue - 9));
                frame.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(g * (contrastRatioValue - 9));
                frame.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(r * (contrastRatioValue - 9));
            }
            else//对比度的值大于10，则修改像素值，b,g,r同×一个系数，系数值为 1/（-x+11)
            {
                frame.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(b * float(1.0) / (-contrastRatioValue + 11));
                frame.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(g * float(1.0) / (-contrastRatioValue + 11));
                frame.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(r * float(1.0) / (-contrastRatioValue + 11));
            }
        }
    }
}

//直方图均衡化
void HistEqual(Mat frame)
{
    // 将彩色图像拆分成通道
    vector<Mat> channels;
    split(frame, channels);

    for (int c = 0; c < 3; c++) // 遍历每个颜色通道（0：蓝色，1：绿色，2：红色）
    {
        // 将当前通道转换为灰度图像
        Mat channel = channels[c];
        
        // 计算直方图分布
        int hist[256] = { 0 };
        for (int i = 0; i < channel.rows; i++)
        {
            for (int j = 0; j < channel.cols; j++)
            {
                hist[int(channel.at<uchar>(i, j))]++;
            }
        }

        // 计算累积直方图
        long hist_all[256] = { 0 };
        int bMap[256] = { 0 };
        for (int i = 0; i < 256; i++)
        {
            if (i == 0)
            {
                hist_all[i] = hist[i];
            }
            else
            {
                hist_all[i] = hist_all[i - 1] + hist[i];
            }
            // 计算新的强度值
            bMap[i] = int(double(hist_all[i]) / (channel.cols * channel.rows) * 255 + 0.5);
        }

        // 将直方图均衡化应用于当前通道
        for (int i = 0; i < channel.rows; i++)
        {
            for (int j = 0; j < channel.cols; j++)
            {
                channel.at<uchar>(i, j) = bMap[int(channel.at<uchar>(i, j))];
            }
        }

        // 更新修改后的通道
        channels[c] = channel;
    }

    // 将修改后的通道合并回彩色图像
    merge(channels, frame);

    // 显示结果彩色图像
    imshow("视频", frame);
}


//均值滤波,filterN为滤波模板的宽度
void MeanFilter(Mat frame, int filterN)
{
    if(filterN == 0)
    {
        return;
    }
            if (filterN % 2 == 0) {//如果是偶数，滤波核的边长减1
                filterN = filterN - 1;
            }
            else {
                filterN = filterN;
            }
            //卷积核的一半
            int aheight = (filterN - 1) / 2;
            int awidth = (filterN - 1) / 2;

            for (int i = aheight; i < frame.rows - aheight - 1; i++)
            {
                for (int j = awidth; j < frame.cols - awidth - 1; j++)
                {
                    int sum[3] = { 0 };
                    int mean[3] = { 0 };
                    for (int k = i - aheight; k <= i + aheight; k++)
                    {
                        for (int l = j - awidth; l <= j + awidth; l++)
                        {
                            //三个通道分别求总和
                            sum[0] += frame.at<Vec3b>(k, l)[0];
                            sum[1] += frame.at<Vec3b>(k, l)[1];
                            sum[2] += frame.at<Vec3b>(k, l)[2];
                        }
                    }
                    //分别求均值
                    for (int m = 0; m < 3; m++) {
                        mean[m] = sum[m] / (filterN * filterN);
                        frame.at<Vec3b>(i, j)[m] = mean[m];
                        if (frame.at<Vec3b>(i, j)[m] > 255)
                            frame.at<Vec3b>(i, j)[m] = 255;
                        if (frame.at<Vec3b>(i, j)[m] < 0)
                            frame.at<Vec3b>(i, j)[m] = 0;
                    }
                }
            }
}

//高斯滤波
void GaussFilter(Mat frame, int filterN)//高斯滤波
{
    int a[3][3] = { 1,2,1,2,4,2,1,2,1 };//模板数组
    for (long i = 0; i < frame.rows; i++)
    {
        for (long j = 0; j < frame.cols; j++)
        {
            float tempb = 0;
            float tempg = 0;
            float tempr = 0;
            for (int m = 0; m < filterN; m++)
            {
                for (int n = 0; n < filterN; n++)
                {
                    int tempy = j - filterN / 2 + n, tempx = i - filterN / 2 + m;
                    tempx = tempx < 0 ? 0 : tempx;//限制tempx的范围
                    tempx = tempx > frame.rows - 1 ? frame.rows - 1 : tempx;
                    tempy = tempy < 0 ? 0 : tempy;//限制tempy的范围
                    tempy = tempy > frame.cols - 1 ? frame.cols - 1 : tempy;
                    float b = frame.at<Vec3b>(tempx, tempy)[0];
                    float g = frame.at<Vec3b>(tempx, tempy)[1];
                    float r = frame.at<Vec3b>(tempx, tempy)[2];
                    tempb += (b * a[m][n]);
                    tempg += (g * a[m][n]);
                    tempr += (r * a[m][n]);
                }
            }
            tempb = tempb / (16.0); tempg = tempg / (16.0); tempr = tempr / (16.0);
            frame.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(tempb);
            frame.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(tempg);
            frame.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(tempr);
        }
    }
}

//中值滤波
void MidFilter(Mat frame, int filterN)
{
    //Mat im, m1, m2, m3, m4, m5, m6, B, G, R;
    //vector <Mat> channels;
    //split(frame, channels);
    //B = channels.at(0);
    //G = channels.at(1);
    //R = channels.at(2);//通道分离
    //int a = frame.rows;
    //int b = frame.cols; //获得行和列
    //int k = filterN;
    //for (int i = k / 2; i < a - k / 2; i++)//遍历图像，注意要把边框部分去除了
    //{
    //    for (int j = k / 2; j < b - k / 2; j++)
    //    {
    //        m1 = B(Rect(j - k / 2, i - k / 2, k, k)).clone();//选取k*k窗口中的B通道
    //        m2 = G(Rect(j - k / 2, i - k / 2, k, k)).clone();//选取k*k窗口G通道
    //        m3 = R(Rect(j - k / 2, i - k / 2, k, k)).clone();//选取k*k窗口R通道
    //        m1 = m1.reshape(1, 1);
    //        m2 = m2.reshape(1, 1);
    //        m3 = m3.reshape(1, 1); //像素值将其变为一行
    //        cv::sort(m1, m4, 0);
    //        cv::sort(m2, m5, 0);
    //        cv::sort(m3, m6, 0);//对其排序
    //        int p0 = m4.at<uchar>(0, k * k / 2 + 1);
    //        int p1 = m5.at<uchar>(0, k * k / 2 + 1);
    //        int p2 = m6.at<uchar>(0, k * k / 2 + 1);//选取各通道的像素中值点
    //        frame.at<Vec3b>(i - k / 2, j - k / 2)[0] = p0;
    //        frame.at<Vec3b>(i - k / 2, j - k / 2)[1] = p1;
    //        frame.at<Vec3b>(i - k / 2, j - k / 2)[2] = p2;//对其进行重赋值
    //    }
    //}
    //   if (filterN % 2 == 0) {
    //    filterN = filterN - 1;
    //}
    //else {
    //    filterN = filterN;
    //}
    ////卷积核的一半
    //int aheight = (filterN - 1) / 2;
    //int awidth = (filterN - 1) / 2;

    //for (int i = aheight; i < frame.rows - aheight - 1; i++)
    //{
    //    for (int j = awidth; j < frame.cols - awidth - 1; j++)
    //    {
    //        int* tempr = new int[int(filterN)* int(filterN)];
    //        int* tempg = new int[int(filterN) * int(filterN)];
    //        int* tempb = new int[int(filterN) * int(filterN)];
    //        int s = 0;
    //        for (int k = i - aheight; k <= i + aheight; k++)
    //        {
    //            for (int l = j - awidth; l <= j + awidth; l++)
    //            {
    //                tempr[s] = frame.at<Vec3b>(k, l)[0];
    //                tempg[s] = frame.at<Vec3b>(k, l)[1];
    //                tempb[s] = frame.at<Vec3b>(k, l)[2];
    //                s++;
    //            }
    //        }
    //        int iFilterLen = filterN * filterN;
    //        // 循环变量
    //        int	m;
    //        int	n;
    //        // 中间变量
    //       int rTemp=0;
    //       int gTemp=0;
    //       int bTemp=0;
    //        // 用冒泡法对数组进行排序
    //        for (n = 0; n < iFilterLen - 1; n++)
    //        {
    //            for (m = 0; m < iFilterLen - j - 1; m++)
    //            {
    //                if (tempr[m] > tempr[m + 1])
    //                {
    //                    // 互换
    //                    rTemp = tempr[m];
    //                    tempr[m] = tempr[m + 1];
    //                    tempr[m + 1] = rTemp;
    //                }
    //                if (tempg[m] > tempg[m + 1])
    //                {
    //                    // 互换
    //                    gTemp = tempg[m];
    //                    tempg[m] = tempg[m + 1];
    //                    tempg[m + 1] = gTemp;
    //                }
    //                if (tempb[m] > tempb[m + 1])
    //                {
    //                    // 互换
    //                    bTemp = tempb[m];
    //                    tempb[m] = tempb[m + 1];
    //                    tempb[m + 1] = bTemp;
    //                }
    //            }
    //        }
    //        // 计算中值
    //        if ((iFilterLen & 1) > 0)
    //        {
    //            // 数组有奇数个元素，返回中间一个元素
    //            rTemp = tempr[(iFilterLen + 1) / 2];
    //            gTemp = tempg[(iFilterLen + 1) / 2];
    //            bTemp = tempb[(iFilterLen + 1) / 2];
    //        }
    //        else
    //        {
    //            // 数组有偶数个元素，返回中间两个元素平均值
    //            rTemp = (tempr[iFilterLen / 2] + tempr[iFilterLen / 2 + 1]) / 2;
    //            gTemp = (tempg[iFilterLen / 2] + tempg[iFilterLen / 2 + 1]) / 2;
    //            bTemp = (tempb[iFilterLen / 2] + tempb[iFilterLen / 2 + 1]) / 2;
    //        }

    //        // 获取中值
    //        frame.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(rTemp);
    //        frame.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(gTemp);
    //        frame.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(bTemp);
    //    }
    //}
            
     
}

//傅里叶变换得到频谱图和复数域结果
void My_DFT(Mat input_image, Mat& output_image, Mat& transform_image)
{
    //1.扩展图像矩阵，为2，3，5的倍数时运算速度快
    int m = getOptimalDFTSize(input_image.rows);
    int n = getOptimalDFTSize(input_image.cols);
    copyMakeBorder(input_image, input_image, 0, m - input_image.rows, 0, n - input_image.cols, BORDER_CONSTANT, Scalar::all(0));

    //2.创建一个双通道矩阵planes，用来储存复数的实部与虚部
    Mat planes[] = { Mat_<float>(input_image), Mat::zeros(input_image.size(), CV_32F) };

    //3.从多个单通道数组中创建一个多通道数组:transform_image。函数Merge将几个数组合并为一个多通道阵列，即输出数组的每个元素将是输入数组元素的级联
    merge(planes, 2, transform_image);

    //4.进行傅立叶变换
    dft(transform_image, transform_image);

    //5.计算复数的幅值，保存在output_image（频谱图）
    split(transform_image, planes); // 将双通道分为两个单通道，一个表示实部，一个表示虚部
    Mat transform_image_real = planes[0];
    Mat transform_image_imag = planes[1];

    magnitude(planes[0], planes[1], output_image); //计算复数的幅值，保存在output_image（频谱图）

    //6.前面得到的频谱图数级过大，不好显示，因此转换
    output_image += Scalar(1);   // 取对数前将所有的像素都加1，防止log0
    log(output_image, output_image);   // 取对数
    normalize(output_image, output_image, 0, 1, NORM_MINMAX); //归一化

    //7.剪切和重分布幅度图像限
    output_image = output_image(Rect(0, 0, output_image.cols & -2, output_image.rows & -2));
    imshow("output_image_1", output_image);
    // 重新排列傅里叶图像中的象限，使原点位于图像中心
    int cx = output_image.cols / 2;
    int cy = output_image.rows / 2;
    Mat q0(output_image, Rect(0, 0, cx, cy));   // 左上区域
    Mat q1(output_image, Rect(cx, 0, cx, cy));  // 右上区域
    Mat q2(output_image, Rect(0, cy, cx, cy));  // 左下区域
    Mat q3(output_image, Rect(cx, cy, cx, cy)); // 右下区域

      //交换象限中心化
    Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);//左上与右下进行交换
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);//右上与左下进行交换


    Mat q00(transform_image_real, Rect(0, 0, cx, cy));   // 左上区域
    Mat q01(transform_image_real, Rect(cx, 0, cx, cy));  // 右上区域
    Mat q02(transform_image_real, Rect(0, cy, cx, cy));  // 左下区域
    Mat q03(transform_image_real, Rect(cx, cy, cx, cy)); // 右下区域
    q00.copyTo(tmp); q03.copyTo(q00); tmp.copyTo(q03);//左上与右下进行交换
    q01.copyTo(tmp); q02.copyTo(q01); tmp.copyTo(q02);//右上与左下进行交换

    Mat q10(transform_image_imag, Rect(0, 0, cx, cy));   // 左上区域
    Mat q11(transform_image_imag, Rect(cx, 0, cx, cy));  // 右上区域
    Mat q12(transform_image_imag, Rect(0, cy, cx, cy));  // 左下区域
    Mat q13(transform_image_imag, Rect(cx, cy, cx, cy)); // 右下区域
    q10.copyTo(tmp); q13.copyTo(q10); tmp.copyTo(q13);//左上与右下进行交换
    q11.copyTo(tmp); q12.copyTo(q11); tmp.copyTo(q12);//右上与左下进行交换
    imshow("output_image_2", output_image);
    planes[0] = transform_image_real;
    planes[1] = transform_image_imag;
    merge(planes, 2, transform_image);//将傅里叶变换结果中心化
}

//理想低通滤波

void lixiangditong(Mat image)
{
    Mat image_gray, image_output, image_transform;   //定义输入图像，灰度图像，输出图像
    cvtColor(image, image_gray, COLOR_BGR2GRAY); //转换为灰度图

     //1、傅里叶变换，image_output为可显示的频谱图，image_transform为傅里叶变换的复数结果
    My_DFT(image_gray, image_output, image_transform);
     //imshow("image_output", image_output);
    
     //2、理想低通滤波
     Mat planes[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
     split(image_transform, planes);//分离通道，获取实部虚部
     Mat image_transform_real = planes[0];
     Mat image_transform_imag = planes[1];

     int core_x = image_transform_real.rows / 2;//频谱图中心坐标
     int core_y = image_transform_real.cols / 2;
     int r = 90;  //滤波半径
     for (int i = 0; i < image_transform_real.rows; i++)
     {
         for (int j = 0; j < image_transform_real.cols; j++)
            {
                //距离中心的距离大于设置半径r的点所在值设为0
                if (((i - core_x) * (i - core_x) + (j - core_y) * (j - core_y)) > r * r)
                {
                    image_transform_real.at<float>(i, j) = 0;
                    image_transform_imag.at<float>(i, j) = 0;
                }
            }
     }
     planes[0] = image_transform_real;
     planes[1] = image_transform_imag;
     Mat image_transform_ilpf;//定义理想低通滤波矩阵
     merge(planes, 2, image_transform_ilpf);

     //3、傅里叶逆变换
     Mat iDft[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
     idft(image_transform_ilpf, image_transform_ilpf);//傅立叶逆变换
     split(image_transform_ilpf, iDft);//分离通道，主要获取0通道
     magnitude(iDft[0], iDft[1], iDft[0]); //计算复数的幅值，保存在iDft[0]
     normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);//归一化处理
     frame = iDft[0];   
}

//巴特沃斯低通滤波
void BLPF(Mat image)
{
    Mat  image_gray, image_output, image_transform;   //定义输入图像，灰度图像，输出图像
    cvtColor(image, image_gray, COLOR_BGR2GRAY); //转换为灰度图
    //1、傅里叶变换，image_output为可显示的频谱图，image_transform为傅里叶变换的复数结果
    My_DFT(image_gray, image_output, image_transform);

    //2、巴特沃斯低通滤波
    Mat planes[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
    split(image_transform, planes);//分离通道，获取实部虚部
    Mat image_transform_real = planes[0];
    Mat image_transform_imag = planes[1];

    int core_x = image_transform_real.rows / 2;//频谱图中心坐标
    int core_y = image_transform_real.cols / 2;
    int r = 90;  //滤波半径
    float h;
    float n = 2; //巴特沃斯系数
    float D;  //距离中心距离
    for (int i = 0; i < image_transform_real.rows; i++)
    {
        for (int j = 0; j < image_transform_real.cols; j++)
        {
            D = (i - core_x) * (i - core_x) + (j - core_y) * (j - core_y);
            h = 1 / (1 + pow((D / (r * r)), n));
            image_transform_real.at<float>(i, j) = image_transform_real.at<float>(i, j) * h;
            image_transform_imag.at<float>(i, j) = image_transform_imag.at<float>(i, j) * h;

        }
    }
    planes[0] = image_transform_real;
    planes[1] = image_transform_imag;
    Mat image_transform_ilpf;//定义巴特沃斯低通滤波结果
    merge(planes, 2, image_transform_ilpf);

    //3、傅里叶逆变换
    Mat iDft[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
    idft(image_transform_ilpf, image_transform_ilpf);//傅立叶逆变换
    split(image_transform_ilpf, iDft);//分离通道，主要获取0通道
    magnitude(iDft[0], iDft[1], iDft[0]); //计算复数的幅值，保存在iDft[0]
    normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);//归一化处理
    frame = iDft[0];
}

//高斯低通滤波
void GausslowFliter(Mat image)
{
    Mat  image_gray, image_output, image_transform;   //定义输入图像，灰度图像，输出图像
    cvtColor(image, image_gray, COLOR_BGR2GRAY); //转换为灰度图
    //1、傅里叶变换，image_output为可显示的频谱图，image_transform为傅里叶变换的复数结果
    My_DFT(image_gray, image_output, image_transform);
    //2、高斯低通滤波
    Mat planes[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
    split(image_transform, planes);//分离通道，获取实部虚部
    Mat image_transform_real = planes[0];
    Mat image_transform_imag = planes[1];

    int core_x = image_transform_real.rows / 2;//频谱图中心坐标
    int core_y = image_transform_real.cols / 2;
    int r = 80;  //滤波半径
    float h;
    for (int i = 0; i < image_transform_real.rows; i++)
    {
        for (int j = 0; j < image_transform_real.cols; j++)
        {
            h = exp(-((i - core_x) * (i - core_x) + (j - core_y) * (j - core_y)) / (2 * r * r));
            image_transform_real.at<float>(i, j) = image_transform_real.at<float>(i, j) * h;
            image_transform_imag.at<float>(i, j) = image_transform_imag.at<float>(i, j) * h;

        }
    }
    planes[0] = image_transform_real;
    planes[1] = image_transform_imag;
    Mat image_transform_ilpf;//定义高斯低通滤波结果
    merge(planes, 2, image_transform_ilpf);

    //3、傅里叶逆变换
    Mat iDft[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
    idft(image_transform_ilpf, image_transform_ilpf);//傅立叶逆变换
    split(image_transform_ilpf, iDft);//分离通道，主要获取0通道
    magnitude(iDft[0], iDft[1], iDft[0]); //计算复数的幅值，保存在iDft[0]
    normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);//归一化处理

    frame = iDft[0];
}


