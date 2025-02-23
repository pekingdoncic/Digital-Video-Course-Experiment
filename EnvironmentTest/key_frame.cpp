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

//���������ɫ
Vec3b ranColor(int value)//���������ɫ
{
    value = value % 255;  //����0~255�������
    RNG rng;//�����������
    int r = rng.uniform(0, value);
    int g = rng.uniform(0, value);
    int b = rng.uniform(0, value);
    return Vec3b(r, g, b);
}

//������������ֵ�ָ�
Mat iterative(Mat frame)
{
    int iMaxGrayValue = 0, iMinGrayValue = 255;//��ȡ���Ҷ�ֵ����С�Ҷ�ֵ
    long hist[256] = { 0 };//ֱ��ͼ����
    //���ֱ��ͼ
    for (int i = 0; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
        for (int j = 0; j < frame.cols; j++)
        {
            hist[int(data[j])]++;
            if (iMinGrayValue > data[j])
                iMinGrayValue = data[j];
            if (iMaxGrayValue < data[j])
                iMaxGrayValue = data[j];
        }
    }
    int iThreshold = 0;///��ֵ
    int iNewThreshold= (iMinGrayValue + iMaxGrayValue) / 2;//��ʼ��ֵ

    long totalValue_f = 0;//����ǰһ����ĻҶ�ֵ��
    long meanValue_f;//����ǰһ����ĻҶ�ƽ��

    long totalValue_b = 0;//�����һ����ĻҶ�ֵ��
    long meanValue_b;//�����һ����ĻҶ�ƽ��
  
    long lp1 = 0,lp2=0;//���ڼ�������Ҷ�ƽ��ֵ���м����
  
    for (int iIterationTimes = 0; abs(iThreshold - iNewThreshold) > 2 && iIterationTimes < 100; iIterationTimes++)//������100��
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
        iNewThreshold = (meanValue_f + meanValue_b) / 2;//����ֵ
    }
    cout << "�� "<<iter << " ���ؼ�֡�ĵ���������ֵΪ  " << iNewThreshold << endl;
    iter++;
    Mat after1;
    threshold(frame, after1, iNewThreshold, 255, THRESH_BINARY);//��ֵ�ָ�
    return after1;
}

//OTSU����ֵ�ָ�
Mat OTSU(Mat frame)
{
    //ѭ������
    long i, j, t;
    //��ֵ�����Ҷ�ֵ����С�Ҷ�ֵ�����������ƽ���Ҷ�ֵ
    int iThreshold, iNewThreshold, iMaxGrayValue, iMinGrayValue, iMean1GrayValue, iMean2GrayValue;
    iMaxGrayValue = 0;
    iMinGrayValue = 255;
    //ǰ������ռͼ���������������ռͼ�����
    double w0, w1;
    //����
    double G = 0, tempG = 0;
    //���ڼ�������Ҷ�ƽ��ֵ���м����
    long lP1, lP2, lS1, lS2;
    long lHistogram[256] = { 0 };
    //���ֱ��ͼ
    for (int i = 0; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
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
        //����������ĻҶ�ƽ��ֵ
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
        if (G > tempG)//���ʹGֵ����t
        {
            tempG = G;
            iThreshold = iNewThreshold;
        }
    }
    cout << "�� " << otsu << " ���ؼ�֡��OTSU������ֵΪ  " << iThreshold << endl;
    otsu++;
    Mat after2 ;
    threshold(frame, after2, iThreshold, 255, THRESH_BINARY);////��ֵ�ָ�
    return after2;
}

//˫�岨�ȷ���ֵ�ָ�
Mat TwoPeak(Mat frame) {
    typedef struct Peak
    {
        int grey;
        int num;
    }Peak;
    unsigned char* p;
    int histarr[256];
    //���ֱ��ͼ
    for (int i = 0; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
        for (int j = 0; j < frame.cols; j++)
        {
            histarr[int(data[j])]++;
        }
    }

    Peak peaks[256];//����ͳ�Ʋ���
    int cnt = 0;//����
    peaks[cnt].grey = 0;//peaks[i]�ĻҶ�ֵ���ȴ�0
    peaks[cnt].num = histarr[0];//peaks[i]��ͳ����
    cnt++;//cnt=1
    for (int i = 1; i < 255; i++)
    {
        if (histarr[i] > histarr[i + 1] && histarr[i] > histarr[i - 1])
        {
            //����
            peaks[cnt].grey = i;
            peaks[cnt].num = histarr[i];
            cnt++;
        }
    }
    peaks[cnt].grey = 255;
    peaks[cnt].num = histarr[255];
    cnt++;//������
    Peak max = peaks[0], snd = peaks[0];//Ѱ�����͵ڶ���Ĳ���
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

    int i = (max.grey > snd.grey ? snd.grey : max.grey);//��С�Ŀ�ʼ
    int mnum = (max.grey > snd.grey ? max.grey : snd.grey);
    Peak K = max;
    for (; i < mnum; i++)//����������֮��Ĳ���
    {
        if (histarr[i] < K.num)
        {
            K.num = histarr[i];
            K.grey = i;
        }
    }
    cout << "�� " << twopeak << " ���ؼ�֡��˫�岨�ȷ�����ֵΪ  " << K.grey << endl;
    twopeak++;
    //���ȵĻҶ�ֵ(��ֵ)Ϊ K.grey	
    Mat after3;
    threshold(frame, after3, K.grey, 255, THRESH_BINARY);////��ֵ�ָ�
    return after3;
}

//��ˮ���㷨
Mat WatershedSegmentation(Mat frame, Mat frame_gray)
{
    //----1.����ͼ��
    GaussianBlur(frame_gray, frame_gray, Size(5, 5), 2);   //���Ҷ�ͼ���и�˹�˲���������ȷָ�
    Canny(frame_gray, frame_gray, 80, 150);//���Ҷ�ͼ����canny�ı�Ե��⣬ȷ�����ӵĴ�ű߽�

    //----2.��������
    //����findCounter��������������
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(frame_gray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());//���Ҵ����ĻҶ�ͼ���������hierarchy�������
    //CV_CHAIN_APPROX_SIMPLE �����������Ĺյ���Ϣ�������������յ㴦�ĵ㱣����contours�����ڣ��յ���յ�֮��ֱ�߶��ϵ���Ϣ�㲻�豣��

    //---3.�������
    Mat mask(frame.rows, frame.cols, CV_32SC1);//��ʼ���������
    mask = Scalar::all(0);//��ÿ��ͨ������ֵ0
    int index = 0;//���ڼ�������
    int compCount = 0;//���ڸı���ɫ
    for (; index >= 0; index = hierarchy[index][0], compCount++)//ÿ���Һ�һ������
    {
        //��marks���б�ǣ��Բ�ͬ������������б�ţ��൱������עˮ�㣬�ж������������ж���עˮ��
        drawContours(mask, contours, index, Scalar::all(compCount + 80), 1, 8, hierarchy);//�������
        //1Ϊ�������߿�
    }
    //imshow("markers", mask * 1000);//չʾ�߽�����

    //---4.watershed
    watershed(frame, mask);//ʹ��opencv�Ŀ⺯��������ǰ�õ�������ͼ����עˮ
    //---5.��ɫ���
    //��ÿһ�����������ɫ���
    Mat afterFilled = Mat::zeros(frame.size(), CV_8UC3);
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            int index = mask.at<int>(i, j);//����ı�Ƿ���
            if (mask.at<int>(i, j) == -1)//����������֮��ķֽ紦��ֵ����Ϊ��-1������������
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

//����ؽ�����ֵ�ָ�
Mat Max_Entropy(Mat frame)
{
    long lHistogram[256] = { 0 };
    //���ֱ��ͼ
    for (int i = 0; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
        for (int j = 0; j < frame.cols; j++)
        {
            if(data[j]==0)
                continue; //�ų�����ɫ�����ص�
            lHistogram[int(data[j])]++;
        }
    }
    
     float probability = 0.0; //����
     float max_Entropy = 0.0; //�����
     int totalpix= frame.rows * frame.cols;//�ܵ�������

     int Max_threshold = 0;
     for (int i = 0; i < 256; ++i){//������ؽ����ж�
        float HO = 0.0; //ǰ����
        float HB = 0.0; //������
         //����ǰ��������
         int frontpix = 0;
         for (int j = 0; j < i; ++j){
             frontpix += lHistogram[j];
         }
        //����ǰ����
        for (int j = 0; j < i; ++j){
            if (lHistogram[j] != 0){
                 probability = (float)lHistogram[j] / frontpix;
                HO = HO + probability*log(1/probability);
             }
        }
         //���㱳����
         for (int k = i; k < 256; ++k){
            if (lHistogram[k] != 0){
                 probability = (float)lHistogram[k] / (totalpix - frontpix);
                 HB = HB + probability*log(1/probability);
             }
         }
         //���������
        if(HO + HB > max_Entropy){
             max_Entropy = HO + HB;
             Max_threshold = i + 8;
         }
     }
     cout << "�� " << max_entropy << " ���ؼ�֡������ط�����ֵΪ  " << Max_threshold << endl;
     max_entropy++;
     //���ȵĻҶ�ֵ(��ֵ)Ϊ K.grey	
     Mat after3;
     threshold(frame, after3, Max_threshold, 255, THRESH_BINARY);////��ֵ�ָ�
     return after3;

}

//����Ӧ��ֵ�ָ�
Mat RegionSegAdaptive(Mat frame)
{
    int nLocAvg;
    //���ϣ��ֳ�4����
    nLocAvg = 0;
    for (int i = 0; i < frame.rows/2; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
        for (int j = 0; j < frame.cols/2; j++)
        {
            nLocAvg += data[j];
        }
    }
    nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));//���ÿһ�������ƽ����ֵ
    for (int i = 0; i < frame.rows/2; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
        for (int j = 0; j < frame.cols/2; j++)
        {
            if (data[j] < nLocAvg)//��ֵ�ָ�
                data[j] = 0;
            else 
                data[j] = 255;
        }
    }
//����
    nLocAvg = 0;
    for (int i = frame.rows/2; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
        for (int j = 0; j < frame.cols/2; j++)
        {
            nLocAvg += data[j];
        }
    }
    nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));
    for (int i = frame.rows/2; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
        for (int j = 0; j < frame.cols/2; j++)
        {
            if (data[j] < nLocAvg)
                data[j] = 0;
            else
                data[j] = 255;
        }
    }

    //����
    nLocAvg = 0;
    for (int i = 0; i < frame.rows/2; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
        for (int j = frame.cols / 2; j < frame.cols ; j++)
        {
            nLocAvg += data[j];
        }
    }
    nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));
    for (int i = 0; i < frame.rows/2; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
        for (int j = frame.cols / 2; j < frame.cols; j++)
        {
            if (data[j] < nLocAvg)
                data[j] = 0;
            else
                data[j] = 255;
        }
    }

    //����
    nLocAvg = 0;
    for (int i = frame.rows / 2; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
        for (int j = frame.cols / 2; j < frame.cols; j++)
        {
            nLocAvg += data[j];
        }
    }
    nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));
    for (int i = frame.rows / 2; i < frame.rows; i++)
    {
        uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
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

//������
int main()
{
    cout << "-------------------------------------------" << endl;
    cout << "             ѡ����ֵ�ָ�ķ�ʽ            " << endl;
    cout << "-------------------------------------------" << endl;
    cout << "            1.��ˮ�뷨" << endl;
    cout << "            2.OTSU��" << endl;
    cout << "            3.������" << endl;
    cout << "            4.˫�岨�ȷ�" << endl;
    cout << "            5.����ط�" << endl;
    cout << "            6.����Ӧ��ֵ�ָ" << endl;
    cout << "����������ѡ�񣺣��磺1��" << endl;
    int m;
    cin >> m;
    VideoCapture capture;
    String filename = "E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1(2).avi";
    capture.open(filename);//��ȡ��Ƶ
    if (!capture.isOpened())//
    {
        cout << "�޷�����Ƶ�ļ���" << endl;
        return -1;
    }
    long totalFrameNumber = capture.get(CAP_PROP_FRAME_COUNT);//���ܵ���Ƶ֡��
    Mat frame_key;//�ؼ�֡
    capture >> frame_key;//��Ƶ�ĵ�һ֡������Ϊ�ؼ�֡
    imshow("�ؼ�֡", frame_key);//չʾ�ؼ�֡
    waitKey(10);
    Mat frame_key_gray;
    cvtColor(frame_key, frame_key_gray, COLOR_BGR2GRAY);//���ؼ�֡�ҶȻ�
    cout << "������Ƶ��" << totalFrameNumber << "֡" << endl;
    cout << "����д��1֡" << endl;
    /*stringstream str;
    str << "D:\\1.jpg";
    imwrite(str.str(), frame_key);*/
    switch (m)
    {
    case 1: 
        {  
            Mat Segmentation3 = WatershedSegmentation(frame_key, frame_key_gray);
            imshow("��ˮ����ֵ�ָ�", Segmentation3);
            waitKey(10); 
        }
        break;
    case 2: 
        {
            Mat Segmentation2 = OTSU(frame_key_gray);
            imshow("OTSU��ֵ�ָ�", Segmentation2);
            waitKey(10); 
        }
        break;
    case 3: 
        {
            Mat Segmentation1 = iterative(frame_key_gray);
            imshow("��������ֵ�ָ�", Segmentation1);
            waitKey(10);  
        }
        break;
    case 4:  
        { 
            Mat Segmentation4 = TwoPeak(frame_key_gray);
            imshow("˫�岨�ȷ���ֵ�ָ�", Segmentation4);
            waitKey(10); 
        }
        break;
    case 5:
        { 
            Mat Segmentation5 = Max_Entropy(frame_key_gray);
            imshow("����ط���ֵ�ָ�", Segmentation5);
            waitKey(10); 
        }
        break;
    case 6:
        {
            Mat Segmentation6 = RegionSegAdaptive(frame_key_gray);
            imshow("����Ӧ��ֵ�ָ�", Segmentation6);
            waitKey(10); 
        }
        break;
    default:
        break;
    }
   // int threhold=5800000;//��ֵ������ֵ
    int framecount = 0;//��¼��ǰΪ�ڼ�֡
    capture.set(CAP_PROP_FRAME_COUNT, 2);//����֡�ӵڶ�֡��ʼ
    while (1)
    {
        Mat frame;//��ǰ֡��ÿ�ζ���Ҫ��ʼ��
        capture >> frame;//����Ƶ�ж���һ֡ͼ��
        if (frame.empty())  //��Ƶ��������˳�
        {
            break;
        }
        imshow("��Ƶ", frame);
        waitKey(10);
        framecount++;
        Mat current_frame=frame;//��ǰ֡
        Mat previous_frame = frame_key;//��һ�ؼ�֡
        Mat diff;//��֮֡��Ĳ�ֵͼ��
        Mat current_framegray;//��ǰ֡�ҶȻ��Ľ��
        Mat previous_framegray;//��һ�ؼ�֡�ҶȻ��Ľ��
        cvtColor(current_frame, current_framegray, COLOR_BGR2GRAY);//����ǰ֡�ҶȻ�
        cvtColor(previous_frame, previous_framegray, COLOR_BGR2GRAY);//��ǰһ�ؼ�֡�ҶȻ�
        absdiff(current_framegray, previous_framegray, diff);//֡�����֮֡��Ĳ�ֵͼ��
        int delta= 0;
        float num = 0;
        float counter = 0;//������ֵ����Ҫ��(>15)�����ظ���
        for (int i = 0; i < diff.rows; i++)//������ò�ֵͼ��ĻҶȲ�ֵ֮��Ϊdelta
        {
            uchar* data = diff.ptr<uchar>(i); //��ȡÿһ�е�ָ��
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
        //cout << "��֮֡��Ľ��Ϊ" << delta << endl;
        //if (delta > threhold)
        if(p>=0.55)
        {   
            frame_key = frame;//����ǰ֡��Ϊ�ؼ�֡
            Mat frame_key_gray;
            cvtColor(frame_key, frame_key_gray, COLOR_BGR2GRAY);//���ؼ�֡�ҶȻ�
            imshow("�ؼ�֡", frame_key);
            waitKey(10);

            cout << "����д��" << framecount << "֡" << endl;
            /*stringstream str;
            str << "D:\\" << framecount << ".jpg";
            imwrite(str.str(), frame_key);*/
            if (m == 1)
            {
                Mat Segmentation3 = WatershedSegmentation(frame_key, frame_key_gray);
                imshow("��ˮ����ֵ�ָ�", Segmentation3);
                waitKey(10);
               
            }
            if (m == 2)
            {
                Mat Segmentation2 = OTSU(frame_key_gray);
                imshow("OTSU��ֵ�ָ�", Segmentation2);
                waitKey(10);
            }

            if (m == 3)
            {
                Mat Segmentation1 = iterative(frame_key_gray);
                imshow("��������ֵ�ָ�", Segmentation1);
                waitKey(10);
            } 

            if (m == 4)
            {
                Mat Segmentation4 = TwoPeak(frame_key_gray);
                imshow("˫�岨�ȷ���ֵ�ָ�", Segmentation4);
                waitKey(10);
            }

            if (m == 5)
            {
                Mat Segmentation5 = Max_Entropy(frame_key_gray);
                imshow("����ط���ֵ�ָ�", Segmentation5);
                waitKey(10);
            }

            if (m == 6)
            {
                Mat Segmentation6 = RegionSegAdaptive(frame_key_gray);
                imshow("����Ӧ��ֵ�ָ�", Segmentation6);
                waitKey(10);
            }
        }
    }

}



