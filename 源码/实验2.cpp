//ʵ��Ŀ����
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
    //ǰ����֡ͼ����лҶȻ�
    cvtColor(currentframe, frame_gray, COLOR_BGR2GRAY);
    cvtColor(old, old_gray, COLOR_BGR2GRAY);
    Size size; size.height = size.width = 7;

    //��ǰ����֡�ĻҶ�ͼ�����Ԥ����
    GaussianBlur(frame_gray, frame_gray, size, 1);
    GaussianBlur(old_gray, old_gray, size, 1);

    //֡�
    Mat diff;
    absdiff(frame_gray, old_gray, diff);

    //��ֵͼ��Ķ�ֵ��
    threshold(diff, afterThresold, 0, 255, THRESH_OTSU);

    dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)));//����
    dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(8, 8)));
    erode(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//��ʴ
    //// ��ʴ����
    //erode(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

    ////// ���߿�����
    //morphologyEx(afterThresold, afterThresold, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    imshow("������֡֡�Ŀ����", afterThresold);
    return afterThresold;
}

Mat frameDiff3(Mat old, Mat currnetframe,Mat next_frame,Mat afterThresold)
{
    Mat frame_gray, old_gray, next_gray;
    //��֡ͼ����лҶȻ�
    cvtColor(currnetframe, frame_gray, COLOR_BGR2GRAY);
    cvtColor(old, old_gray, COLOR_BGR2GRAY);
    cvtColor(next_frame, next_gray, COLOR_BGR2GRAY);

    //��֡�ĻҶ�ͼ�����Ԥ����
    GaussianBlur(frame_gray, frame_gray, Size(5, 5), 1);
    GaussianBlur(old_gray, old_gray, Size(5, 5), 1);
    GaussianBlur(next_gray, next_gray, Size(5, 5), 1);

    //֡�
    Mat diff1, diff2;
    absdiff(frame_gray, old_gray, diff1);
    absdiff(frame_gray, next_gray, diff2);
    Mat afterThresold1, afterThresold2;

    //��ֵͼ��Ķ�ֵ��
    threshold(diff1, afterThresold1, 0, 255, THRESH_OTSU);
    threshold(diff2, afterThresold2, 0, 255, THRESH_OTSU);
    /* threshold(diff1, afterThresold1,60, 255, THRESH_BINARY);
     threshold(diff2, afterThresold2,60, 255, THRESH_BINARY);*/

     //��֡���
    add(afterThresold1, afterThresold2, afterThresold);
    //blur(afterThresold, afterThresold, Size(5, 5));
    //medianBlur(afterThresold, afterThresold, 3);
    // GaussianBlur(afterThresold, afterThresold, Size(5, 5), 0);
    dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(3, 5)));//����
    erode(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//��ʴ
    //dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));//����
    morphologyEx(afterThresold, afterThresold, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1)));     // ��̬ѧ����--������
    imshow("������֡֡�Ŀ����", afterThresold);
    return afterThresold;
}

Mat KNN(Mat currentframe, Mat afterThresold, Ptr<BackgroundSubtractor> pKNN)
{
    pKNN->apply(currentframe, afterThresold);
    threshold(afterThresold, afterThresold, 200, 255, THRESH_BINARY);//��ֵ�ָ���ж�ֵ��
    morphologyEx(afterThresold, afterThresold, MORPH_OPEN, getStructuringElement(1, Size(3, 5)));//��̬ѧ����--������
    dilate(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//����
    imshow("��������--KNN", afterThresold);
    return afterThresold;
}

Mat GMM(Mat currentframe, Mat afterThresold, Ptr<BackgroundSubtractorMOG2> bgsubtractor)
{
    bgsubtractor->apply(currentframe, afterThresold, 0.01);
    threshold(afterThresold, afterThresold, 200, 255, THRESH_BINARY);//��ֵ�ָ���ж�ֵ��

    morphologyEx(afterThresold, afterThresold, MORPH_OPEN, getStructuringElement(1, Size(3, 5)));//��̬ѧ����--������
    dilate(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//����
    imshow("��������--��ϸ�˹ģ��", afterThresold);
    return afterThresold;
}

Mat EdgeDetection(Mat old,Mat currentframe, Mat next_frame, Mat afterThresold )
{
    Mat frame_gray, old_gray, next_gray;
    //��֡ͼ����лҶȻ�
    cvtColor(currentframe, frame_gray, COLOR_BGR2GRAY);
    cvtColor(old, old_gray, COLOR_BGR2GRAY);
    cvtColor(next_frame, next_gray, COLOR_BGR2GRAY);
    //��֡�ĻҶ�ͼ�����Ԥ����
    GaussianBlur(frame_gray, frame_gray, Size(5, 5), 1);
    GaussianBlur(old_gray, old_gray, Size(5, 5), 1);
    GaussianBlur(next_gray, next_gray, Size(5, 5), 1);

     //֡�
    Mat diff1, diff2;
    absdiff(frame_gray, old_gray, diff1);
    absdiff(frame_gray, next_gray, diff2);
    Mat afterThresold1, afterThresold2;

    //��ֵͼ��Ķ�ֵ��
    threshold(diff1, afterThresold1, 0, 255, THRESH_OTSU);
    threshold(diff2, afterThresold2, 0, 255, THRESH_OTSU);

    Mat Canny_afterThresold;
    //��֡���
    add(afterThresold1, afterThresold2, afterThresold);
    Canny(afterThresold, Canny_afterThresold, 100, 200, 3, false);//Canny���Ӵ�����֡֡��õ���ͼ��
    bitwise_or(afterThresold, Canny_afterThresold, afterThresold);//��Ե�������ԭͼ������߼������

    dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));//����
    erode(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//��ʴ
    morphologyEx(afterThresold, afterThresold, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1)));     // ��̬ѧ����--������
    imshow("��֡��ֺͱ�Ե������㷨", afterThresold);
    return afterThresold;
}

void menu()
{
    cout << "-------------------------------------------" << endl;
    cout << "             ѡ��Ŀ����ķ�ʽ            " << endl;
    cout << "-------------------------------------------" << endl;
    cout << "            1.������֡֡�" << endl;
    cout << "            2.������֡֡�" << endl;
    cout << "            3.��������--KNN" << endl;
    cout << "            4.��������--GMM,��ϸ�˹ģ��" << endl;
    cout << "            5.������֡֡��ͱ�Ե���" << endl;
    cout << "����������ѡ�񣺣��磺1��" << endl;
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
    String filename = "F:\\ѧϰ\\������Ƶ����\\EnvironmentTest\\Videos\\exp2.avi";
    video.open(filename);
    if (!video.isOpened())
    {
        cout << "�޷�����Ƶ�ļ���" << endl;
        return -1;
    }
    long totalFrameNumber = video.get(CAP_PROP_FRAME_COUNT);//���ܵ���Ƶ֡��
    cout << totalFrameNumber << endl;
    menu();
    int m;
    cin >> m;

 
    Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
    bgsubtractor->setVarThreshold(20);
    Ptr<BackgroundSubtractor> pKNN = createBackgroundSubtractorKNN(false);

    int flag = 0;

    Mat currentframe;  //����һ֡ͼ
    Mat old; //ǰһ֡ͼ��
    Mat next_frame;//��ȡ��Ƶ��һ֡

    Mat frame_gray;  //����һ֡�Ҷ�ͼ
    Mat old_gray; //ǰһ֡�Ҷ�ͼ��
    Mat next_gray;//��Ƶ��һ֡�ĻҶ�ͼ��

    string num;
    while (true)
    {
        video >> currentframe; //����Ƶ�ж���һ֡ͼ��

        Mat frame2;
        currentframe.copyTo(frame2);

        if (currentframe.empty())  //��Ƶ��������˳�
        {
            break;
        }
        if (flag != 0)
        {
            Mat afterThresold;
            video >> next_frame;
            if (next_frame.empty())  //��Ƶ��������˳�
            {
                break;
            }
            afterThresold = choice(m, afterThresold, currentframe, old, next_frame, pKNN , bgsubtractor);

            //Ѱ�ұ߿�
            vector<vector<Point>> contours;
            //contours��һ��˫��������������ÿ��Ԫ�ر�����һ����������Point�㹹�ɵĵ�ļ���������ÿһ��Point�㼯����һ������
            //�ж�������������contours���ж���Ԫ��
            vector<Vec4i> hierarchy;
            //������ÿһ��Ԫ�ذ�����4��int�ͱ���
            //����hierarchy�ڵ�Ԫ�غ�contours�ڵ�Ԫ����һһ��Ӧ��
            findContours(afterThresold, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            //afterThresold:����������Ķ�ֵͼ��
            //RETR_TREE:�����ļ���ģʽ
            //CHAIN_APPROX_SIMPLE:�����Ľ��Ʒ���

            //���߿�
            drawContours(frame2, contours, -1, Scalar::all(0.7), 1, 8);
            //frame2:Ҫ����������ͼ��
            //contours:�����������������findContours()�����ҵ�������������ÿ�������������һ��point����
            //contourldx��ָ��Ҫ���������ı�ţ�����Ǹ�������������е�������һ��Ĭ��-1�Ϳ�
            //color��Ҫ������������ɫ
            //thickness��Ҫ���������Ĵ�ϸ
            //lineType��Ҫ���Ƶ��������ߵ�����


            RNG rngs = { 012345 };//���췽���趨һ������ֵ����ʾ�������ÿ�����ɵĽ������һ����
            //Scalar colors = Scalar(rngs.uniform(0, 255), rngs.uniform(0, 255), rngs.uniform(0, 255));//��[0,255)��Χ�����һ��ֵ

            Scalar colors = Scalar(0, 0, 255);
            int index = 0;
            int compCount = 0;
            for (; index >= 0; index = hierarchy[index][0], compCount++)//ÿ���Һ�һ������
            {//4��int�������ֱ��ʾ�������ĺ�һ��������ǰһ������������������Ƕ�������������
                Rect rec = boundingRect(contours[index]);//���������Ĵ�ֱ�߽���С���Σ���������ͼ�����±߽�ƽ�е�
                if (rec.area() <= 500)//����С��500���򲻻滭
                    continue;
                rectangle(frame2, rec, colors, 2);
                //frame2:Ҫ�������ͼƬ
                //rec:��������
                //colors:������ɫ
                //2���������
            }
        }
        //putText(frame2,num, Point(50, 60), FONT_HERSHEY_SIMPLEX, 200, Scalar(0, 0, 255), 4, 8);
        imshow("������Ƶ", frame2);  //��ʾ��ǰ�����һ֡ͼ��
        waitKey(30);    //��ʱ10ms
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



