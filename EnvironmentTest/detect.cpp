//ʵ��Ŀ����
#include "stdio.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

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
    cout << "-------------------------------------------" << endl;
    cout << "             ѡ��Ŀ����ķ�ʽ            " << endl;
    cout << "-------------------------------------------" << endl;
    cout << "            1.������֡֡�" << endl;
    cout << "            2.������֡֡�" << endl;
    cout << "            3.��������--KNN" << endl;
    cout << "            4.��������--GMM,��ϸ�˹ģ��" << endl;
    cout << "            5.������֡֡��ͱ�Ե���" << endl;
    cout << "����������ѡ�񣺣��磺1��" << endl;
    int m;
    cin >> m;

    //Mat bg = Mat::zeros(576, 768, CV_8U);
    //for (int m = 0; m < 576; m++)
    // {
    //     for (int n = 0; n <768; n++)
    //     {
    //        bg.at<uchar>(m, n)=80;
    //        a[m][n] += float(frame1.at<uchar>(m, n));
    //     }
    //} 
    //Mat bg = Mat::zeros(576, 768, CV_8U);
    //bg.convertTo(bg, CV_32FC1);
    //float a[576][768];
    //for (int i = 2; i < 100; i++)
    //{
    //    Mat frame1;
    //    video.set(CAP_PROP_POS_FRAMES, i);
    //    video >> frame1;
    //    cvtColor(frame1, frame1, COLOR_BGR2GRAY);
    //    frame1.convertTo(frame1, CV_32FC1);

    //    cout <<int( frame1.at<uchar>(10, 10) )<< endl;
    //    for (int m = 0;m < bg.rows; m++)
    //    {
    //        for (int n = 0; n < bg.cols; n++)
    //        {
    //           bg.at<float>(m, n)+= frame1.at<float>(m, n);
    //            //a[m][n] += float(frame1.at<uchar>(m, n));
    //        }
    //    } 
    //    //cout << a[10][10] << endl;

    //    cout << bg.at<float>(10, 10) << endl;
    //}
    ////for (int m = 0; m < bg.rows; m++)
    ////{
    ////    for (int n = 0; n < bg.cols; n++)
    ////    {
    ////        cout << bg.at<uint>(m, n) << endl;
    ////        //bg.at<Vec3b>(m, n)[0] = bg.at<Vec3b>(m, n)[0]/100;
    ////        bg.at<uint>(m, n) = bg.at<uint>(m, n) / 100;
    ////    }
    ////}
  /*  imshow("bg", bg);*/

    Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
    bgsubtractor->setVarThreshold(20);
    Ptr<BackgroundSubtractor> pKNN = createBackgroundSubtractorKNN(false);

    int flag = 0;

    Mat frame;  //����һ֡ͼ
    Mat old; //ǰһ֡ͼ��
    Mat next_frame;//��ȡ��Ƶ��һ֡

    Mat frame_gray;  //����һ֡�Ҷ�ͼ
    Mat old_gray; //ǰһ֡�Ҷ�ͼ��
    Mat next_gray;//��Ƶ��һ֡�ĻҶ�ͼ��

    string num;
    while (true)
    {
        video >> frame; //����Ƶ�ж���һ֡ͼ��
        
        Mat frame2;
        frame.copyTo(frame2);

        if (frame.empty())  //��Ƶ��������˳�
        {
            break;
        }
        if (flag != 0)
        {
            Mat afterThresold;

            if (m == 1)
            {
                //ǰ����֡ͼ����лҶȻ�
                cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
                cvtColor(old, old_gray, COLOR_BGR2GRAY);
                Size size; size.height = size.width = 7;

                //��ǰ����֡�ĻҶ�ͼ�����Ԥ����
                GaussianBlur(frame_gray, frame_gray, size, 1);
                GaussianBlur(old_gray, old_gray, size, 1);

                //֡�
                Mat diff;
                absdiff(frame_gray, old_gray, diff);

                //��ֵͼ��Ķ�ֵ��
                threshold(diff, afterThresold, 0,255, THRESH_OTSU);

                dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)));//����
                dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(8, 8)));
                erode(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//��ʴ
                imshow("������֡֡�Ŀ����", afterThresold);
            }

            //������֡֡�
            if (m == 2)
            {
                video >> next_frame;
                if (next_frame.empty())  //��Ƶ��������˳�
                {
                    break;
                }
                //��֡ͼ����лҶȻ�
                cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
                cvtColor(old, old_gray, COLOR_BGR2GRAY);
                cvtColor(next_frame, next_gray, COLOR_BGR2GRAY);
                
                //��֡�ĻҶ�ͼ�����Ԥ����
                GaussianBlur(frame_gray, frame_gray, Size(5,5), 1);
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

            }
            //��ϸ�˹ģ��
            if (m == 4)
            {
                bgsubtractor->apply(frame, afterThresold, 0.01);
                threshold(afterThresold, afterThresold, 200, 255, THRESH_BINARY);//��ֵ�ָ���ж�ֵ��

                morphologyEx(afterThresold, afterThresold, MORPH_OPEN, getStructuringElement(1, Size(3, 5)));//��̬ѧ����--������
                dilate(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//����
                imshow("��������--��ϸ�˹ģ��", afterThresold);
            }

            if (m == 3)
            {
                pKNN->apply(frame, afterThresold);

                threshold(afterThresold, afterThresold, 200, 255, THRESH_BINARY);//��ֵ�ָ���ж�ֵ��
                morphologyEx(afterThresold, afterThresold, MORPH_OPEN, getStructuringElement(1, Size(3, 5)));//��̬ѧ����--������
                dilate(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//����
                imshow("��������--KNN", afterThresold);
            }
            //��֡��ֺͱ�Ե������㷨
            if (m ==5)
            {
                video >> next_frame;
                if (next_frame.empty())  //��Ƶ��������˳�
                {
                    break;
                }
                //��֡ͼ����лҶȻ�
                cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
                cvtColor(old, old_gray, COLOR_BGR2GRAY);
                cvtColor(next_frame, next_gray, COLOR_BGR2GRAY);


                //��֡�ĻҶ�ͼ�����Ԥ����
                GaussianBlur(frame_gray, frame_gray, Size(5, 5), 1);
                GaussianBlur(old_gray, old_gray, Size(5, 5), 1);
                GaussianBlur(next_gray, next_gray, Size(5, 5), 1);

                //��֡�ĻҶ�ͼ����б�Ե���
                /*Canny(frame_gray, frame_gray, 100, 200, 3, false);
                Canny(old_gray, old_gray, 100, 200, 3, false);
                Canny(next_gray, next_gray, 100, 200, 3, false);*/
                /* Sobel(frame_gray, frame_gray, CV_64F, 1, 1, 5);
                 Sobel(old_gray, old_gray, CV_64F, 1, 1, 5);
                 Sobel(next_gray, next_gray, CV_64F, 1, 1, 5);*/

                 //֡�
                Mat diff1, diff2;
                absdiff(frame_gray, old_gray, diff1);
                absdiff(frame_gray, next_gray, diff2);
                Mat afterThresold1, afterThresold2;

                //��ֵͼ��Ķ�ֵ��
                threshold(diff1, afterThresold1,0,255, THRESH_OTSU);
                threshold(diff2, afterThresold2, 0, 255, THRESH_OTSU);

                Mat Canny_afterThresold;
                //��֡���
                add(afterThresold1, afterThresold2, afterThresold);
                Canny(afterThresold, Canny_afterThresold, 100, 200, 3, false);//Canny���Ӵ�����֡֡��õ���ͼ��
                //blur(afterThresold, afterThresold, Size(5, 5));
                //medianBlur(afterThresold, afterThresold, 3);
                // GaussianBlur(afterThresold, afterThresold, Size(5, 5), 0);
               bitwise_or(afterThresold, Canny_afterThresold, afterThresold);//��Ե�������ԭͼ������߼������
               
               dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));//����
               erode(afterThresold, afterThresold, getStructuringElement(1, Size(3, 3)));//��ʴ
               //erode(afterThresold, afterThresold, getStructuringElement(1, Size(5, 5)));//��ʴ
               morphologyEx(afterThresold, afterThresold, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1)));     // ��̬ѧ����--������
               //dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(3, 7)));//����
               //dilate(afterThresold, afterThresold, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));//����
               imshow("��֡��ֺͱ�Ե������㷨", afterThresold);
            }

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
            int index =  0;
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
        frame.copyTo(old);
        frame_gray.copyTo(old_gray);
        flag = 1;
        char key = waitKey(1);
        if (key == 27|| key == 113|| key == 81)
        {
            break;
        }
    }
    video.release();
}