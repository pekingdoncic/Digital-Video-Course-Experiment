//ʵ��� ������Ƶ���ź��˲�
#include "stdio.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>


using namespace std;
using namespace cv;


Mat frame;//��ǰ֡

void ChangeBrightness(int brightnessValue, Mat frame);//�ı�ͼ�������
void AdjustContrastRatio(int contrastRatioValue, Mat frame);//��߶Աȶ�

void HistEqual(Mat frame);//ֱ��ͼ���⻯

void MeanFilter(Mat frame, int filterN);//��ֵ�˲�
void GaussFilter(Mat frame, int filterN);//��˹�˲�
void MidFilter(Mat frame, int filterN);//��ֵ�˲�

void lixiangditong(Mat frame);//�����ͨ�˲�
void BLPF(Mat frame);//������˹��ͨ�˲�
void GausslowFliter(Mat image);//��˹��ͨ�˲�

int trackValue = 0; // ��¼�����λ�ã�����������ֵ
int brightnessValue = 255; // ���ȱ仯ֵ,��ʼ��Ϊ255ʱΪԭ��ͼ��
int contrastRatioValue = 10; // �Աȶȣ���ʼ��Ϊ10ʱΪԭ��ͼ��
int equalHistValue = 0; // ֱ��ͼ�Ƿ���⻯��Ϊ1��ʾ����ֱ��ͼ���⻯
int meanFilterValue = 1;//�Ƿ���о�ֵ�˲�
int fangkuangFilterValue = 1;//�����˲���ʼֵ
int gaussFilterValue = 0;//
int paused = 0;  // �Ƿ���ͣ
int framePos = 0;//��¼��Ƶ֡λ��
int MidFilterValue = 1;//��ֵ�˲�
int lixiangtditongFilterValue = 0;//�����ͨ�˲���ʼֵ
int BLPFValue = 0;//�Ƿ���а�����˹��ͨ�˲�
int GausslowValue = 0;//�Ƿ���и�˹��ͨ�˲�

 // ���ƽ�����λ�õĻص�����
void ControlVideo(int, void* param)
{
    VideoCapture cap = *(VideoCapture*)param;//��ȡ�ص���������Ķ���
    cap.set(CAP_PROP_POS_FRAMES, trackValue);//ͨ��������λ�ã���������Ƶ֡λ��
}


int main()
{
    VideoCapture video;
    String filename = "F:\\ѧϰ\\������Ƶ����\\EnvironmentTest\\Videos\\exp3.avi";
    video.open(filename);
    if (!video.isOpened())
    {
        cout << "�޷�����Ƶ�ļ���" << endl;
        return -1;
    }
    namedWindow("video", 0);//��������
    resizeWindow("video", 500, 500); //����һ��500*500��С�Ĵ���
    int frameallNums = video.get(CAP_PROP_FRAME_COUNT); //��Ƶ��֡��
    createTrackbar("������", "video", &trackValue, frameallNums, ControlVideo, &video); //��Ƶ����������
    createTrackbar("pause", "video", &paused, 1); //��Ƶ������ͣ����
    createTrackbar("����", "video", &brightnessValue, 510); //��Ƶ����
    createTrackbar("�Աȶ�", "video", &contrastRatioValue, 20); //��Ƶ�Աȶ�
    createTrackbar("ֱ��ͼ���⻯", "video", &equalHistValue, 1); //��Ƶֱ��ͼ���⻯
    createTrackbar("��ֵ�˲�", "video", &meanFilterValue, 5); //��Ƶ��ֵ�˲�
    createTrackbar("��˹�˲�", "video", &gaussFilterValue, 1); //��Ƶ��˹�˲�
    createTrackbar("��ֵ�˲�", "video", &MidFilterValue, 5); //��Ƶ��ֵ�˲�
    createTrackbar("�����˲�", "video", &fangkuangFilterValue, 3); //��Ƶ�����˲�
    createTrackbar("�����ͨ�˲�", "video", &lixiangtditongFilterValue, 1); //�����ͨ�˲�
    createTrackbar("��˹��ͨ�˲�", "video", &GausslowValue, 1); //��˹��ͨ�˲�
    createTrackbar("������˹��ͨ�˲�", "video", &BLPFValue, 1); //������˹��ͨ�˲�
    
    while (1)
    {
        framePos = video.get(CAP_PROP_POS_FRAMES);//��ȡ��Ƶ֡λ��
        setTrackbarPos("������", "video", framePos);//ͨ����Ƶ֡λ�ã������»�����λ��
        if (paused == 0)//����
        {
            //ʵ��ѭ������
            if (framePos == int(video.get(CAP_PROP_FRAME_COUNT)))//���������֡�����һ֡
            {
                framePos = 0;//֡����Ϊ0
                video.set(CAP_PROP_POS_FRAMES, 0);//����Ƶ��0��ʼ��
            }
            video >> frame;//ȡ֡��frame
            if (frame.empty())//ȡ֡ʧ��
            {
                break;
            }   
        }
        if (contrastRatioValue != 10)//����ƶ��˶Աȶ�ֵ
            AdjustContrastRatio(contrastRatioValue, frame);
        if (brightnessValue != 255)//����ƶ�������ֵ      
            ChangeBrightness(brightnessValue, frame);
        if (meanFilterValue != 1)
            MeanFilter(frame, meanFilterValue);//��ֵ�˲�
        if (gaussFilterValue != 0)
            GaussFilter(frame, 3);//��˹�˲�
        if (fangkuangFilterValue != 1)//�����˲�
        {
            if (fangkuangFilterValue == 0)
                return 0;
            boxFilter(frame, frame, -1, Size(fangkuangFilterValue, fangkuangFilterValue), Point(-1, -1), 0, 4);
        
        }
            
        if (MidFilterValue != 1)//���ѡ������ֵ�˲�
        {
            if (MidFilterValue % 2 == 0)//�����ż�������1��Ϊ����
                MidFilterValue += 1;//2,3->3  4,5->5
            medianBlur(frame, frame, MidFilterValue);
        }
        if (BLPFValue != 0)//���ѡ���˰�����˹��ͨ�˲�
            BLPF(frame);
        if (GausslowValue != 0)//���ѡ���˸�˹��ͨ�˲�
            GausslowFliter(frame);
        if (lixiangtditongFilterValue == 1)//���ѡ���������ͨ�˲�
            lixiangditong(frame);
        if (equalHistValue !=0)//���ѡ����ֱ��ͼ���⻯
            HistEqual(frame);//���о��⻯
        else
            imshow("��Ƶ", frame);
        waitKey(10);
        char c = waitKey(3);
        if (c == 27)
            break;//��esc���˳�
    }
    video.release();
}

//�ı�֡ͼ�������
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

//�ı�ͼƬ�ĶԱȶ�
void AdjustContrastRatio(int contrastRatioValue, Mat frame)//�ı�ͼ��ĶԱȶ�
{
    for (int i = 0; i < frame.rows; i++)//��ɫͼ��Ҫ��RGB����ͨ����ֵ���иı�
    {
        for (int j = 0; j < frame.cols; j++)
        {
            float b = frame.at<Vec3b>(i, j)[0];//��ȡ����ͨ����ֵ
            float g = frame.at<Vec3b>(i, j)[1];
            float r = frame.at<Vec3b>(i, j)[2];
            if (contrastRatioValue >= 10)//�Աȶȵ�ֵ����10�����޸�����ֵ��b,g,rͬ��һ��ϵ��,ϵ��ֵΪ��x-9)
            {
                frame.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(b * (contrastRatioValue - 9));
                frame.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(g * (contrastRatioValue - 9));
                frame.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(r * (contrastRatioValue - 9));
            }
            else//�Աȶȵ�ֵ����10�����޸�����ֵ��b,g,rͬ��һ��ϵ����ϵ��ֵΪ 1/��-x+11)
            {
                frame.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(b * float(1.0) / (-contrastRatioValue + 11));
                frame.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(g * float(1.0) / (-contrastRatioValue + 11));
                frame.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(r * float(1.0) / (-contrastRatioValue + 11));
            }
        }
    }
}

//ֱ��ͼ���⻯
void HistEqual(Mat frame)
{
    // ����ɫͼ���ֳ�ͨ��
    vector<Mat> channels;
    split(frame, channels);

    for (int c = 0; c < 3; c++) // ����ÿ����ɫͨ����0����ɫ��1����ɫ��2����ɫ��
    {
        // ����ǰͨ��ת��Ϊ�Ҷ�ͼ��
        Mat channel = channels[c];
        
        // ����ֱ��ͼ�ֲ�
        int hist[256] = { 0 };
        for (int i = 0; i < channel.rows; i++)
        {
            for (int j = 0; j < channel.cols; j++)
            {
                hist[int(channel.at<uchar>(i, j))]++;
            }
        }

        // �����ۻ�ֱ��ͼ
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
            // �����µ�ǿ��ֵ
            bMap[i] = int(double(hist_all[i]) / (channel.cols * channel.rows) * 255 + 0.5);
        }

        // ��ֱ��ͼ���⻯Ӧ���ڵ�ǰͨ��
        for (int i = 0; i < channel.rows; i++)
        {
            for (int j = 0; j < channel.cols; j++)
            {
                channel.at<uchar>(i, j) = bMap[int(channel.at<uchar>(i, j))];
            }
        }

        // �����޸ĺ��ͨ��
        channels[c] = channel;
    }

    // ���޸ĺ��ͨ���ϲ��ز�ɫͼ��
    merge(channels, frame);

    // ��ʾ�����ɫͼ��
    imshow("��Ƶ", frame);
}


//��ֵ�˲�,filterNΪ�˲�ģ��Ŀ��
void MeanFilter(Mat frame, int filterN)
{
    if(filterN == 0)
    {
        return;
    }
            if (filterN % 2 == 0) {//�����ż�����˲��˵ı߳���1
                filterN = filterN - 1;
            }
            else {
                filterN = filterN;
            }
            //����˵�һ��
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
                            //����ͨ���ֱ����ܺ�
                            sum[0] += frame.at<Vec3b>(k, l)[0];
                            sum[1] += frame.at<Vec3b>(k, l)[1];
                            sum[2] += frame.at<Vec3b>(k, l)[2];
                        }
                    }
                    //�ֱ����ֵ
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

//��˹�˲�
void GaussFilter(Mat frame, int filterN)//��˹�˲�
{
    int a[3][3] = { 1,2,1,2,4,2,1,2,1 };//ģ������
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
                    tempx = tempx < 0 ? 0 : tempx;//����tempx�ķ�Χ
                    tempx = tempx > frame.rows - 1 ? frame.rows - 1 : tempx;
                    tempy = tempy < 0 ? 0 : tempy;//����tempy�ķ�Χ
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

//��ֵ�˲�
void MidFilter(Mat frame, int filterN)
{
    //Mat im, m1, m2, m3, m4, m5, m6, B, G, R;
    //vector <Mat> channels;
    //split(frame, channels);
    //B = channels.at(0);
    //G = channels.at(1);
    //R = channels.at(2);//ͨ������
    //int a = frame.rows;
    //int b = frame.cols; //����к���
    //int k = filterN;
    //for (int i = k / 2; i < a - k / 2; i++)//����ͼ��ע��Ҫ�ѱ߿򲿷�ȥ����
    //{
    //    for (int j = k / 2; j < b - k / 2; j++)
    //    {
    //        m1 = B(Rect(j - k / 2, i - k / 2, k, k)).clone();//ѡȡk*k�����е�Bͨ��
    //        m2 = G(Rect(j - k / 2, i - k / 2, k, k)).clone();//ѡȡk*k����Gͨ��
    //        m3 = R(Rect(j - k / 2, i - k / 2, k, k)).clone();//ѡȡk*k����Rͨ��
    //        m1 = m1.reshape(1, 1);
    //        m2 = m2.reshape(1, 1);
    //        m3 = m3.reshape(1, 1); //����ֵ�����Ϊһ��
    //        cv::sort(m1, m4, 0);
    //        cv::sort(m2, m5, 0);
    //        cv::sort(m3, m6, 0);//��������
    //        int p0 = m4.at<uchar>(0, k * k / 2 + 1);
    //        int p1 = m5.at<uchar>(0, k * k / 2 + 1);
    //        int p2 = m6.at<uchar>(0, k * k / 2 + 1);//ѡȡ��ͨ����������ֵ��
    //        frame.at<Vec3b>(i - k / 2, j - k / 2)[0] = p0;
    //        frame.at<Vec3b>(i - k / 2, j - k / 2)[1] = p1;
    //        frame.at<Vec3b>(i - k / 2, j - k / 2)[2] = p2;//��������ظ�ֵ
    //    }
    //}
    //   if (filterN % 2 == 0) {
    //    filterN = filterN - 1;
    //}
    //else {
    //    filterN = filterN;
    //}
    ////����˵�һ��
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
    //        // ѭ������
    //        int	m;
    //        int	n;
    //        // �м����
    //       int rTemp=0;
    //       int gTemp=0;
    //       int bTemp=0;
    //        // ��ð�ݷ��������������
    //        for (n = 0; n < iFilterLen - 1; n++)
    //        {
    //            for (m = 0; m < iFilterLen - j - 1; m++)
    //            {
    //                if (tempr[m] > tempr[m + 1])
    //                {
    //                    // ����
    //                    rTemp = tempr[m];
    //                    tempr[m] = tempr[m + 1];
    //                    tempr[m + 1] = rTemp;
    //                }
    //                if (tempg[m] > tempg[m + 1])
    //                {
    //                    // ����
    //                    gTemp = tempg[m];
    //                    tempg[m] = tempg[m + 1];
    //                    tempg[m + 1] = gTemp;
    //                }
    //                if (tempb[m] > tempb[m + 1])
    //                {
    //                    // ����
    //                    bTemp = tempb[m];
    //                    tempb[m] = tempb[m + 1];
    //                    tempb[m + 1] = bTemp;
    //                }
    //            }
    //        }
    //        // ������ֵ
    //        if ((iFilterLen & 1) > 0)
    //        {
    //            // ������������Ԫ�أ������м�һ��Ԫ��
    //            rTemp = tempr[(iFilterLen + 1) / 2];
    //            gTemp = tempg[(iFilterLen + 1) / 2];
    //            bTemp = tempb[(iFilterLen + 1) / 2];
    //        }
    //        else
    //        {
    //            // ������ż����Ԫ�أ������м�����Ԫ��ƽ��ֵ
    //            rTemp = (tempr[iFilterLen / 2] + tempr[iFilterLen / 2 + 1]) / 2;
    //            gTemp = (tempg[iFilterLen / 2] + tempg[iFilterLen / 2 + 1]) / 2;
    //            bTemp = (tempb[iFilterLen / 2] + tempb[iFilterLen / 2 + 1]) / 2;
    //        }

    //        // ��ȡ��ֵ
    //        frame.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(rTemp);
    //        frame.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(gTemp);
    //        frame.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(bTemp);
    //    }
    //}
            
     
}

//����Ҷ�任�õ�Ƶ��ͼ�͸�������
void My_DFT(Mat input_image, Mat& output_image, Mat& transform_image)
{
    //1.��չͼ�����Ϊ2��3��5�ı���ʱ�����ٶȿ�
    int m = getOptimalDFTSize(input_image.rows);
    int n = getOptimalDFTSize(input_image.cols);
    copyMakeBorder(input_image, input_image, 0, m - input_image.rows, 0, n - input_image.cols, BORDER_CONSTANT, Scalar::all(0));

    //2.����һ��˫ͨ������planes���������渴����ʵ�����鲿
    Mat planes[] = { Mat_<float>(input_image), Mat::zeros(input_image.size(), CV_32F) };

    //3.�Ӷ����ͨ�������д���һ����ͨ������:transform_image������Merge����������ϲ�Ϊһ����ͨ�����У�����������ÿ��Ԫ�ؽ�����������Ԫ�صļ���
    merge(planes, 2, transform_image);

    //4.���и���Ҷ�任
    dft(transform_image, transform_image);

    //5.���㸴���ķ�ֵ��������output_image��Ƶ��ͼ��
    split(transform_image, planes); // ��˫ͨ����Ϊ������ͨ����һ����ʾʵ����һ����ʾ�鲿
    Mat transform_image_real = planes[0];
    Mat transform_image_imag = planes[1];

    magnitude(planes[0], planes[1], output_image); //���㸴���ķ�ֵ��������output_image��Ƶ��ͼ��

    //6.ǰ��õ���Ƶ��ͼ�������󣬲�����ʾ�����ת��
    output_image += Scalar(1);   // ȡ����ǰ�����е����ض���1����ֹlog0
    log(output_image, output_image);   // ȡ����
    normalize(output_image, output_image, 0, 1, NORM_MINMAX); //��һ��

    //7.���к��طֲ�����ͼ����
    output_image = output_image(Rect(0, 0, output_image.cols & -2, output_image.rows & -2));
    imshow("output_image_1", output_image);
    // �������и���Ҷͼ���е����ޣ�ʹԭ��λ��ͼ������
    int cx = output_image.cols / 2;
    int cy = output_image.rows / 2;
    Mat q0(output_image, Rect(0, 0, cx, cy));   // ��������
    Mat q1(output_image, Rect(cx, 0, cx, cy));  // ��������
    Mat q2(output_image, Rect(0, cy, cx, cy));  // ��������
    Mat q3(output_image, Rect(cx, cy, cx, cy)); // ��������

      //�����������Ļ�
    Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);//���������½��н���
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);//���������½��н���


    Mat q00(transform_image_real, Rect(0, 0, cx, cy));   // ��������
    Mat q01(transform_image_real, Rect(cx, 0, cx, cy));  // ��������
    Mat q02(transform_image_real, Rect(0, cy, cx, cy));  // ��������
    Mat q03(transform_image_real, Rect(cx, cy, cx, cy)); // ��������
    q00.copyTo(tmp); q03.copyTo(q00); tmp.copyTo(q03);//���������½��н���
    q01.copyTo(tmp); q02.copyTo(q01); tmp.copyTo(q02);//���������½��н���

    Mat q10(transform_image_imag, Rect(0, 0, cx, cy));   // ��������
    Mat q11(transform_image_imag, Rect(cx, 0, cx, cy));  // ��������
    Mat q12(transform_image_imag, Rect(0, cy, cx, cy));  // ��������
    Mat q13(transform_image_imag, Rect(cx, cy, cx, cy)); // ��������
    q10.copyTo(tmp); q13.copyTo(q10); tmp.copyTo(q13);//���������½��н���
    q11.copyTo(tmp); q12.copyTo(q11); tmp.copyTo(q12);//���������½��н���
    imshow("output_image_2", output_image);
    planes[0] = transform_image_real;
    planes[1] = transform_image_imag;
    merge(planes, 2, transform_image);//������Ҷ�任������Ļ�
}

//�����ͨ�˲�

void lixiangditong(Mat image)
{
    Mat image_gray, image_output, image_transform;   //��������ͼ�񣬻Ҷ�ͼ�����ͼ��
    cvtColor(image, image_gray, COLOR_BGR2GRAY); //ת��Ϊ�Ҷ�ͼ

     //1������Ҷ�任��image_outputΪ����ʾ��Ƶ��ͼ��image_transformΪ����Ҷ�任�ĸ������
    My_DFT(image_gray, image_output, image_transform);
     //imshow("image_output", image_output);
    
     //2�������ͨ�˲�
     Mat planes[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
     split(image_transform, planes);//����ͨ������ȡʵ���鲿
     Mat image_transform_real = planes[0];
     Mat image_transform_imag = planes[1];

     int core_x = image_transform_real.rows / 2;//Ƶ��ͼ��������
     int core_y = image_transform_real.cols / 2;
     int r = 90;  //�˲��뾶
     for (int i = 0; i < image_transform_real.rows; i++)
     {
         for (int j = 0; j < image_transform_real.cols; j++)
            {
                //�������ĵľ���������ð뾶r�ĵ�����ֵ��Ϊ0
                if (((i - core_x) * (i - core_x) + (j - core_y) * (j - core_y)) > r * r)
                {
                    image_transform_real.at<float>(i, j) = 0;
                    image_transform_imag.at<float>(i, j) = 0;
                }
            }
     }
     planes[0] = image_transform_real;
     planes[1] = image_transform_imag;
     Mat image_transform_ilpf;//���������ͨ�˲�����
     merge(planes, 2, image_transform_ilpf);

     //3������Ҷ��任
     Mat iDft[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
     idft(image_transform_ilpf, image_transform_ilpf);//����Ҷ��任
     split(image_transform_ilpf, iDft);//����ͨ������Ҫ��ȡ0ͨ��
     magnitude(iDft[0], iDft[1], iDft[0]); //���㸴���ķ�ֵ��������iDft[0]
     normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);//��һ������
     frame = iDft[0];   
}

//������˹��ͨ�˲�
void BLPF(Mat image)
{
    Mat  image_gray, image_output, image_transform;   //��������ͼ�񣬻Ҷ�ͼ�����ͼ��
    cvtColor(image, image_gray, COLOR_BGR2GRAY); //ת��Ϊ�Ҷ�ͼ
    //1������Ҷ�任��image_outputΪ����ʾ��Ƶ��ͼ��image_transformΪ����Ҷ�任�ĸ������
    My_DFT(image_gray, image_output, image_transform);

    //2��������˹��ͨ�˲�
    Mat planes[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
    split(image_transform, planes);//����ͨ������ȡʵ���鲿
    Mat image_transform_real = planes[0];
    Mat image_transform_imag = planes[1];

    int core_x = image_transform_real.rows / 2;//Ƶ��ͼ��������
    int core_y = image_transform_real.cols / 2;
    int r = 90;  //�˲��뾶
    float h;
    float n = 2; //������˹ϵ��
    float D;  //�������ľ���
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
    Mat image_transform_ilpf;//���������˹��ͨ�˲����
    merge(planes, 2, image_transform_ilpf);

    //3������Ҷ��任
    Mat iDft[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
    idft(image_transform_ilpf, image_transform_ilpf);//����Ҷ��任
    split(image_transform_ilpf, iDft);//����ͨ������Ҫ��ȡ0ͨ��
    magnitude(iDft[0], iDft[1], iDft[0]); //���㸴���ķ�ֵ��������iDft[0]
    normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);//��һ������
    frame = iDft[0];
}

//��˹��ͨ�˲�
void GausslowFliter(Mat image)
{
    Mat  image_gray, image_output, image_transform;   //��������ͼ�񣬻Ҷ�ͼ�����ͼ��
    cvtColor(image, image_gray, COLOR_BGR2GRAY); //ת��Ϊ�Ҷ�ͼ
    //1������Ҷ�任��image_outputΪ����ʾ��Ƶ��ͼ��image_transformΪ����Ҷ�任�ĸ������
    My_DFT(image_gray, image_output, image_transform);
    //2����˹��ͨ�˲�
    Mat planes[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
    split(image_transform, planes);//����ͨ������ȡʵ���鲿
    Mat image_transform_real = planes[0];
    Mat image_transform_imag = planes[1];

    int core_x = image_transform_real.rows / 2;//Ƶ��ͼ��������
    int core_y = image_transform_real.cols / 2;
    int r = 80;  //�˲��뾶
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
    Mat image_transform_ilpf;//�����˹��ͨ�˲����
    merge(planes, 2, image_transform_ilpf);

    //3������Ҷ��任
    Mat iDft[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
    idft(image_transform_ilpf, image_transform_ilpf);//����Ҷ��任
    split(image_transform_ilpf, iDft);//����ͨ������Ҫ��ȡ0ͨ��
    magnitude(iDft[0], iDft[1], iDft[0]); //���㸴���ķ�ֵ��������iDft[0]
    normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);//��һ������

    frame = iDft[0];
}


