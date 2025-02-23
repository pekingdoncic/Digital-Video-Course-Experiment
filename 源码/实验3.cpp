#include "stdio.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//ȫ�ֱ�����
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
int hue_shift = 0;//ɫ�ȵ���
Mat frame;//��ǰ֡

//����������
void ChangeBrightness(int brightnessValue, Mat frame);//�ı�ͼ�������
void AdjustContrastRatio(int contrastRatioValue, Mat frame);//��߶Աȶ�

void HistEqual(Mat frame);//ֱ��ͼ���⻯


void adjustHue(Mat& frame, int hue_shift);//����ɫ��
void BoxFilterAlgorithm(Mat& frame, int fangkuangFilterValue);
void MeanFilter(Mat frame, int filterN);//��ֵ�˲�
void GaussFilter(Mat frame, int filterN);//��˹�˲�
void MidFilter(Mat frame, int filterN);//��ֵ�˲�

void lixiangditong(Mat frame);//�����ͨ�˲�
void BLPF(Mat frame);//������˹��ͨ�˲�
void GausslowFliter(Mat image);//��˹��ͨ�˲�

// ���ƽ�����λ�õĻص�����
void ControlVideo(int, void* param)
{
    VideoCapture cap = *(VideoCapture*)param;//��ȡ�ص���������Ķ���
    cap.set(CAP_PROP_POS_FRAMES, trackValue);//ͨ��������λ�ã���������Ƶ֡λ��
}

//������������
void CreateUI(int frameallNums, VideoCapture &video)
{
    createTrackbar("������", "video", &trackValue, frameallNums, ControlVideo, &video); //��Ƶ����������
    createTrackbar("pause", "video", &paused, 1); //��Ƶ������ͣ����
    createTrackbar("����", "video", &brightnessValue, 510); //��Ƶ����
    createTrackbar("�Աȶ�", "video", &contrastRatioValue, 20); //��Ƶ�Աȶ�
    createTrackbar("ֱ��ͼ���⻯", "video", &equalHistValue, 1); //��Ƶֱ��ͼ���⻯
    createTrackbar("����ɫ��", "video", &hue_shift, 360);
    createTrackbar("��ֵ�˲�", "video", &meanFilterValue, 5); //��Ƶ��ֵ�˲�
    createTrackbar("��˹�˲�", "video", &gaussFilterValue, 1); //��Ƶ��˹�˲�
    createTrackbar("��ֵ�˲�", "video", &MidFilterValue, 5); //��Ƶ��ֵ�˲�
    createTrackbar("�����˲�", "video", &fangkuangFilterValue, 3); //��Ƶ�����˲�
    createTrackbar("�����ͨ�˲�", "video", &lixiangtditongFilterValue, 1); //�����ͨ�˲�
    createTrackbar("��˹��ͨ�˲�", "video", &GausslowValue, 1); //��˹��ͨ�˲�
    createTrackbar("������˹��ͨ�˲�", "video", &BLPFValue, 1); //������˹��ͨ�˲�
}

int main()
{
	VideoCapture video;
	video.open("F:\\ѧϰ\\������Ƶ����\\EnvironmentTest\\Videos\\exp3.avi");
	if (!video.isOpened())
	{
		cout << "�޷�����Ƶ�ļ���" << endl;
		return -1;
	}

	//����ui������;
    namedWindow("video", 0);//��������
    resizeWindow("video", 500, 500); //����һ��500*500��С�Ĵ���
    int frameallNums = video.get(CAP_PROP_FRAME_COUNT); //��Ƶ��֡��
    CreateUI(frameallNums, video);
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
        if (hue_shift != 0)//���������ɫ��
            adjustHue(frame, hue_shift);
        if (contrastRatioValue != 10)//����ƶ��˶Աȶ�ֵ
            AdjustContrastRatio(contrastRatioValue, frame);
        if (brightnessValue != 255)//����ƶ�������ֵ      
            ChangeBrightness(brightnessValue, frame);
        if (meanFilterValue != 1)
            MeanFilter(frame, meanFilterValue);//��ֵ�˲�
        if (gaussFilterValue != 0)
            GaussFilter(frame, 3);//��˹�˲�
        if (fangkuangFilterValue != 1)//�����˲�
           // BoxFilterAlgorithm(frame, fangkuangFilterValue);
            boxFilter(frame, frame, -1, Size(fangkuangFilterValue, fangkuangFilterValue), Point(-1, -1), 0, 4);
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
        if (equalHistValue != 0)//���ѡ����ֱ��ͼ���⻯
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
            //�������ó�Ϊvalue-255��Ϊ�˼ȿ��԰�ͼ����������ڣ�Ҳ���԰�ͼ�����������
            frame.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(b + brightnessValue - 255);
            frame.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(g + brightnessValue - 255);
            frame.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(r + brightnessValue - 255);
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
void HistEqual(Mat frame)//ֱ��ͼ���⻯
{
    cvtColor(frame, frame, COLOR_BGR2GRAY);//��֡ͼ��ת��Ϊ�Ҷ�ͼ
    //��ֱ��ͼ�ֲ�
    int hist[256] = { 0 };
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            hist[int(frame.at<uchar>(i, j))]++;
        }
    }
    long hist_all[256] = { 0 };//��ʱ�洢����
    int bMap[256] = { 0 };//�Ҷ�ӳ���
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
        //���ݹ�������»Ҷ�ֵ
        bMap[i] = int(double(hist_all[i]) / (frame.cols * frame.rows) * 255 + 0.5);
    }
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            frame.at<uchar>(i, j) = bMap[int(frame.at<uchar>(i, j))];//���Ҷ�ӳ���õ��ĻҶ�ֵ���Ƹ�ԭͼ��
        }
    }
    imshow("��Ƶ", frame);
}

//����ɫ�ȣ�
void adjustHue(Mat& frame, int hue_shift) {
    // a) ��RGBͼ��ת��ΪHSI�ռ�
    Mat hsi_image;
    cvtColor(frame, hsi_image, COLOR_BGR2HSV);

    // b) ����ͨ��
    vector<Mat> hsi_channels;
    split(hsi_image, hsi_channels);

    // c) ����ɫ�ȣ�Hue��ͨ��
    hsi_channels[0] += hue_shift;

    // d) ��ͨ�����ºϳ�ΪHSIͼ��
   merge(hsi_channels, hsi_image);

    // e) ��HSIͼ��ת����RGB�ռ�
    cvtColor(hsi_image, frame, COLOR_HSV2BGR);

    // f) ��ʾ��� (���Ը�����Ҫѡ���Ƿ���ʾ)
    //imshow("�������ɫ��֡", frame);
    //waitKey(1);  // ���һ��С���ӳ���������ʾ
}

//��ֵ�˲�,filterNΪ�˲�ģ��Ŀ��
void MeanFilter(Mat frame, int filterN)
{
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
   Mat dst = frame.clone();

   int half_kernel = filterN / 2;

   for (int y = half_kernel; y < frame.rows - half_kernel; ++y) {
       for (int x = half_kernel; x < frame.cols - half_kernel; ++x) {
           vector<uchar> values_b;
           vector<uchar> values_g;
           vector<uchar> values_r;

            // ��ȡ���е�����ֵ
            for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
                for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                    values_b.push_back(frame.at<Vec3b>(y + ky, x + kx)[0]);
                    values_g.push_back(frame.at<Vec3b>(y + ky, x + kx)[1]);
                    values_r.push_back(frame.at<Vec3b>(y + ky, x + kx)[2]);
                }
            }

            // ������ֵ��������
            sort(values_b.begin(), values_b.end());
            sort(values_g.begin(), values_g.end());
            sort(values_r.begin(), values_r.end());

            // ȡ��ֵ��Ϊ�˲��������ֵ
            dst.at<Vec3b>(y, x) = Vec3b(values_b[filterN * filterN / 2],
                values_g[filterN * filterN / 2],
                values_r[filterN * filterN / 2]);
        }
    }
    // �����д��ԭʼ֡
    frame = dst;

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
   // imshow("output_image_1", output_image);
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
    //imshow("output_image_2", output_image);
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

//�����˲���
void BoxFilterAlgorithm(Mat& frame, int fangkuangFilterValue)
{
    cv::Mat dst = frame.clone();

    int half_kernel = fangkuangFilterValue / 2;

    for (int y = half_kernel; y < frame.rows - half_kernel; ++y) {
        for (int x = half_kernel; x < frame.cols - half_kernel; ++x) {
            // ���㷽���˲��������ص�ƽ��ֵ
            int sum_b = 0, sum_g = 0, sum_r = 0;

            for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
                for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                    sum_b += frame.at<cv::Vec3b>(y + ky, x + kx)[0];
                    sum_g += frame.at<cv::Vec3b>(y + ky, x + kx)[1];
                    sum_r += frame.at<cv::Vec3b>(y + ky, x + kx)[2];
                }
            }

            // ����ƽ��ֵ������Ŀ��ͼ��
            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(
                sum_b / (fangkuangFilterValue * fangkuangFilterValue),
                sum_g / (fangkuangFilterValue * fangkuangFilterValue),
                sum_r / (fangkuangFilterValue * fangkuangFilterValue)
            );
        }
    }

    // �����д��ԭʼ֡
    frame = dst;
}