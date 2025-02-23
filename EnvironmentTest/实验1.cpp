
#include <opencv2/opencv.hpp>
#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <queue>
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

void maxEntropyThresholdSegmentation(Mat currentframe)
{
	cvtColor(currentframe, currentframe, COLOR_BGR2GRAY);
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

//double calculateEntropy(const Mat& img, int threshold) {
//	Mat mask = (img <= threshold);
//	int totalPixels = img.rows * img.cols;
//
//	Mat maskZero = (mask == 0);
//	Mat maskOne = (mask == 255);
//
//	double p1 = static_cast<double>(countNonZero(~maskZero)) / totalPixels;
//	double p2 = static_cast<double>(countNonZero(maskOne)) / totalPixels;
//
//	if (p1 == 0 || p2 == 0) {
//		return 0;
//	}
//
//	double entropy = -p1 * log2(p1) - p2 * log2(p2);
//	return entropy;
//}
//
//
//int findMaxEntropyThreshold(const Mat& img) {
//	int maxThreshold = 0;
//	double maxEntropy = 0;
//
//	for (int threshold = 1; threshold < 256; ++threshold) {
//		double entropy = calculateEntropy(img, threshold);
//		if (entropy > maxEntropy) {
//			maxEntropy = entropy;
//			maxThreshold = threshold;
//		}
//	}
//
//	return maxThreshold;
//}
//
//void applyThreshold(const Mat& input, Mat& output, int threshold) {
//	output = (input > threshold) * 255;
//}
//
//void maxEntropyThreshold(InputArray inputImage) {
//	// ��InputArrayת��ΪMat
//	Mat image = inputImage.getMat();
//
//	// �����������ֵ
//	int threshold = findMaxEntropyThreshold(image);
//
//	cout << "Optimal Threshold: " << threshold << endl;
//
//	// Ӧ����ֵ
//	Mat segmentedImage;
//	applyThreshold(image, segmentedImage, threshold);
//
//	imshow("�������ֵ�ָ��㷨", segmentedImage);
//	waitKey(1);
//	/*return segmentedImage;*/
//}


void OTSU(Mat frame)
{
	// ������֡ת��Ϊ�Ҷ�ͼ
	cvtColor(frame, frame, COLOR_BGR2GRAY);

	// ��ʼ������
	long i, j, t;
	int iThreshold, iNewThreshold, iMaxGrayValue, iMinGrayValue, iMean1GrayValue, iMean2GrayValue;
	iMaxGrayValue = 0;
	iMinGrayValue = 255;

	// ǰ���ͱ������ر���
	double w0, w1;

	// ����
	double G = 0, tempG = 0;

	// ���ڼ�������Ҷ�ƽ��ֵ�ı���
	long lP1, lP2, lS1, lS2;

	// ֱ��ͼ���洢����ǿ��Ƶ��
	long lHistogram[256] = { 0 };

	// ����ֱ��ͼ
	for (int i = 0; i < frame.rows; i++)
	{
		uchar* data = frame.ptr<uchar>(i);
		for (int j = 0; j < frame.cols; j++)
		{
			lHistogram[int(data[j])]++;
			if (iMinGrayValue > data[j])
				iMinGrayValue = data[j];
			if (iMaxGrayValue < data[j])
				iMaxGrayValue = data[j];
		}
	}

	// ��������
	int n = frame.rows * frame.cols;

	// �������ܵ���ֵֵ
	for (t = iMinGrayValue; t < iMaxGrayValue; t++)
	{
		iNewThreshold = t;
		lP1 = 0; lP2 = 0; lS1 = 0; lS2 = 0;

		// ������������ĻҶ�ƽ��ֵ
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

		// �����Ȩƽ��ֵ
		double mean = w0 * iMean1GrayValue + w1 * iMean2GrayValue;

		// ���㷽��
		G = (double)w0 * (iMean1GrayValue - mean) * (iMean1GrayValue - mean) + w1 * (iMean2GrayValue - mean) * (iMean2GrayValue - mean);

		// �����������������ֵ
		if (G > tempG)
		{
			tempG = G;
			iThreshold = iNewThreshold;
		}
	}

	// ʹ�ü���õ�����ֵ��ͼ����ж�ֵ������
	Mat after2;
	threshold(frame, after2, iThreshold, 255, THRESH_BINARY);

	// ��ʾ���
	imshow("OTSU", after2);
	waitKey(10);
}



double calculateEntropy(const Mat& img, int threshold) {
	int totalPixels = img.rows * img.cols;
	int countBelow = 0, countAbove = 0;
	double entropyBelow = 0, entropyAbove = 0;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			int pixelValue = static_cast<int>(img.at<uchar>(i, j));
			if (pixelValue < threshold) {
				countBelow++;
			}
			else {
				countAbove++;
			}
		}
	}

	for (int i = 0; i < 256; ++i) {
		if (i < threshold) {
			double probability = static_cast<double>(countBelow) / totalPixels;
			if (probability > 0) {
				entropyBelow -= probability * log2(probability);
			}
		}
		else {
			double probability = static_cast<double>(countAbove) / totalPixels;
			if (probability > 0) {
				entropyAbove -= probability * log2(probability);
			}
		}
	}

	return entropyBelow + entropyAbove;
}

void maxEntropyThreshold(const InputArray& inputImage) {

	Mat image = inputImage.getMat();

	// ��ʼ������
	int bestThreshold = 0;
	double maxEntropy = 0;

	// ����ÿ�����ܵ���ֵ
	for (int threshold = 0; threshold < 256; ++threshold) {
		double entropy = calculateEntropy(image, threshold);

		// ���������ֵ�Ͷ�Ӧ����ֵ
		if (entropy > maxEntropy) {
			maxEntropy = entropy;
			bestThreshold = threshold;
		}
	}

	// ���������ֵ��ͼ����ж�ֵ��
	Mat segmented;
	
	cvtColor(image, image, COLOR_BGR2GRAY);
	threshold(image, segmented, bestThreshold, 255, THRESH_BINARY);
	imshow("�������ֵ�ָ��㷨", segmented);
	waitKey(10);
}
/**
* @brief ��ˮ���㷨
* @param currentframe ����ͼ��
*/

Vec3b RandomColor(int value) //���������ɫ����</span>
{
	value = value % 255;  //����0~255�������
	RNG rng;
	int aa = rng.uniform(0, value);
	int bb = rng.uniform(0, value);
	int cc = rng.uniform(0, value);
	return Vec3b(aa, bb, cc);
}

void watershedAlgorithm(InputArray image, InputOutputArray markers) {
	// ���� 1: ͼ��ҶȻ�
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// ���� 2: ����ͼ���ݶ�
	Mat gradX, gradY;
	Sobel(gray, gradX, CV_32F, 1, 0, 3);
	Sobel(gray, gradY, CV_32F, 0, 1, 3);

	// �����ݶȷ��Ⱥͷ���
	Mat gradient, gradientMagnitude;
	magnitude(gradX, gradY, gradientMagnitude);
	convertScaleAbs(gradientMagnitude, gradient);

	// ���� 3: ���ݶ�ֵ����
	std::vector<Point> sortedPoints;
	for (int y = 0; y < gradient.rows; ++y) {
		for (int x = 0; x < gradient.cols; ++x) {
			sortedPoints.push_back(Point(x, y));
		}
	}

	std::sort(sortedPoints.begin(), sortedPoints.end(), [&](const Point& a, const Point& b) {
		return gradient.at<uchar>(a) < gradient.at<uchar>(b);
		});

	// ���� 4-7: ��ˮ���㷨
	std::queue<Point> seeds;
	int currentLabel = 0;

	Mat markersMat = markers.getMat();

	for (const Point& point : sortedPoints) {
		if (markersMat.at<int>(point) == -1) {
			// �µļ�С���򣬿�ʼɨ��
			seeds.push(point);
			markersMat.at<int>(point) = ++currentLabel;

			while (!seeds.empty()) {
				Point current = seeds.front();
				seeds.pop();

				for (int ny = -1; ny <= 1; ++ny) {
					for (int nx = -1; nx <= 1; ++nx) {
						int y = current.y + ny;
						int x = current.x + nx;

						if (y >= 0 && y < markersMat.rows && x >= 0 && x < markersMat.cols) {
							if (markersMat.at<int>(y, x) == -1 && gradient.at<uchar>(y, x) == gradient.at<uchar>(current)) {
								markersMat.at<int>(y, x) = currentLabel;
								seeds.push(Point(x, y));
							}
						}
					}
				}
			}
		}
	}
}


void watershedSegmentation(Mat& _img)
{
	// ������ͼ�񱣴浽����image��
	Mat image = _img;
	Mat GreyImage;
	// ת���ɻҶ�ͼ��
	cvtColor(image, GreyImage, COLOR_RGB2GRAY);
	// ��˹�˲��ͱ�Ե���
	GaussianBlur(GreyImage, GreyImage, Size(3, 3), 0, 0);
	Canny(GreyImage, GreyImage, 50, 150, 3);
	
	// ����洢������Ϣ������ 
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	
	// Ѱ�һҶ�ͼ���е�������Ϣ
	findContours(GreyImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	
	// ���ɷ�ˮ��ͼ�񣬳�ʼ��Ϊȫ�����
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);
	
	// ����Mark�������ڱ������
	Mat Mark(image.size(), CV_32S);
	Mark = Scalar::all(0);
	int index = 0, n1 = 0;
	// ������������Mark���б�ǣ��൱��Ϊ��ͬ�������������עˮ��
	for (; index >= 0; index = hierarchy[index][0], n1++)
	{
		// ��marks���б�ǣ��Բ�ͬ������������б�ţ��൱������עˮ�㣬�ж������������ж���עˮ��
		drawContours(Mark, contours, index, Scalar::all(n1 + 1), 1, 8, hierarchy);	// ��������
		drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);	// ��������
	}

	// ��Mark����ת��Ϊ8λͼ�񣬲�Ӧ�÷�ˮ���㷨
	Mat biaojiShows;
	convertScaleAbs(Mark, biaojiShows);
	watershedAlgorithm(image, Mark);
	//watershed(image, Mark);
	
	// ����w1�������ڴ洢��ˮ���㷨���ͼ��
	Mat w1;	// ��ˮ����ͼ��
	convertScaleAbs(Mark, w1);	// ת����8λͼ��
	
	// ����PerspectiveImage����������ʾ��ˮ����ͼ��
	 // ����Mark�����ֵΪPerspectiveImage�����е�ÿ�����ظ��費ͬ����ɫ
	Mat PerspectiveImage = Mat::zeros(image.size(), CV_8UC3);
	for (int i = 0; i < Mark.rows; i++)
	{
		for (int j = 0; j < Mark.cols; j++)
		{
			// ��ȡ��ǰ���ص�Markֵ
			int index = Mark.at<int>(i, j);
			// ���MarkֵΪ-1����ʾΪעˮ�㣬������������Ϊ��ɫ
			if (Mark.at<int>(i, j) == -1)
			{
				PerspectiveImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			// ���򣬸���עˮ��ı�ǩΪ�����ظ��������ɫ
			else
			{
				PerspectiveImage.at<Vec3b>(i, j) = RandomColor(index);
			}
		}
	}
	// ����wshed���󣬽�ԭʼͼ���PerspectiveImage��Ȩ�ص��ӣ�������ʾ���ս��
	Mat wshed;
	addWeighted(image, 0.4, PerspectiveImage, 0.6, 0, wshed);
	// �ڴ�������ʾ��ˮ���㷨�Ľ��ͼ��
	imshow("��ˮ���㷨", wshed);
	waitKey(10);
}

void momentInvariantThresholding(const InputArray& image) {
	// ������ͼ��ת��Ϊ�Ҷ�ͼ��
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);

	// ����ͼ���ǰ���׾�
	Moments moment = moments(grayImage, true);
	double m1 = moment.m10 / moment.m00;
	double m2 = moment.m01 / moment.m00;
	double m3 = (moment.m20 / moment.m00) - m1 * m1;

	// �Ƶ�p1����m1��m2��m3�Ĺ�ϵʽ
	double p1 = m3 - 3 * m2 * m1 + 2 * m1 * m1 * m1;

	// ����ÿ���Ҷ�ֵ������ֱ��ͼ
	int histogram[256] = { 0 };
	for (int i = 0; i < grayImage.rows; ++i) {
		for (int j = 0; j < grayImage.cols; ++j) {
			int pixelValue = grayImage.at<uchar>(i, j);
			histogram[pixelValue]++;
		}
	}

	// ��ʼ����ֵѡ��ı���
	double minDifference = DBL_MAX;
	int threshold = 0;

	// �������ܵ���ֵֵ��ѡ��ʹ�ò�ֵp0-p1��С����ֵ
	for (int k = 0; k < 256; ++k) {
		double p0 = 0, p1 = 0;
		for (int i = 0; i <= k; ++i) {
			p0 += histogram[i];
		}
		for (int i = k + 1; i < 256; ++i) {
			p1 += histogram[i];
		}

		// ���㵱ǰ��ֵ��Ӧ�Ĳ�ֵ
		double difference = std::abs(p0 - p1);

		// ������С��ֵ�Ͷ�Ӧ����ֵ
		if (difference < minDifference) {
			minDifference = difference;
			threshold = k;
		}
	}

	// ����ѡ�����ֵ����ͼ��ָ�
	Mat segmentedImage = grayImage.clone();
	for (int i = 0; i < segmentedImage.rows; ++i) {
		for (int j = 0; j < segmentedImage.cols; ++j) {
			// ������ֵ��ͼ���Ϊ�������򣬺�ɫ�Ͱ�ɫ
			if (segmentedImage.at<uchar>(i, j) < threshold) {
				segmentedImage.at<uchar>(i, j) = 0;
			}
			else {
				segmentedImage.at<uchar>(i, j) = 255;
			}
		}
	}

	// ��ʾ��ֵ�ָ���ͼ��
	imshow("�ز��䷨��ֵ�ָ�ͼ��", segmentedImage);
	waitKey(10);
}

//����Ӧ��ֵ�ָ�
void RegionSegAdaptive(Mat frame)
{
	cvtColor(frame, frame, COLOR_BGR2GRAY);
	int nLocAvg;
	//���ϣ��ֳ�4����
	nLocAvg = 0;
	for (int i = 0; i < frame.rows / 2; i++)
	{
		uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
		for (int j = 0; j < frame.cols / 2; j++)
		{
			nLocAvg += data[j];
		}
	}
	nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));//���ÿһ�������ƽ����ֵ
	for (int i = 0; i < frame.rows / 2; i++)
	{
		uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
		for (int j = 0; j < frame.cols / 2; j++)
		{
			if (data[j] < nLocAvg)//��ֵ�ָ�
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
		for (int j = 0; j < frame.cols / 2; j++)
		{
			nLocAvg += data[j];
		}
	}
	nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));
	for (int i = frame.rows / 2; i < frame.rows; i++)
	{
		uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
		for (int j = 0; j < frame.cols / 2; j++)
		{
			if (data[j] < nLocAvg)
				data[j] = 0;
			else
				data[j] = 255;
		}
	}

	//����
	nLocAvg = 0;
	for (int i = 0; i < frame.rows / 2; i++)
	{
		uchar* data = frame.ptr<uchar>(i); //��ȡÿһ�е�ָ��
		for (int j = frame.cols / 2; j < frame.cols; j++)
		{
			nLocAvg += data[j];
		}
	}
	nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));
	for (int i = 0; i < frame.rows / 2; i++)
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
	imshow("����Ӧ��ֵ�ָ�",frame);
	waitKey(10);
	/*return frame;*/
}

//��������ֵ�ָ
void iterative(Mat frame)
{
	cvtColor(frame, frame, COLOR_BGR2GRAY);
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
	int iNewThreshold = (iMinGrayValue + iMaxGrayValue) / 2;//��ʼ��ֵ

	long totalValue_f = 0;//����ǰһ����ĻҶ�ֵ��
	long meanValue_f;//����ǰһ����ĻҶ�ƽ��

	long totalValue_b = 0;//�����һ����ĻҶ�ֵ��
	long meanValue_b;//�����һ����ĻҶ�ƽ��

	long lp1 = 0, lp2 = 0;//���ڼ�������Ҷ�ƽ��ֵ���м����

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
	/*cout << "�� " << iter << " ���ؼ�֡�ĵ���������ֵΪ  " << iNewThreshold << endl;
	iter++;*/
	Mat after1;
	threshold(frame, after1, iNewThreshold, 255, THRESH_BINARY);//��ֵ�ָ�
	imshow("��������ֵ�ָ�ͼ��", after1);
	waitKey(0);
	/*return after1;*/
}


//�˵�
void menu(Mat currentframe, Mat threshImage, int in) {
	switch (in) {
	case 1: {
		/*cvtColor(currentframe, currentframe, COLOR_BGR2GRAY);
		threshBasic(currentframe);*/
		OTSU(currentframe);
		break;
	}
	case 2: {
		maxEntropyThreshold(currentframe);
		break;
	}
	case 3: {
		watershedSegmentation(currentframe);
		break;
	}
	case 4:
	{
		momentInvariantThresholding(currentframe);
		break;
	}
	case 5:
	{
		iterative(currentframe);
		break;
	}
	case 6:
	{
		RegionSegAdaptive(currentframe);
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
		cout << "1.OTSU��ֵ�ָ�" << endl;
		cout << "2.������㷨" << endl;
		cout << "3.�����ˮ���㷨" << endl;
		cout << "4.�ز��䷨��ֵ�ָ�" << endl;
		cout << "5.��������ֵ�ָ�" << endl;
		cout << "6.����Ӧ��ֵ�ָ�" << endl;
			cin >> in;
		flag = 0;
	}
	VideoCapture cap;
	cap.open("E:\\������Ƶʵ����Ƶ\\1ʵ��һ��Ƶ\\exp1(1).avi");
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
	while (true) {
		//��ʾ��Ƶ
		Mat frame1;
		// ��ȡ����֡
		bool ok = cap.read(frame);
		if (!ok) break;
		imshow("��Ƶ", frame);
		k = waitKey(10);

		cap >> frame1;	//��ȡ��Ƶ��ÿ֡
		if (frame1.empty())
		{
			cout << "frame1 is empty!" << endl;
			break;
		}
		
		//Mat markers(frame1.size(), CV_32S, Scalar::all(0));
		tempframe = frame1;
		framenum++;
		// ��һ֡������ǵ�һ֡���ͽ���ǰ֡��ֵ��ǰһ֡
		if (framenum == 1) {
			keyframe = tempframe;//��һ֡��Ϊһ���ؼ�֡
			imshow("�ؼ�֡", keyframe);//�ؼ�֡1
			cvtColor(keyframe, currentframe, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
			Mat threshImage;
			//imshow("�Ҷ�ͼ", currentframe);//��ʾ�Ҷ�ͼ
			menu(keyframe, threshImage, in);
		}
		//��һ֮֡�������֡
		if (framenum >= 2) {
			cvtColor(tempframe, currentframe, COLOR_BGR2GRAY);//��ǰ֡ת�Ҷ�ͼ
			cvtColor(keyframe, previousframe, COLOR_BGR2GRAY);//�ο�֡ת�Ҷ�ͼ
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
			float thresholdValue = sum / (diff.rows * diff.cols);
			if (thresholdValue > 40) {
				// ��ֵ
				imshow("�ؼ�֡", tempframe);
				keyframe = tempframe;
				Mat threshImage;
				//imshow("�Ҷ�ͼ", currentframe);
				menu(keyframe, threshImage, in);
			}
		}
	}
	return 0;
}

