
#include <opencv2/opencv.hpp>
#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <queue>
using namespace std;
using namespace cv;

/**
* @brief 计算直方图，两个峰值，然后波谷阈值法
* @param currentframe 输入图像
*/

void threshBasic(const Mat& currentframe)
{
	
	// 计算灰度直方图，zero函数用于创建一个大小为256的Mat对象，类型为CV_32F，全部初始化为0
	Mat histogram = Mat::zeros(Size(256, 1), CV_32SC1);
	// 获取行高
	int rows = currentframe.rows;
	int cols = currentframe.cols;
	// 遍历图像，计算灰度直方图
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int index = int(currentframe.at<uchar>(i, j));
			histogram.at<int>(0, index) += 1;
		}
	}
	// 找到灰度直方图最大峰值对应的灰度值
	Point peak1;
	// 在数组中找到全局最小和最大值
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
	//找到两个峰值之间的最小值对应的灰度值，作为阈值
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
	* src:输入图像
	* dst:输出图像
	* thresh:阈值
	* maxval:输出图像中最大值
	* type:阈值类型，THRESH_BINARY 指二值化
	*/
	Mat threshImage;
	threshold(currentframe, threshImage, thresh, 255, THRESH_BINARY);
	imshow("直方图阈值分割结果", threshImage);
}

/**
* @brief 最大熵法
* @param currentframe 输入图像
* @param threshImage 输出图像
*/

void maxEntropyThresholdSegmentation(Mat currentframe)
{
	cvtColor(currentframe, currentframe, COLOR_BGR2GRAY);
	int hist_t[256] = { 0 }; //每个像素值个数
	int index = 0;//最大熵对应的灰度
	double Property = 0.0;// 像素占有比率
	double maxEntropy = -1.0;//最大熵
	double frontEntropy = 0.0;//前景熵
	double backEntropy = 0.0;//背景熵
	int sum_t = 0;	//像素总数
	int nCol = currentframe.cols;//每行的像素数
	// 计算每个像素值的个数
	for (int i = 0; i < currentframe.rows; i++)
	{
		uchar* pData = currentframe.ptr<uchar>(i);
		for (int j = 0; j < nCol; ++j)
		{
			++sum_t;
			hist_t[pData[j]] += 1;
		}
	}
	// 计算最大熵
	for (int i = 0; i < 256; i++)
	{
		// 背景像素数
		double sum_back = 0;
		for (int j = 0; j < i; j++)
		{
			sum_back += hist_t[j];
		}

		//背景熵
		for (int j = 0; j < i; j++)
		{
			if (hist_t[j] != 0)
			{
				Property = hist_t[j] / sum_back;
				backEntropy += -Property * logf((float)Property);
			}
		}
		//前景熵
		for (int k = i; k < 256; k++)
		{
			if (hist_t[k] != 0)
			{
				Property = hist_t[k] / (sum_t - sum_back);
				frontEntropy += -Property * logf((float)Property);
			}
		}

		if (frontEntropy + backEntropy > maxEntropy)// 求最大熵
		{
			maxEntropy = frontEntropy + backEntropy;
			index = i;
		}
		frontEntropy = 0.0;	// 前景熵清零
		backEntropy = 0.0;	// 背景熵清零
	}
	Mat threshImage;
	threshold(currentframe, threshImage, index, 255, 0); //进行阈值分割
	imshow("最大熵法", threshImage);
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
//	// 将InputArray转换为Mat
//	Mat image = inputImage.getMat();
//
//	// 计算最大熵阈值
//	int threshold = findMaxEntropyThreshold(image);
//
//	cout << "Optimal Threshold: " << threshold << endl;
//
//	// 应用阈值
//	Mat segmentedImage;
//	applyThreshold(image, segmentedImage, threshold);
//
//	imshow("最大熵阈值分割算法", segmentedImage);
//	waitKey(1);
//	/*return segmentedImage;*/
//}


void OTSU(Mat frame)
{
	// 将输入帧转换为灰度图
	cvtColor(frame, frame, COLOR_BGR2GRAY);

	// 初始化变量
	long i, j, t;
	int iThreshold, iNewThreshold, iMaxGrayValue, iMinGrayValue, iMean1GrayValue, iMean2GrayValue;
	iMaxGrayValue = 0;
	iMinGrayValue = 255;

	// 前景和背景像素比例
	double w0, w1;

	// 方差
	double G = 0, tempG = 0;

	// 用于计算区域灰度平均值的变量
	long lP1, lP2, lS1, lS2;

	// 直方图，存储像素强度频率
	long lHistogram[256] = { 0 };

	// 计算直方图
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

	// 总像素数
	int n = frame.rows * frame.cols;

	// 遍历可能的阈值值
	for (t = iMinGrayValue; t < iMaxGrayValue; t++)
	{
		iNewThreshold = t;
		lP1 = 0; lP2 = 0; lS1 = 0; lS2 = 0;

		// 计算两个区域的灰度平均值
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

		// 计算加权平均值
		double mean = w0 * iMean1GrayValue + w1 * iMean2GrayValue;

		// 计算方差
		G = (double)w0 * (iMean1GrayValue - mean) * (iMean1GrayValue - mean) + w1 * (iMean2GrayValue - mean) * (iMean2GrayValue - mean);

		// 如果方差更大，则更新阈值
		if (G > tempG)
		{
			tempG = G;
			iThreshold = iNewThreshold;
		}
	}

	// 使用计算得到的阈值对图像进行二值化处理
	Mat after2;
	threshold(frame, after2, iThreshold, 255, THRESH_BINARY);

	// 显示结果
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

	// 初始化变量
	int bestThreshold = 0;
	double maxEntropy = 0;

	// 遍历每个可能的阈值
	for (int threshold = 0; threshold < 256; ++threshold) {
		double entropy = calculateEntropy(image, threshold);

		// 更新最大熵值和对应的阈值
		if (entropy > maxEntropy) {
			maxEntropy = entropy;
			bestThreshold = threshold;
		}
	}

	// 根据最佳阈值对图像进行二值化
	Mat segmented;
	
	cvtColor(image, image, COLOR_BGR2GRAY);
	threshold(image, segmented, bestThreshold, 255, THRESH_BINARY);
	imshow("最大熵阈值分割算法", segmented);
	waitKey(10);
}
/**
* @brief 分水岭算法
* @param currentframe 输入图像
*/

Vec3b RandomColor(int value) //生成随机颜色函数</span>
{
	value = value % 255;  //生成0~255的随机数
	RNG rng;
	int aa = rng.uniform(0, value);
	int bb = rng.uniform(0, value);
	int cc = rng.uniform(0, value);
	return Vec3b(aa, bb, cc);
}

void watershedAlgorithm(InputArray image, InputOutputArray markers) {
	// 步骤 1: 图像灰度化
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// 步骤 2: 计算图像梯度
	Mat gradX, gradY;
	Sobel(gray, gradX, CV_32F, 1, 0, 3);
	Sobel(gray, gradY, CV_32F, 0, 1, 3);

	// 计算梯度幅度和方向
	Mat gradient, gradientMagnitude;
	magnitude(gradX, gradY, gradientMagnitude);
	convertScaleAbs(gradientMagnitude, gradient);

	// 步骤 3: 对梯度值排序
	std::vector<Point> sortedPoints;
	for (int y = 0; y < gradient.rows; ++y) {
		for (int x = 0; x < gradient.cols; ++x) {
			sortedPoints.push_back(Point(x, y));
		}
	}

	std::sort(sortedPoints.begin(), sortedPoints.end(), [&](const Point& a, const Point& b) {
		return gradient.at<uchar>(a) < gradient.at<uchar>(b);
		});

	// 步骤 4-7: 分水岭算法
	std::queue<Point> seeds;
	int currentLabel = 0;

	Mat markersMat = markers.getMat();

	for (const Point& point : sortedPoints) {
		if (markersMat.at<int>(point) == -1) {
			// 新的极小区域，开始扫描
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
	// 将输入图像保存到变量image中
	Mat image = _img;
	Mat GreyImage;
	// 转换成灰度图像
	cvtColor(image, GreyImage, COLOR_RGB2GRAY);
	// 高斯滤波和边缘检测
	GaussianBlur(GreyImage, GreyImage, Size(3, 3), 0, 0);
	Canny(GreyImage, GreyImage, 50, 150, 3);
	
	// 定义存储轮廓信息的向量 
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	
	// 寻找灰度图像中的轮廓信息
	findContours(GreyImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	
	// 生成分水岭图像，初始化为全零矩阵
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);
	
	// 创建Mark矩阵，用于标记轮廓
	Mat Mark(image.size(), CV_32S);
	Mark = Scalar::all(0);
	int index = 0, n1 = 0;
	// 遍历轮廓并对Mark进行标记，相当于为不同区域的轮廓设置注水点
	for (; index >= 0; index = hierarchy[index][0], n1++)
	{
		// 对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点
		drawContours(Mark, contours, index, Scalar::all(n1 + 1), 1, 8, hierarchy);	// 绘制轮廓
		drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);	// 绘制轮廓
	}

	// 将Mark矩阵转换为8位图像，并应用分水岭算法
	Mat biaojiShows;
	convertScaleAbs(Mark, biaojiShows);
	watershedAlgorithm(image, Mark);
	//watershed(image, Mark);
	
	// 创建w1矩阵，用于存储分水岭算法后的图像
	Mat w1;	// 分水岭后的图像
	convertScaleAbs(Mark, w1);	// 转换成8位图像
	
	// 创建PerspectiveImage矩阵，用于显示分水岭后的图像
	 // 根据Mark矩阵的值为PerspectiveImage矩阵中的每个像素赋予不同的颜色
	Mat PerspectiveImage = Mat::zeros(image.size(), CV_8UC3);
	for (int i = 0; i < Mark.rows; i++)
	{
		for (int j = 0; j < Mark.cols; j++)
		{
			// 获取当前像素的Mark值
			int index = Mark.at<int>(i, j);
			// 如果Mark值为-1，表示为注水点，将该像素设置为白色
			if (Mark.at<int>(i, j) == -1)
			{
				PerspectiveImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			// 否则，根据注水点的标签为该像素赋予随机颜色
			else
			{
				PerspectiveImage.at<Vec3b>(i, j) = RandomColor(index);
			}
		}
	}
	// 创建wshed矩阵，将原始图像和PerspectiveImage按权重叠加，用于显示最终结果
	Mat wshed;
	addWeighted(image, 0.4, PerspectiveImage, 0.6, 0, wshed);
	// 在窗口中显示分水岭算法的结果图像
	imshow("分水岭算法", wshed);
	waitKey(10);
}

void momentInvariantThresholding(const InputArray& image) {
	// 将输入图像转换为灰度图像
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);

	// 计算图像的前三阶矩
	Moments moment = moments(grayImage, true);
	double m1 = moment.m10 / moment.m00;
	double m2 = moment.m01 / moment.m00;
	double m3 = (moment.m20 / moment.m00) - m1 * m1;

	// 推导p1关于m1，m2，m3的关系式
	double p1 = m3 - 3 * m2 * m1 + 2 * m1 * m1 * m1;

	// 遍历每个灰度值，计算直方图
	int histogram[256] = { 0 };
	for (int i = 0; i < grayImage.rows; ++i) {
		for (int j = 0; j < grayImage.cols; ++j) {
			int pixelValue = grayImage.at<uchar>(i, j);
			histogram[pixelValue]++;
		}
	}

	// 初始化阈值选择的变量
	double minDifference = DBL_MAX;
	int threshold = 0;

	// 遍历可能的阈值值，选择使得差值p0-p1最小的阈值
	for (int k = 0; k < 256; ++k) {
		double p0 = 0, p1 = 0;
		for (int i = 0; i <= k; ++i) {
			p0 += histogram[i];
		}
		for (int i = k + 1; i < 256; ++i) {
			p1 += histogram[i];
		}

		// 计算当前阈值对应的差值
		double difference = std::abs(p0 - p1);

		// 更新最小差值和对应的阈值
		if (difference < minDifference) {
			minDifference = difference;
			threshold = k;
		}
	}

	// 根据选择的阈值进行图像分割
	Mat segmentedImage = grayImage.clone();
	for (int i = 0; i < segmentedImage.rows; ++i) {
		for (int j = 0; j < segmentedImage.cols; ++j) {
			// 根据阈值将图像分为两个区域，黑色和白色
			if (segmentedImage.at<uchar>(i, j) < threshold) {
				segmentedImage.at<uchar>(i, j) = 0;
			}
			else {
				segmentedImage.at<uchar>(i, j) = 255;
			}
		}
	}

	// 显示阈值分割后的图像
	imshow("矩不变法阈值分割图像", segmentedImage);
	waitKey(10);
}

//自适应阈值分割
void RegionSegAdaptive(Mat frame)
{
	cvtColor(frame, frame, COLOR_BGR2GRAY);
	int nLocAvg;
	//左上，分成4部分
	nLocAvg = 0;
	for (int i = 0; i < frame.rows / 2; i++)
	{
		uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
		for (int j = 0; j < frame.cols / 2; j++)
		{
			nLocAvg += data[j];
		}
	}
	nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));//求得每一个区域的平均阈值
	for (int i = 0; i < frame.rows / 2; i++)
	{
		uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
		for (int j = 0; j < frame.cols / 2; j++)
		{
			if (data[j] < nLocAvg)//阈值分割
				data[j] = 0;
			else
				data[j] = 255;
		}
	}
	//左下
	nLocAvg = 0;
	for (int i = frame.rows / 2; i < frame.rows; i++)
	{
		uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
		for (int j = 0; j < frame.cols / 2; j++)
		{
			nLocAvg += data[j];
		}
	}
	nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));
	for (int i = frame.rows / 2; i < frame.rows; i++)
	{
		uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
		for (int j = 0; j < frame.cols / 2; j++)
		{
			if (data[j] < nLocAvg)
				data[j] = 0;
			else
				data[j] = 255;
		}
	}

	//右上
	nLocAvg = 0;
	for (int i = 0; i < frame.rows / 2; i++)
	{
		uchar* data = frame.ptr<uchar>(i); //获取每一行的指针
		for (int j = frame.cols / 2; j < frame.cols; j++)
		{
			nLocAvg += data[j];
		}
	}
	nLocAvg /= ((frame.rows / 2) * (frame.cols / 2));
	for (int i = 0; i < frame.rows / 2; i++)
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
	imshow("自适应阈值分割",frame);
	waitKey(10);
	/*return frame;*/
}

//迭代法阈值分割：
void iterative(Mat frame)
{
	cvtColor(frame, frame, COLOR_BGR2GRAY);
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
	int iNewThreshold = (iMinGrayValue + iMaxGrayValue) / 2;//初始阈值

	long totalValue_f = 0;//计算前一区域的灰度值和
	long meanValue_f;//计算前一区域的灰度平均

	long totalValue_b = 0;//计算后一区域的灰度值和
	long meanValue_b;//计算后一区域的灰度平均

	long lp1 = 0, lp2 = 0;//用于计算区域灰度平均值的中间变量

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
	/*cout << "第 " << iter << " 个关键帧的迭代法的阈值为  " << iNewThreshold << endl;
	iter++;*/
	Mat after1;
	threshold(frame, after1, iNewThreshold, 255, THRESH_BINARY);//阈值分割
	imshow("迭代法阈值分割图像", after1);
	waitKey(0);
	/*return after1;*/
}


//菜单
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
		cout << "输入错误" << endl;
		break;
	}
	}
}

int main() {
	int in;
	int flag = 1;
	while (flag) {
		cout << "选择值分割算法：" << endl;
		cout << "1.OTSU阈值分割" << endl;
		cout << "2.最大熵算法" << endl;
		cout << "3.经典分水岭算法" << endl;
		cout << "4.矩不变法阈值分割" << endl;
		cout << "5.迭代法阈值分割" << endl;
		cout << "6.自适应阈值分割" << endl;
			cin >> in;
		flag = 0;
	}
	VideoCapture cap;
	cap.open("E:\\数字视频实验视频\\1实验一视频\\exp1(1).avi");
	// 打开失败
	if (!cap.isOpened()) {
		cout << "无法读取视频" << endl;
		system("pause");
		return -1;
	}
	int k = -1;

	Mat tempframe, currentframe, keyframe, previousframe, diff;// 当前帧，前一帧，关键帧
	Mat frame;
	int framenum = 0;
	int dur_time = 1;
	while (true) {
		//显示视频
		Mat frame1;
		// 读取所有帧
		bool ok = cap.read(frame);
		if (!ok) break;
		imshow("视频", frame);
		k = waitKey(10);

		cap >> frame1;	//读取视频的每帧
		if (frame1.empty())
		{
			cout << "frame1 is empty!" << endl;
			break;
		}
		
		//Mat markers(frame1.size(), CV_32S, Scalar::all(0));
		tempframe = frame1;
		framenum++;
		// 第一帧，如果是第一帧，就将当前帧赋值给前一帧
		if (framenum == 1) {
			keyframe = tempframe;//第一帧作为一个关键帧
			imshow("关键帧", keyframe);//关键帧1
			cvtColor(keyframe, currentframe, COLOR_BGR2GRAY);//转换为灰度图
			Mat threshImage;
			//imshow("灰度图", currentframe);//显示灰度图
			menu(keyframe, threshImage, in);
		}
		//第一帧之后的所有帧
		if (framenum >= 2) {
			cvtColor(tempframe, currentframe, COLOR_BGR2GRAY);//当前帧转灰度图
			cvtColor(keyframe, previousframe, COLOR_BGR2GRAY);//参考帧转灰度图
			absdiff(currentframe, previousframe, diff);// 两帧差分

			float sum = 0.0;
			// 计算两帧差分的累计
			for (int i = 0; i < diff.rows; i++) {
				uchar* data = diff.ptr<uchar>(i);
				for (int j = 0; j < diff.cols; j++)
				{
					sum += data[j];
				}
			}
			float thresholdValue = sum / (diff.rows * diff.cols);
			if (thresholdValue > 40) {
				// 阈值
				imshow("关键帧", tempframe);
				keyframe = tempframe;
				Mat threshImage;
				//imshow("灰度图", currentframe);
				menu(keyframe, threshImage, in);
			}
		}
	}
	return 0;
}

