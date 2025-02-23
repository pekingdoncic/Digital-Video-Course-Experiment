//#include <iostream>
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//using namespace std;
//
//double calculateEntropy(const Mat& img, int threshold) {
//    int totalPixels = img.rows * img.cols;
//    int countBelow = 0, countAbove = 0;
//    double entropyBelow = 0, entropyAbove = 0;
//
//    for (int i = 0; i < img.rows; ++i) {
//        for (int j = 0; j < img.cols; ++j) {
//            int pixelValue = static_cast<int>(img.at<uchar>(i, j));
//            if (pixelValue < threshold) {
//                countBelow++;
//            }
//            else {
//                countAbove++;
//            }
//        }
//    }
//
//    for (int i = 0; i < 256; ++i) {
//        if (i < threshold) {
//            double probability = static_cast<double>(countBelow) / totalPixels;
//            if (probability > 0) {
//                entropyBelow -= probability * log2(probability);
//            }
//        }
//        else {
//            double probability = static_cast<double>(countAbove) / totalPixels;
//            if (probability > 0) {
//                entropyAbove -= probability * log2(probability);
//            }
//        }
//    }
//
//    return entropyBelow + entropyAbove;
//}
//
//Mat maxEntropyThresholdSegmentation(const InputArray& inputImage) {
//    Mat image = inputImage.getMat();
//
//     初始化变量
//    int bestThreshold = 0;
//    double maxEntropy = 0;
//
//     遍历每个可能的阈值
//    for (int threshold = 0; threshold < 256; ++threshold) {
//        double entropy = calculateEntropy(image, threshold);
//
//         更新最大熵值和对应的阈值
//        if (entropy > maxEntropy) {
//            maxEntropy = entropy;
//            bestThreshold = threshold;
//        }
//    }
//
//     根据最佳阈值对图像进行二值化
//    Mat segmented;
//    threshold(image, segmented, bestThreshold, 255, THRESH_BINARY);
//
//    return segmented;
//}
//
//int main() {
//     读取图像
//    Mat image = imread("C:\\Users\\R9000X\\Desktop\\金属材质2.jpg", IMREAD_GRAYSCALE);
//
//    if (image.empty()) {
//        cout << "Could not open or find the image." << endl;
//        return -1;
//    }
//
//     调用函数进行分割
//    Mat segmented = maxEntropyThresholdSegmentation(image);
//
//     显示原始图像和分割结果
//    imshow("Original Image", image);
//    imshow("Segmented Image", segmented);
//    waitKey(0);
//
//    return 0;
//}
//
//
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 自定义方框滤波函数
void boxFilterCustom(const Mat& input, Mat& output, int kernelSize)
{
    output = Mat::zeros(input.size(), input.type());

    // 遍历图像中的每个像素
    for (int i = 0; i < input.rows; ++i)
    {
        for (int j = 0; j < input.cols; ++j)
        {
            int sum = 0;
            int count = 0;

            // 计算邻域均值
            for (int m = -kernelSize / 2; m <= kernelSize / 2; ++m)
            {
                for (int n = -kernelSize / 2; n <= kernelSize / 2; ++n)
                {
                    int row = i + m;
                    int col = j + n;

                    // 边界检查
                    if (row >= 0 && row < input.rows && col >= 0 && col < input.cols)
                    {
                        sum += input.at<uchar>(row, col);
                        ++count;
                    }
                }
            }

            // 计算均值并更新输出图像
            output.at<uchar>(i, j) = static_cast<uchar>(sum / count);
        }
    }
}

int main()
{
    // 读取图像
    Mat image = imread("C:\\Users\\R9000X\\Desktop\\金属材质2.jpg", IMREAD_GRAYSCALE);

    // 检查图像是否成功读取
    if (image.empty())
    {
        cerr << "Error: Could not read the image." << endl;
        return -1;
    }

    // 自定义方框滤波
    Mat boxFiltered;
    boxFilterCustom(image, boxFiltered, 5);  // 使用5x5的方框滤波核

    // 显示原始图像和自定义方框滤波后的图像
    imshow("Original Image", image);
    imshow("Custom Box Filtered Image", boxFiltered);
    waitKey(0);

    return 0;
}

