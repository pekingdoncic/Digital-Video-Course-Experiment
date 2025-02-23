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
//     ��ʼ������
//    int bestThreshold = 0;
//    double maxEntropy = 0;
//
//     ����ÿ�����ܵ���ֵ
//    for (int threshold = 0; threshold < 256; ++threshold) {
//        double entropy = calculateEntropy(image, threshold);
//
//         ���������ֵ�Ͷ�Ӧ����ֵ
//        if (entropy > maxEntropy) {
//            maxEntropy = entropy;
//            bestThreshold = threshold;
//        }
//    }
//
//     ���������ֵ��ͼ����ж�ֵ��
//    Mat segmented;
//    threshold(image, segmented, bestThreshold, 255, THRESH_BINARY);
//
//    return segmented;
//}
//
//int main() {
//     ��ȡͼ��
//    Mat image = imread("C:\\Users\\R9000X\\Desktop\\��������2.jpg", IMREAD_GRAYSCALE);
//
//    if (image.empty()) {
//        cout << "Could not open or find the image." << endl;
//        return -1;
//    }
//
//     ���ú������зָ�
//    Mat segmented = maxEntropyThresholdSegmentation(image);
//
//     ��ʾԭʼͼ��ͷָ���
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

// �Զ��巽���˲�����
void boxFilterCustom(const Mat& input, Mat& output, int kernelSize)
{
    output = Mat::zeros(input.size(), input.type());

    // ����ͼ���е�ÿ������
    for (int i = 0; i < input.rows; ++i)
    {
        for (int j = 0; j < input.cols; ++j)
        {
            int sum = 0;
            int count = 0;

            // ���������ֵ
            for (int m = -kernelSize / 2; m <= kernelSize / 2; ++m)
            {
                for (int n = -kernelSize / 2; n <= kernelSize / 2; ++n)
                {
                    int row = i + m;
                    int col = j + n;

                    // �߽���
                    if (row >= 0 && row < input.rows && col >= 0 && col < input.cols)
                    {
                        sum += input.at<uchar>(row, col);
                        ++count;
                    }
                }
            }

            // �����ֵ���������ͼ��
            output.at<uchar>(i, j) = static_cast<uchar>(sum / count);
        }
    }
}

int main()
{
    // ��ȡͼ��
    Mat image = imread("C:\\Users\\R9000X\\Desktop\\��������2.jpg", IMREAD_GRAYSCALE);

    // ���ͼ���Ƿ�ɹ���ȡ
    if (image.empty())
    {
        cerr << "Error: Could not read the image." << endl;
        return -1;
    }

    // �Զ��巽���˲�
    Mat boxFiltered;
    boxFilterCustom(image, boxFiltered, 5);  // ʹ��5x5�ķ����˲���

    // ��ʾԭʼͼ����Զ��巽���˲����ͼ��
    imshow("Original Image", image);
    imshow("Custom Box Filtered Image", boxFiltered);
    waitKey(0);

    return 0;
}

