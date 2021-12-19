#include <iostream>
#include "opencv2/opencv.hpp"
#include "omp.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

Mat kernel =
(Mat_<double>(3, 3) <<
    -1.0, -1.0, -1.0,
    -1.0, 9.0, -1.0,
    -1.0, -1.0, -1.0);

int clip(int value) {
    if (value >= 255)
        return 255;
    else if (value <= 0)
        return 0;
    else
        return value;
}

Mat sharpen_img(Mat& input) {
    Mat output = input.clone();
    Mat image = input;
    Mat processed_image[3] =
    { Mat::zeros(image.size(), CV_8UC1), Mat::zeros(image.size(), CV_8UC1), Mat::zeros(image.size(), CV_8UC1) };

    int offset = 1;

    // Сплит каналов
    Mat split_channels[3];
    split(image, split_channels);
    

    // Основной цикл
    for (int r = offset; r < image.rows - offset; r++) {
        for (int c = offset; c < image.cols - offset; c++) {
            double rgb[3] = { 0, 0, 0 };
            for (int x = 0; x < kernel.rows; x++) {
                for (int y = 0; y < kernel.cols; y++) {
                    int rn = r + x - offset;
                    int cn = c + y - offset;
                    rgb[0] += split_channels[0].at<uint8_t>(rn, cn) * kernel.at<double>(x, y);
                    rgb[1] += split_channels[1].at<uint8_t>(rn, cn) * kernel.at<double>(x, y);
                    rgb[2] += split_channels[2].at<uint8_t>(rn, cn) * kernel.at<double>(x, y);
                }
            }
            processed_image[0].at<uint8_t>(r, c) = clip((int)(rgb[0]));
            processed_image[1].at<uint8_t>(r, c) = clip((int)(rgb[1]));
            processed_image[2].at<uint8_t>(r, c) = clip((int)(rgb[2]));
        }
    }

    //Объединмяем каналы
    merge(processed_image, 3, output);
    return output;
}

Mat sharpen_img_omp(Mat& input) {
    Mat output = input.clone();
    Mat image = input;
    int size = 3;
    Mat processed_image[3] = { Mat::zeros(Size(image.cols + size * 2, image.rows + size * 2), CV_8UC1),
                              Mat::zeros(Size(image.cols + size * 2, image.rows + size * 2), CV_8UC1),
                              Mat::zeros(Size(image.cols + size * 2, image.rows + size * 2), CV_8UC1) };
    Mat extended_image;

    //Рамка
    copyMakeBorder(image, extended_image, size, size, size, size, BORDER_REFLECT);

    int offset = 1;

    // Сплит каналов
    Mat split_channels[3];
    split(extended_image, split_channels);

    // параллельная часть
    #pragma omp parallel for shared(extended_image)
    for (int r = offset; r < extended_image.rows - offset; r++) {
        for (int c = offset; c < extended_image.cols - offset; c++) {
            double rgb[3] = { 0, 0, 0 };
            for (int x = 0; x < kernel.rows; x++) {
                for (int y = 0; y < kernel.cols; y++) {
                    int rn = r + x - offset;
                    int cn = c + y - offset;
                    rgb[0] += split_channels[0].at<uint8_t>(rn, cn) * kernel.at<double>(x, y);
                    rgb[1] += split_channels[1].at<uint8_t>(rn, cn) * kernel.at<double>(x, y);
                    rgb[2] += split_channels[2].at<uint8_t>(rn, cn) * kernel.at<double>(x, y);
                }
            }
            processed_image[0].at<uint8_t>(r, c) = clip((int)(rgb[0]));
            processed_image[1].at<uint8_t>(r, c) = clip((int)(rgb[1]));
            processed_image[2].at<uint8_t>(r, c) = clip((int)(rgb[2]));
        }
        //cout << omp_get_thread_num() << "\n";
    }

    // Объединяем каналы
    merge(processed_image, 3, output);
    // Сокращаем до начальных если размеры изменились
    return output(Rect(size, size, image.cols, image.rows));
}

int main() {
    string fileName = "input2160p.jpg";
    ofstream fileOutput = ofstream("output.txt");

    omp_set_num_threads(4);
    Mat image = imread(fileName);

    auto time1 = clock();
    Mat processedImage = sharpen_img_omp(image);
    cout << "Time = " << (static_cast<double>(clock()) - time1) << " ms" << endl;

    auto time2 = clock();
    Mat processedImage1 = sharpen_img(image);
    cout << "Time = " << (static_cast<double>(clock()) - time2) << " ms" << endl;

    imwrite(fileName + "_sharp.jpg", processedImage);
    imwrite(fileName + "_sharp1.jpg", processedImage1);

    return 0;
}
