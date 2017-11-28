#pragma once

#include <opencv2/opencv.hpp>
using namespace cv;

void auto_tune_single(Mat & srcImg, Mat & dstImg, double percent = 0.001)
{
	Mat matVector = srcImg.reshape(1, 1);	//(1, 116749), no change to srcImg
	assert(matVector.cols == srcImg.rows * srcImg.cols);
		// cout << matVector.rows << ", " << matVector.cols << endl;
	cv::Mat_<int> sortIndex;
	sortIdx(matVector, 
			sortIndex, 
			CV_SORT_EVERY_ROW | cv::SORT_ASCENDING);
	float_t min = matVector.at<float_t>(sortIndex.at<int>(int(percent * matVector.cols)));
	float_t max = matVector.at<float_t>(sortIndex.at<int>(int((1.0 - percent) * matVector.cols)));
	
	for (int i = 0; i < matVector.cols; ++i) {
		float_t & pix = matVector.at<float_t>(i);
		if (pix < min) {
			pix = min;
		}
		else if (pix > max) {
			pix = max;
		}
	}

	float_t diff = max - min;
	srcImg -= min;
	srcImg /= diff;
	dstImg = srcImg;
	// cout << "1:\t" << (int)matVector.at<uchar>(0, 1) << "\t100: " << (int)matVector.at<uchar>(0, 100) << endl;

}

void auto_tune(Mat & srcImg, Mat & dstImg, double percent = 0.001)
{
// #if DEBUG
// 	clock_t start = clock();
// #endif
	srcImg.convertTo(srcImg, CV_32FC3, 1.0 / 255, 0);
	std::vector<Mat> vrgb(3);
	split(srcImg, vrgb);
	auto_tune_single(vrgb[0], vrgb[0]);
	auto_tune_single(vrgb[1], vrgb[1]);
	auto_tune_single(vrgb[2], vrgb[2]);
	merge(vrgb, srcImg);
	srcImg.convertTo(dstImg, CV_8UC3, 255, 0);
// #if DEBUG
// 	cout << "time of auto_tune: " << (clock() - start) / 1000000.0 << " s" << "\n";
// #endif
}