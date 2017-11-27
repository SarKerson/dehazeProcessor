#include "../include/Atmosphere.h"
using namespace cv;
Atmosphere::Atmosphere() {}
Atmosphere::Atmosphere(cv::Vec3f atmosphere)
{
	this->atmosphere = atmosphere;
}

void Atmosphere::calAtmosphere(cv::Mat & darkchannel, cv::Mat & srcImg, cv::Vec3f & output, double rate = 0.001)
{
	if (srcImg.type() != CV_8UC3)
		srcImg.convertTo(srcImg, CV_8UC3, 1.0 / 255, 0);
	if (darkchannel.type() != CV_8UC1)
		darkchannel.convertTo(darkchannel, CV_8UC1, 1.0 / 255, 0);

	cv::Mat matVector = darkchannel.reshape(1, 1);  					 //(1, 116749)
	// cout << matVector.rows << ", " << matVector.cols << endl;
	cv::Mat_<int> sortIndex;
	cv::sortIdx(darkchannel, 
				sortIndex, 
				cv::SORT_EVERY_ROW | 
				cv::SORT_DESCENDING);

	long n_pixel = srcImg.rows * srcImg.cols;
	long n_search = n_pixel * rate;
	Mat srcImgVector = srcImg.reshape(1, n_pixel).clone();  //1channel, 3rows(b, g, r)
												   			//(n_pixel, 3)

	Mat accumulator = Mat::zeros(1, 3, CV_8UC1); //[(8c)0, (8c)0,  (8c)0]
	for(int i = 0; i < n_search; ++i) {
		int index = sortIndex.at<int>(i);
		Mat temp = srcImgVector.rowRange(index, index + 1);   //row:index
		accumulator += temp;
	}
	float t = 255.0 / (float)n_search;

	output[0] = accumulator.at<uchar>(0, 0) * t;
	output[1] = accumulator.at<uchar>(0, 1) * t;
	output[2] = accumulator.at<uchar>(0, 2) * t;
	
	for (int i = 0; i < 3; ++i) {
		if (output[i] == 0) output[i] = 1;
	}

}

void Atmosphere::getAtmosphere(cv::Mat & output)
{
	output = this->atmosphere;
}

void getAtmosphere(cv::Mat & darkchannel, 
					cv::Mat & srcImg, 
					cv::Vec3f & atmosphere)
{
	Atmosphere atms;
	atms.calAtmosphere(darkchannel, srcImg, atmosphere);
}