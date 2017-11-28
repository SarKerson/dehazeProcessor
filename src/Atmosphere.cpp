#include "../include/Atmosphere.h"
using namespace cv;
Atmosphere::Atmosphere() {}
Atmosphere::Atmosphere(cv::Vec3f atmosphere)
{
	this->atmosphere = atmosphere;
}

void Atmosphere::calAtmosphere(cv::Mat & darkchannel, cv::Mat & srcImg, cv::Vec3f & output, double rate = 0.001)
{
	if (srcImg.type() != CV_32FC3)
		srcImg.convertTo(srcImg, CV_32FC3, 1.0 / 255, 0);
	if (darkchannel.type() != CV_8UC1)
		darkchannel.convertTo(darkchannel, CV_8UC1, 255, 0);

	cv::Mat matVector = darkchannel.reshape(1, 1);  					 //(1, 116749)
	// cout << matVector.rows << ", " << matVector.cols << endl;
	cv::Mat_<int> sortIndex;
	cv::sortIdx(matVector, 
				sortIndex, 
				cv::SORT_EVERY_ROW | 
				cv::SORT_DESCENDING);

	long n_pixel = srcImg.rows * srcImg.cols;
	long n_search = n_pixel * rate;
	// Mat srcImgVector = srcImg.reshape(1, n_pixel).clone();  //1channel, 3rows(b, g, r)
												   			//(n_pixel, 3)

	Vec3f accumulator(0.0, 0.0, 0.0); //[(8c)0, (8c)0,  (8c)0]
	for(int i = 0; i < n_search; ++i) {
		int index = sortIndex.at<int>(i);
		// Mat temp = srcImgVector.rowRange(index, index + 1);   //row:index
		// accumulator += temp;

		Vec3f vec = srcImg.at<Vec3f>(index / srcImg.cols, index % srcImg.cols);
		accumulator += vec;
		std::cout << cv::Mat(accumulator) << "\n";
	}

	// output[0] = accumulator[0] / n_search;
	// output[1] = accumulator[1] / n_search;
	// output[2] = (float)accumulator.at<uchar>(0, 2) / n_search;
	output = accumulator / (float)n_search;
	
	if (output[0] > 0.8) output[0] *= 0.7;
	if (output[1] > 0.8) output[1] *= 0.7;
	if (output[2] > 0.8) output[2] *= 0.7;

	if (output[0] < 0.0001) output[0] = 0.005;
	if (output[1] < 0.0001) output[1] = 0.005;
	if (output[2] < 0.0001) output[2] = 0.005;

	if (srcImg.type() != CV_8UC3)
		srcImg.convertTo(srcImg, CV_8UC3, 255, 0);

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