#pragma once
#include <opencv2/opencv.hpp>

class Transmission {
private:
	cv::Mat t;
public:
	void getTransmission(cv::Mat & output);
	void calTransmission(cv::Mat & darkchannel, float_t w);
	void calTransmission(cv::Mat & srcImg, cv::Vec3f & A, cv::flann::Index & kdtree, const int NUM_SPH);
};

void getTransmission(cv::Mat & darkchannel, cv::Mat & output);
void getTransmission(cv::Mat & input, 
					 cv::Mat & output,
					 cv::Vec3f & A, 
					 cv::flann::Index & kdtree,
					 const int NUM_SPH);