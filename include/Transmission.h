#pragma once
#include <opencv2/opencv.hpp>

class Transmission {
private:
	cv::Mat t;
public:
	void getTransmission(cv::Mat & output);
	void calTransmission(cv::Mat & darkchannel, float_t w);
};

void getTransmission(cv::Mat & darkchannel, cv::Mat & output);