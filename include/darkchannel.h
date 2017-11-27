#pragma once

#include <opencv2/opencv.hpp>

class darkchannel
{
private:
	cv::Mat d;
public:
	void getDarkchannel(cv::Mat & output);
	void calDarkChannel(cv::Mat & srcImg, int kernel);
};

void getDarkChannel(cv::Mat & srcImg, cv::Mat & output, int kernel);