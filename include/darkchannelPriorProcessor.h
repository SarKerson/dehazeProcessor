#pragma once
#include "dehazeProcessor.h"

class darkchannelPriorProcessor: public dehazeProcessor
{
public:
	darkchannelPriorProcessor(const cv::Mat& srcImg)
		:dehazeProcessor(srcImg) {}
	darkchannelPriorProcessor(const cv::String & filename)
		:dehazeProcessor(filename) {}
	darkchannelPriorProcessor(const cv::Mat & srcImg, const cv::String & filename)
		:dehazeProcessor(srcImg, filename) {}
	darkchannelPriorProcessor() {}

public:
	void getDarkchannel(cv::Mat & output);
	void getTransmission(cv::Mat & output);

private:
	cv::Mat darkchannel;
	cv::Mat transmission;
	cv::Vec3f atmosphere;
};