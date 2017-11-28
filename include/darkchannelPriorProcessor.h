#pragma once
#include "dehazeProcessor.h"


/**
 * darkchannelPriorProcessor dp(file);
 * darkchannelPriorProcessor dp(file);
 */
class darkchannelPriorProcessor: public dehazeProcessor
{
public:
	darkchannelPriorProcessor(const cv::Mat& srcImg)
		:dehazeProcessor(srcImg) {}
	darkchannelPriorProcessor(const cv::String & filename)
		:dehazeProcessor(filename) {}
	// darkchannelPriorProcessor(const cv::Mat & srcImg, const cv::String & filename)
	// 	:dehazeProcessor(srcImg, filename) {}
	darkchannelPriorProcessor() {}

public:

	void process();
	void hazeFree();

private:
	cv::Mat darkchannel;
	cv::Mat transmission;
	cv::Vec3f atmosphere;
};

void deHazeByDarkChannelPrior(cv::Mat & input, cv::Mat & output);