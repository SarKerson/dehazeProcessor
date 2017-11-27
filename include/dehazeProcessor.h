#pragma once
#include <opencv2/opencv.hpp>
class dehazeProcessor
{
public:
	void setInput(const cv::Mat& input);

	void setFileName(cv::String filename);

	void getOutput(cv::Mat & output);

	dehazeProcessor(const cv::Mat& srcImg);

	dehazeProcessor(const cv::String & filename);

	dehazeProcessor(const cv::Mat & srcImg, const cv::String & filename);

	dehazeProcessor() {}

public:
	virtual void process() = 0;

private:
	cv::Mat srcImg;
	cv::Mat dstImg;
	cv::String filename;
};

void hazeFree(cv::Mat & srcImg, cv::Mat & output, int type);