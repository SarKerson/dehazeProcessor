#include "../include/dehazeProcessor.h"

dehazeProcessor::dehazeProcessor(const cv::Mat& srcImg)
{
	this->srcImg = srcImg.clone();
}

dehazeProcessor::dehazeProcessor(const cv::String & filename)
{
	this->filename = filename;
}

dehazeProcessor::dehazeProcessor(const cv::Mat & srcImg, const cv::String & filename)
{
	this->filename = filename;
	this->srcImg = srcImg.clone();
}

void dehazeProcessor::setInput(const cv::Mat& input)
{
	this->srcImg = input.clone();
}

void dehazeProcessor::setFileName(cv::String filename)
{
	this->filename = filename;
}

void dehazeProcessor::getOutput(cv::Mat & output)
{
	output = this->dstImg;
}