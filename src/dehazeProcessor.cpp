#include "../include/dehazeProcessor.h"

dehazeProcessor::dehazeProcessor(const cv::Mat& srcImg)
{
	this->setInput(srcImg);
}

dehazeProcessor::dehazeProcessor(const cv::String & filename)
{
	this->setInput(filename);
}

// dehazeProcessor::dehazeProcessor(const cv::Mat & srcImg, const cv::String & filename)
// {
// 	this->filename = filename;
// 	this->srcImg = srcImg.clone();
// }

void dehazeProcessor::setInput(const cv::Mat& input)
{
	this->srcImg = input.clone();
}

void dehazeProcessor::setInput(const cv::String & filename)
{
	try {
		this->srcImg = cv::imread(filename);
	}
	catch (cv::Exception e) {
		std::cout << e.what() << std::endl;
		exit(1);
	}
}

// void dehazeProcessor::setFileName(cv::String filename)
// {
// 	this->filename = filename;
// }

void dehazeProcessor::getOutput(cv::Mat & output)
{
	output = this->dstImg;
}