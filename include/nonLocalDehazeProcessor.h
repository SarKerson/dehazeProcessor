#pragma once
#include "dehazeProcessor.h"


class nonLocalDehazeProcessor: public dehazeProcessor
{
public:
	nonLocalDehazeProcessor(const cv::Mat& srcImg, const std::string sph_file)
		:dehazeProcessor(srcImg) { init(sph_file); }
	nonLocalDehazeProcessor(const cv::String & filename, const std::string sph_file)
		:dehazeProcessor(filename) { init(sph_file); }
	nonLocalDehazeProcessor(const std::string sph_file) { init(sph_file); }

public:

	void init(std::string sph_file);

	void process();
	void hazeFree();

private:
	cv::Mat darkchannel;
	cv::Mat transmission;
	cv::Vec3f atmosphere;
	cv::flann::Index * kdtree;
	int SPH_NUM;
};

void deHazeByNonLocalMethod(cv::Mat & input, cv::Mat & output, std::string sph_file);