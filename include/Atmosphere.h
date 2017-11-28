#pragma once

#include <opencv2/opencv.hpp>
class Atmosphere
{
private:
	cv::Vec3f atmosphere;
public:
	Atmosphere();
	Atmosphere(cv::Vec3f atmosphere);
	void calAtmosphere(cv::Mat & darkchannel, cv::Mat & srcImg, cv::Vec3f & output, double rate);
	void getAtmosphere(cv::Mat & output);
};

void getAtmosphere(cv::Mat & darkchannel, cv::Mat & srcImg, cv::Vec3f & atmosphere);

