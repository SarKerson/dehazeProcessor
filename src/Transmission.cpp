#include "../include/Transmission.h"

void Transmission::getTransmission(cv::Mat & output)
{
	output = this->t;
}
void Transmission::calTransmission(cv::Mat & darkchannel, float_t w = 0.95)
{
	cv::Mat temp;
	if (darkchannel.type() != CV_8UC1) 
		darkchannel.convertTo(temp, CV_8UC1, 1.0 / 255, 0);
	this->t = 255 - w * darkchannel;
}


void getTransmission(cv::Mat & darkchannel, cv::Mat & output)
{
	Transmission t;
	t.calTransmission(darkchannel);
	t.getTransmission(output);
}