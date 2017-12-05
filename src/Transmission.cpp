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

void Transmission::calTransmission(cv::Mat & srcImg, 
					 cv::Vec3f & A, 
					 cvflann::Index< cvflann::L2_Simple<float> > & kdtree,
					 const int NUM_SPH)
{
	cv::Mat img_scale;
	if (srcImg.type() != CV_32FC3)
		srcImg.convertTo(img_scale, CV_32FC3, 1.0 / 255, 0);
	else
		img_scale = srcImg;
	std::vector<float> max_dist(NUM_SPH);
    float ** dist = new float*[img_scale.rows];
    for (int i = 0; i < img_scale.rows; ++i) {
        dist[i] = new float[img_scale.cols];
    }
    float * query_ar = new float[img_scale.rows * img_scale.cols * 2];
    int count = 0;
    cv::Mat T(img_scale.size(), CV_32FC1);
    for (int i = 0; i < img_scale.rows; ++i) {
        for (int j = 0; j < img_scale.cols; ++j) {
            cv::Vec3f & pixel = img_scale.at<cv::Vec3f>(i, j);
            cv::Vec3f refine = pixel - A, nrefine;
            cv::normalize(refine, nrefine);    //normalization, point to search

            float r = sqrt(nrefine[0] * nrefine[0] +
                            nrefine[1] * nrefine[1] +
                            nrefine[2] * nrefine[2]);
            query_ar[count++] = acos(nrefine[2] / r);
            query_ar[count++] = atan(nrefine[1] / nrefine[0]);
            dist[i][j] = sqrt(refine[0] * refine[0] + refine[1] * refine[1] + refine[2] * refine[2]);
        }
    }

    int num_query = img_scale.rows * img_scale.cols;
    cvflann::Matrix<float> query(query_ar, num_query, 2);

    cvflann::Matrix<int> vecIndex(new int[num_query], num_query, 1);
    cvflann::Matrix<float> vecDist(new float[num_query], num_query, 1);
    kdtree.knnSearch(query, vecIndex, vecDist, 1, cvflann::SearchParams(1));
    
    for (int i = 0; i < img_scale.rows; ++i) {
        for (int j = 0; j < img_scale.cols; ++j) {
            int ind = vecIndex[i * img_scale.cols + j][0];
            if (dist[i][j] > max_dist[ind])
                max_dist[ind] = dist[i][j];
        }
    }

    for (int i = 0; i < img_scale.rows; ++i) {
        for (int j = 0; j < img_scale.cols; ++j) {
            int ind = vecIndex[i * img_scale.cols + j][0];
            T.at<float>(i, j) = dist[i][j] / max_dist[ind];
        }
    }

    T.convertTo(this->t, CV_8UC1, 255, 0);
}


void getTransmission(cv::Mat & darkchannel, cv::Mat & output)
{
	Transmission t;
	t.calTransmission(darkchannel);
	t.getTransmission(output);
}

void getTransmission(cv::Mat & input, 
					 cv::Mat & output,
					 cv::Vec3f & A, 
					 cvflann::Index< cvflann::L2_Simple<float> > & kdtree,
					 const int NUM_SPH)
{
	Transmission t;
	t.calTransmission(input, A, kdtree, NUM_SPH);
	t.getTransmission(output);
}
