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
	// this->t.create(img_scale.size(), CV_32FC1);
	std::vector<float> max_dist(NUM_SPH);
    float ** dist = new float*[img_scale.rows];
    int ** img_index = new int*[img_scale.rows];
    for (int i = 0; i < img_scale.rows; ++i) {
        dist[i] = new float[img_scale.cols];
        img_index[i] = new int[img_scale.cols];
    }
    cv::Mat T(img_scale.size(), CV_32FC1), img_refine(img_scale.size(), CV_32FC3);/*haze_img - A;*/
    for (int i = 0; i < img_scale.rows; ++i) {
        for (int j = 0; j < img_scale.cols; ++j) {
            cv::Vec3f & pixel = img_scale.at<cv::Vec3f>(i, j);
            cv::Vec3f refine = pixel - A, nrefine;
            img_refine.at<cv::Vec3f>(i, j) = refine;
            cv::normalize(refine, nrefine);    //normalization, point to search

            float r = sqrt(nrefine[0] * nrefine[0] +
                            nrefine[1] * nrefine[1] +
                            nrefine[2] * nrefine[2]);

            float * point = new float[2];
            point[0] = acos(nrefine[2] / r);
            point[1] = atan(nrefine[1] / nrefine[0]);
            cvflann::Matrix<float> query(point, 1, 2);

            cvflann::Matrix<int> vecIndex(new int, 1, 1);
            cvflann::Matrix<float> vecDist(new float, 1, 1);
            // kdtree.knnSearch(Mat(point).reshape(1), vecIndex, vecDist, 1);
            kdtree.knnSearch(query, vecIndex, vecDist, 1, cvflann::SearchParams(1));
            img_index[i][j] = vecIndex[0][0];

            dist[i][j] = sqrt(refine[0] * refine[0] + refine[1] * refine[1] + refine[2] * refine[2]);
            if (dist[i][j] > max_dist[vecIndex[0][0]])
                max_dist[vecIndex[0][0]] = dist[i][j];
        }
    }
    for (int i = 0; i < img_scale.rows; ++i) {
        for (int j = 0; j < img_scale.cols; ++j) {
            T.at<float>(i, j) = dist[i][j] / max_dist[img_index[i][j]];
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
