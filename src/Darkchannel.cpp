#include "../include/Darkchannel.h"
void darkchannel::getDarkchannel(cv::Mat & output)
{
	output = this->d;
}
void darkchannel::calDarkChannel(cv::Mat & srcimg, int kernel = 15)
{
	using namespace cv;

	Mat gray(srcimg.size(), CV_8UC1);

	int wins = 2 * kernel + 1, k = kernel;
	int h_min = srcimg.rows / wins * wins + wins;
	int w_min = srcimg.cols / wins * wins + wins;

	this->d.create(srcimg.size(), CV_8UC1);

	Mat img_min(Size(w_min, h_min), CV_8UC1, Scalar::all(0)),
		img_a(Size(w_min, h_min), CV_8UC1, Scalar::all(255)),
		img_b(Size(w_min, h_min), CV_8UC1, Scalar::all(255)),
		img_c(Size(w_min, h_min), CV_8UC1, Scalar::all(255)),
		img_d(Size(w_min, h_min), CV_8UC1, Scalar::all(255));

	for (int i = 0 ; i < srcimg.rows; ++i) {
		Vec3b *src_ptr = srcimg.ptr<Vec3b>(i);
		for (int j = 0; j < srcimg.cols; ++j) {
			gray.at<uchar>(i, j) = min(min(src_ptr[j][0], src_ptr[j][1]), src_ptr[j][2]);
		}
	}

	for(int i = wins - 1; i < h_min; i = i + wins)//img_a
	{
		for(int j = wins - 1; j < w_min; j = j + wins)
		{
			for(int a = i; a >= i - wins + 1; a-- )
			{
				for(int b = j; b >= j - wins + 1; b--)
				{
					if(a >= srcimg.rows || b >= srcimg.cols)
						img_a.at<uchar>(a, b) = 255;
					else if(a == i && b==j)
						img_a.at<uchar>(a, b) = gray.at<uchar>(a, b);
					else if(a == i)
						img_a.at<uchar>(a, b) = min(gray.at<uchar>(a, b), img_a.at<uchar>(a, b + 1));
					else if(b == j)
						img_a.at<uchar>(a, b) = min(gray.at<uchar>(a, b), img_a.at<uchar>(a + 1, b));
					else
						img_a.at<uchar>(a, b) = min(gray.at<uchar>(a, b),min(img_a.at<uchar>(a, b + 1), img_a.at<uchar>(a + 1, b)));
				}
			}
		}
	}

	for(int i = wins - 1; i < h_min; i = i + wins)//img_b
	{
		for(int j = wins - 1; j < w_min; j = j + wins)
		{
			for(int a = i; a >= i - wins + 1; a-- )
			{
				for(int b = j; b >= j - wins + 1; b--)
				{
					if(a >= srcimg.rows || b >= srcimg.cols)
						img_b.at<uchar>(a, b) = 255;
					else if(a == i && b==j)
						img_b.at<uchar>(a, b) = gray.at<uchar>(a, b);
					else if(a == i)
						img_b.at<uchar>(a, b) = min(gray.at<uchar>(a, b), img_b.at<uchar>(a, b - 1));
					else if(b == j)
						img_b.at<uchar>(a, b) = min(gray.at<uchar>(a, b), img_b.at<uchar>(a + 1, b));
					else
						img_b.at<uchar>(a, b) = min(gray.at<uchar>(a, b),min(img_b.at<uchar>(a, b - 1), img_b.at<uchar>(a + 1, b)));
				}
			}
		}
	}


	for(int i = wins - 1; i < h_min; i = i + wins)//img_b
	{
		for(int j = wins - 1; j < w_min; j = j + wins)
		{
			for(int a = i; a >= i - wins + 1; a-- )
			{
				for(int b = j; b >= j - wins + 1; b--)
				{
					if(a >= srcimg.rows || b >= srcimg.cols)
						img_c.at<uchar>(a, b) = 255;
					else if(a == i && b==j)
						img_c.at<uchar>(a, b) = gray.at<uchar>(a, b);
					else if(a == i)
						img_c.at<uchar>(a, b) = min(gray.at<uchar>(a, b), img_c.at<uchar>(a, b + 1));
					else if(b == j)
						img_c.at<uchar>(a, b) = min(gray.at<uchar>(a, b), img_c.at<uchar>(a - 1, b));
					else
						img_c.at<uchar>(a, b) = min(gray.at<uchar>(a, b),min(img_c.at<uchar>(a, b + 1), img_b.at<uchar>(a - 1, b)));
				}
			}
		}
	}

	for(int i = wins - 1; i < h_min; i = i + wins)//img_b
	{
		for(int j = wins - 1; j < w_min; j = j + wins)
		{
			for(int a = i; a >= i - wins + 1; a-- )
			{
				for(int b = j; b >= j - wins + 1; b--)
				{
					if(a >= srcimg.rows || b >= srcimg.cols)
						img_d.at<uchar>(a, b) = 255;
					else if(a == i && b==j)
						img_d.at<uchar>(a, b) = gray.at<uchar>(a, b);
					else if(a == i)
						img_d.at<uchar>(a, b) = min(gray.at<uchar>(a, b), img_c.at<uchar>(a, b - 1));
					else if(b == j)
						img_d.at<uchar>(a, b) = min(gray.at<uchar>(a, b), img_c.at<uchar>(a - 1, b));
					else
						img_d.at<uchar>(a, b) = min(gray.at<uchar>(a, b),min(img_c.at<uchar>(a, b - 1), img_b.at<uchar>(a - 1, b)));
				}
			}
		}
	}

	int hh = srcimg.rows - k - 1;
	int ww = srcimg.cols - k;
	for(int x = k; x < hh; x++)
	{
		int y;
        for(y = k; y < ww; y++)
        {
            img_min.at<uchar>(x - k, y - k) = min(min(img_a.at<uchar>(x - k, y - k), 
            											img_b.at<uchar>(x - k, y + k)),
                                           			min(img_c.at<uchar>(x + k, y - k),
                                           				img_d.at<uchar>(x + k, y + k)));
        }
        --y;
        for (int y1 = y - k; y1 <= y + k ; y1++)
        {
            img_min.at<uchar>(x - k, y1) = min(min(img_a.at<uchar>(x - k, y - k), 
        											 img_b.at<uchar>(x - k, y + k)),
                                       			 min(img_c.at<uchar>(x + k, y - k),
                                       			     img_d.at<uchar>(x + k, y + k)));
    	}
	}

	int i = hh;
	int j = ww -1 ;
	for (int y = k ; y <= ww; y++)
	{
		for (int x2 = i-k; x2 <= i+k; x2++)
		{
            img_min.at<uchar>(x2, y - k) = min(min(img_a.at<uchar>(i - k, y - k), 
        											 img_b.at<uchar>(i - k, y + k)),
                                       			 min(img_c.at<uchar>(i + k, y - k),
                                       			     img_d.at<uchar>(i + k, y + k)));
		}
	}

	for (int x = i - k; x <= i + k; x++)
	{
		for (int y = j-k; y <= j + k; y++)
		{
            img_min.at<uchar>(x, y) = min(min(img_a.at<uchar>(i - k, j - k), 
    											img_b.at<uchar>(i - k, j + k)),
                                   			min(img_c.at<uchar>(i + k, j - k),
                                   			    img_d.at<uchar>(i + k, j + k)));
		}
	}

	for(int h = 0; h < srcimg.rows; h++)
	{
		for(int w = 0; w < srcimg.cols; w++)
		{
			this->d.at<uchar>(h, w) = img_min.at<uchar>(h, w);
		}
	}
	
}


void getDarkChannel(cv::Mat & srcImg, cv::Mat & output, int kernel = 15)
{
	darkchannel d;
	d.calDarkChannel(srcImg, kernel);
	d.getDarkchannel(output);
}