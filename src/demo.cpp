#include "../include/darkchannelPriorProcessor.h"
#include "../include/nonLocalDehazeProcessor.h"
#include "../include/Autotune.h"
#include <sstream>
using namespace cv;
using namespace std;

#define SHOW_FRAME 1

void testOnImg(Mat & src)
{	
	Mat dst;
	if (src.rows > 300)
		resize(src, src, Size(src.cols * 300 / src.rows, 300));
	imshow("src", src);
	auto_tune(src, src);
	deHazeByNonLocalMethod(src, dst, "../TR_SPHERE_2500.txt");
	// deHazeByDarkChannelPrior(src, dst);
	imshow("dst", dst);
}

void testOnImg(cv::String filename)
{
	Mat src = imread(filename);
	testOnImg(src);
}

void writeImg(std::string & in, std::string & out)   //no output will display
{
	clock_t start = clock();	
	cout << "read--------->" << in << "\n";
	Mat src = imread(in), dst;
	if (src.rows > 300)
		resize(src, src, Size(src.cols * 300 / src.rows, 300));
	auto_tune(src, src);
	deHazeByDarkChannelPrior(src, dst);
	std::vector<int> compression_params;  
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);  
    compression_params.push_back(9); 
	try {
		imwrite(out, dst, compression_params);
		cout << "write--------->" << out << "\n";
	}
	catch (std::runtime_error & e) {
		std::cout << e.what() << "\n";
	}
	cout << "##-----------------------time:" << (clock() - start) / 1000000.0 << " s\n";
}


void testOnMedia(std::string filename) 
{
	VideoCapture capture(filename);
#if SHOW_FRAME
	clock_t start = clock();
	int fram = 0;
#endif

	while (true) {
		Mat frame;
		capture >> frame;
#if SHOW_FRAME
		++fram;
#endif
		if (frame.empty()) {
			cout << "empty" << endl;
			break;
		}
		assert(frame.type() == CV_8UC3);
		if (frame.rows > 200)
			resize(frame, frame, Size(frame.cols * 200 / frame.rows, 200));
		
		testOnImg(frame);

		int key = waitKey(5);
		if (key == 'q')
			break;
#if SHOW_FRAME
		if (clock() - start >= 1000000) {
			cout << "frame: " << fram << " /s" << "\n";
			fram = 0;
			start = clock();
		}
#endif
	}
}

void writeMedia(std::string in, std::string out)
{
	VideoCapture capture(in);
#if SHOW_FRAME
    clock_t start = clock();
    int fram = 0;
#endif

    VideoWriter writer(out, CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(355, 200));
    Mat dst;
    while (true) {
        Mat frame;
        capture >> frame;
#if SHOW_FRAME
        ++fram;
#endif
        if (frame.empty()) {
            cout << "empty" << endl;
            break;
        }
        assert(frame.type() == CV_8UC3);

        resize(frame, frame, Size(355, 200));
        
        auto_tune(frame, frame);
		deHazeByDarkChannelPrior(frame, dst);

        writer << dst;
        int key = waitKey(5);
        if (key == 'q')
            break;
#if SHOW_FRAME
        if (clock() - start >= 1000000) {
            cout << "frame: " << fram << " /s" << "\n";
            fram = 0;
            start = clock();
        }
#endif
    }
}

int main(int argc, char const *argv[])
{
	const cv::String keys =
    "{help h usage ? |      | print this message   		  }"
    "{type           |0     | '0' for image, '1' for vedio}"
    "{input          |      | folder for input(s)  		  }"
    "{output         |      | folder for output(s) 		  }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("dehazeProcessor By Leiqi Yao");

    if (parser.has("help")) {
    	parser.printMessage();
    	return 0;
    }

    bool type = parser.get<bool>("type");

	if (parser.has("input")) {
		cv::String input = parser.get<cv::String>("input");
		if (type == false) {		// "false" means for image
									// while "true" means for vedio 
			if (parser.has("output")) {
				cv::String output = parser.get<cv::String>("output");
				std::vector<cv::String> files;
				cv::glob(input, files, false);
				for (auto & file : files) {
					cout << file << endl;
					stringstream ss_in, ss_out;
					string in(file, 9), out;
					ss_in << input.c_str() << in;
					ss_out << output.c_str() << in;
					ss_in >> in;
					ss_out >> out;
					writeImg(in, out);
				}
			}
			else {
				testOnImg(input.c_str());
			}
		}
		else {
			if (parser.has("output")) {
				cv::String output = parser.get<cv::String>("output");
				writeMedia(input.c_str(), output.c_str());
			}
			else {
				testOnMedia(input.c_str());
			}
		}
	}

	waitKey(0);
	return 0;
}
