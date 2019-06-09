#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


// ELEC 474 Take Home Exam
// Andrew Litt

int main()
{
	// Import images

	// Change String 'type' to piano/motorcycle/umbrella for testing on each set.
	// Comment/Uncomment and change proper system parameters, notably the camera parameters and the scaleDownFactor

	String type = "piano";
	String url0 = type+"im0.png";
	String url1 = type+"im1.png";

	Mat img0, img1, gray0, gray1;

	//Read images and convert to greyscale
	img0 = cv::imread(url0, IMREAD_COLOR);
	img1 = cv::imread(url1, IMREAD_COLOR);
	cvtColor(img0,gray0,CV_BGR2GRAY);
	cvtColor(img1,gray1,CV_BGR2GRAY);

	//Scale images & adjust camera parameters
	float scaleDownFactor = 4;
	Size s = Size(round(gray1.cols/scaleDownFactor),round(gray0.rows/scaleDownFactor));
	resize(gray0,gray0,s);
	resize(gray1,gray1,s);

	// Print images
	imshow("img1",gray0);
	imshow("img2",gray1);
	cout << "Press any key to generate displacement map" << endl;
	waitKey();

	float minSAD;

	// BLOCK MATCHING PARAMETERS
	int S = 7;					// S = Pixel Search Radius. Will match best pixel result based in comparisons of SxS blocks
	int c = ceil(S/2)-1;		// c = Center pixel of the block. Used for proper indexing
	int d = 0;					// d = pixel displacement of the best match for current pixel
	int Dmax = 100;				// Dmax = maximum allowed pixel displacement

	Mat diff = Mat::zeros(S,S,CV_8UC1);								// For storing the absolute difference between the current and reference block
	Mat disp = Mat::zeros(gray0.rows,gray0.cols,CV_32FC1);  	// For storing the displacement map values
	Mat minSADs = Mat::zeros(gray0.rows,gray0.cols,CV_32FC1);	// For recording the minimum SAD of each pixel, to classify 'no-matches' later

	for(int i = c; i < gray0.rows-c-2; i++){ //For loops to run through each pixel in image 1
		cout << "Solving Row  = " << i << "/" << gray0.rows-c <<endl;
		for(int j = c; j < gray0.cols-c-2; j++){

			minSAD = 999;
			Mat currentBlock(gray0,Rect(j-c,i-c,S,S)); // Fetch SxS block surrounding current pixel

			for(int k = j-Dmax; k < j+Dmax; k++){ // For loop runs through each pixel from -Dmax to +Dmax away in the row of the current pixel in image 2

				if(k < c){
					// pulls k to be within the image (don't want to query a negative pixel)
					k = c;
				}else if(k >= gray1.cols-c-2){
					// breaks for loop if k is outside image (don't want to query out of bounds)
					k = j + Dmax;
					break;
				}

				Mat referenceBlock(gray1,Rect(k-c,i-c,S,S)); // Fetch SxS block in image2 to compare against currentBlock

				// Calculate best match from using minimum Sum-of-Absolute-Differences (SAD)
				// Find absolute difference for each pixel and summate total error in SAD float
				float SAD = 0;
				absdiff(currentBlock,referenceBlock,diff);

				for(int n = 0; n < diff.rows; n++){
					for(int m = 0; m < diff.cols; m++){
						SAD += diff.at<uchar>(n,m);
					}
				}

				// If the SAD is the lowest recorded, record the new minSAD pixel displacement in d
				if(SAD < minSAD){
					minSAD = SAD;
					d = abs(k-j);
				}
			}
			disp.at<float>(i,j) = d;
			minSADs.at<float>(i,j) = minSAD;
		}
	}



	// Extract relevant parameters

	// PIANO PARAMETERS
	float  f       = 2826.171/scaleDownFactor;
	float cx 	   = 1292.2/scaleDownFactor;
	float cy 	   = 965.806/scaleDownFactor;
	float baseline = 178.089/scaleDownFactor;
	float doffs    = 123.77/scaleDownFactor;

	// MOTORCYCLE PARAMETERS
//	float  f       = 3979.911/scaleDownFactor;
//	float cx 	   = 1244.772/scaleDownFactor;
//	float cy 	   = 1019.507/scaleDownFactor;
//	float baseline = 193.001/scaleDownFactor;
//	float doffs    = 124.343/scaleDownFactor;

	// UMBRELLA PARAMETERS
//	float  f       = 5806.559/scaleDownFactor;
//	float cx 	   = 1429.219/scaleDownFactor;
//	float cy 	   = 993.403/scaleDownFactor;
//	float baseline = 174.019/scaleDownFactor;
//	float doffs    = 114.291/scaleDownFactor;


	ofstream outputFile;
	outputFile.open("output.txt");
	for(int y = 0; y < disp.rows; y++){
		for(int x = 0; x < disp.cols; x++){
			float Z = (baseline*f)/(disp.at<float>(y,x) + doffs);
			float X = (x - cx)*Z/f;
			float Y = (y - cy)*Z/f;
			outputFile << X << "," << Y << "," << Z << endl;
		}
	}
	outputFile.close();

	normalize(disp,disp,0,255,NORM_MINMAX);
	disp.convertTo(disp, CV_8UC1);
	equalizeHist(disp,disp);

	normalize(minSADs,minSADs,0,255,NORM_MINMAX);
	minSADs.convertTo(minSADs, CV_8UC1);

	imshow("disp",disp);
	imshow("minSADs",minSADs);
	waitKey();
	return 0;
}
