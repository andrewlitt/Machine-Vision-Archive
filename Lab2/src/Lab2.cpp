/* ELEC 474: Machine Vision
 * Laboratory 3
 * Andrew Litt & Zack Harley
*/

// Standard C++
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <string>
using namespace std;

// OpenCV Imports
#include <opencv/cv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> // OpenCV Core Functionality
#include <opencv2/highgui/highgui.hpp> // High-Level Graphical User Interface

using namespace cv;

int main(int argc, char **argv)
{
	VideoCapture belt_bg("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\belt2_bg.wmv");
	VideoCapture belt_fg("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\belt2_fg.wmv");

	cv::Mat frame, greyframe;
	belt_bg >> frame;

	int rows = frame.rows;
	int cols = frame.cols;
	float arr[rows][cols] = {};
	int frameCount = 1;

	cv::Mat M(frame.size(),CV_8U, Scalar());
	cv::Mat S(frame.size(),CV_8U, Scalar());
	cv::Mat I(frame.size(),CV_8U, Scalar());
	cv::Mat C(frame.size(),CV_8U, Scalar());

	while(1){
		belt_bg >> frame;
		if(!frame.data) break;
		cvtColor(frame, greyframe, CV_RGB2GRAY);
		for(int i = 0; i < rows; i++){
				for(int j = 0; j < cols; j++){
					arr[i][j] += greyframe.at<uchar>(i,j);
				}
			}
		frameCount++;
	}
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			arr[i][j] = arr[i][j]/frameCount;
			M.at<uchar>(i,j) = round(arr[i][j]);
		}
	}
	belt_bg.set(CV_CAP_PROP_POS_AVI_RATIO,0);
	belt_bg >> frame;
	while(1){
			belt_bg >> frame;
			if(!frame.data) break;
			cvtColor(frame, greyframe, CV_RGB2GRAY);
			for(int i = 0; i < rows; i++){
					for(int j = 0; j < cols; j++){
						arr[i][j] += (greyframe.at<uchar>(i,j) - M.at<uchar>(i,j))*(greyframe.at<uchar>(i,j) - M.at<uchar>(i,j));
					}
				}
	}
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			S.at<uchar>(i,j) = round(sqrt(arr[i][j]/(frameCount-1)));
		}
	}
	//cv::namedWindow("Standard Deviation", WINDOW_AUTOSIZE);
	//cv::namedWindow("Mean", WINDOW_AUTOSIZE);
	//cv::imshow("Standard Deviation", S);
	//cv::imshow("Mean", M);

	vector<vector<Point> > nutC1, nutC2, nutC3;
	vector<vector<Point> > pegC1, pegC2, pegC3;
	vector<vector<Point> > pipeC4;
	vector<vector<Point> > prongC1, prongC3, prongC4, prongC5;
	vector<vector<Point> > qC1, qC2, qC3;
	vector<vector<Point> > washerC1, washerC3;
	Mat nut1 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\nut1.bmp", 0);
	Mat nut2 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\nut2.bmp", 0);
	Mat nut3 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\nut3.bmp", 0);
	Mat peg1 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\peg1.bmp", 0);
	Mat peg2 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\peg2.bmp", 0);
	Mat peg3 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\peg3.bmp", 0);
	Mat pipe4 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\pipe4.bmp", 0);
	Mat prong1 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\prong1.bmp", 0);
	Mat prong3 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\prong3.bmp", 0);
	Mat prong4 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\prong4.bmp", 0);
	Mat prong5 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\prong5.bmp", 0);
	Mat q1 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\q1.bmp", 0);
	Mat q2 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\q2.bmp", 0);
	Mat q3 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\q3.bmp", 0);
	Mat washer1 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\washer1.bmp", 0);
	Mat washer3 = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 2\\washer3.bmp", 0);
	cv::findContours(nut1,nutC1, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(nut2,nutC2, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(nut3,nutC3, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(peg1,pegC1, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(peg2,pegC2, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(peg3,pegC3, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(pipe4,pipeC4, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(prong1,prongC1, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(prong3,prongC3, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(prong4,prongC4, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(prong5,prongC5, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(q1,qC1, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(q2,qC2, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(q3,qC3, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(washer1,washerC1, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
	cv::findContours(washer3,washerC3, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));

	string names[] = {"Nut","Peg","Pipe","Prong","Q","Washer"};
	cout << "Contour Areas" << endl;
	double nutAreaAvg = (contourArea(nutC1[0]) + contourArea(nutC2[0]) + contourArea(nutC3[0]))/3;
	double pegAreaAvg = (contourArea(pegC1[0]) + contourArea(pegC2[0]) + contourArea(pegC3[0]))/3;
	double pipeAreaAvg = contourArea(pipeC4[0]);
	double prongAreaAvg = (contourArea(prongC1[0]) + contourArea(prongC3[0]) + contourArea(prongC4[0]) + contourArea(prongC4[0]))/4;
	double qAreaAvg = (contourArea(qC1[0]) + contourArea(qC2[0]) + contourArea(qC3[0]))/3;
	double washerAreaAvg = (contourArea(washerC1[0]) + contourArea(washerC3[0]))/2;

	double areaAvgs[]= {nutAreaAvg,pegAreaAvg,pipeAreaAvg,prongAreaAvg,qAreaAvg,washerAreaAvg};

	cout << nutAreaAvg <<endl;
	cout << pegAreaAvg <<endl;
	cout << pipeAreaAvg <<endl;
	cout << prongAreaAvg <<endl;
	cout << qAreaAvg <<endl;
	cout << washerAreaAvg <<endl;

	double nutContAvg = (arcLength(nutC1[0],true) + arcLength(nutC2[0],true) + arcLength(nutC3[0],true))/3;
	double pegContAvg = (arcLength(pegC1[0],true) + arcLength(pegC2[0],true) + arcLength(pegC3[0],true))/3;
	double pipeContAvg = arcLength(pipeC4[0],true);
	double prongContAvg = (arcLength(prongC1[0],true) + arcLength(prongC3[0],true) + arcLength(prongC4[0],true) + arcLength(prongC4[0],true))/4;
	double qContAvg = (arcLength(qC1[0],true) + arcLength(qC2[0],true) + arcLength(qC3[0],true))/3;
	double washerContAvg = (arcLength(washerC1[0],true) + arcLength(washerC3[0],true))/2;

	double lenAvgs[]= {nutContAvg,pegContAvg,pipeContAvg,prongContAvg,qContAvg,washerContAvg};

	//double ratios[] = {nutAreaAvg/nutContAvg,pegAreaAvg/pipeContAvg,pipeAreaAvg/pipeContAvg,prongAreaAvg/prongContAvg,qAreaAvg/qContAvg,washerAreaAvg/washerContAvg};

	cout << "Contour Perimeters" << endl;
	cout << nutContAvg <<endl;
	cout << pegContAvg <<endl;
	cout << pipeContAvg <<endl;
	cout << prongContAvg <<endl;
	cout << qContAvg <<endl;
	cout << washerContAvg <<endl;

	int thres = 10;

	cv::namedWindow("Window", WINDOW_AUTOSIZE);
	cv::namedWindow("Window2", WINDOW_AUTOSIZE);

	vector<vector<Point> > contours;
	double length,area = 0;
	int count = 0;
	int prevIndex = 0;
	belt_fg.set(CV_CAP_PROP_POS_MSEC,0);
	belt_fg >> frame;
	while(1){

		belt_fg >> I;
		C = cv::Mat::zeros(I.size(),CV_8U);
		if(!I.data) break;
		cvtColor(I, I, CV_RGB2GRAY);
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				if(abs(I.at<uchar>(i,j)-M.at<uchar>(i,j)) > thres*S.at<uchar>(i,j)){
					I.at<uchar>(i,j) = 255;
				}
				else{
					I.at<uchar>(i,j) = 0;
				}
			}
		}
		cv::findContours(I,contours, CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
		Scalar color( 255, 255, 255 );

		if(contours.size() > 0){
			length = cv::arcLength(contours[0],true);
			area = cv::contourArea(contours[0]);

			int minIndexArea = -1;
			int minIndexLen = -1;
			int minIndex = -1;
			double minAreaDiff = 1000;
			double minLenDiff = 1000;
			double areaDiff, lenDiff;

			for(int k = 0; k < 6; k++){
				areaDiff = abs(area-areaAvgs[k]);
				lenDiff = abs(length-lenAvgs[k]);

				if(areaDiff < minAreaDiff){
					minAreaDiff = areaDiff;
					minIndexArea = k;
				}
				if(lenDiff < minLenDiff){
					minLenDiff = lenDiff;
					minIndexLen = k;
				}
			}

			//cout << length << " " << area<< endl;

			if((minIndexLen != -1) && (minIndexLen == minIndexArea)){
				minIndex = minIndexLen;
				if(prevIndex == minIndex){
					count = count + 1;
				}
				if(count >=4){
					count = 0;
					cout << names[minIndexArea] << endl;
				}
			}
			else if(length > 8 && length < 14 && area > 3 && area < 10 ){
				minIndex = 2;
				if(prevIndex == minIndex){
					count = count + 1;
				}
				if(count >=4){
					count = 0;
					cout << "Pipe" << endl;
				}
			}
			if(prevIndex !=minIndex){
				count = 0;
			}
			prevIndex = minIndex;
		}
		cv::drawContours(C, contours, 0, color, 1);
		cv::imshow("Window",I);
		cv::imshow("Window2",C);
		if(waitKey(30) >= 0) break;
	}
	// Wait for a keystroke in the window
	// NOTE! If you don't add this the window will close immediately!
	cv::waitKey();
	// Terminate the program
	return 0;
}
