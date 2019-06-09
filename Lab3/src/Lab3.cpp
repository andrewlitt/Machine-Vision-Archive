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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include "opencv2/bgsegm.hpp"
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>// High-Level Graphical User Interface

using namespace cv;

float getDist(Point p3, float slope, float b) {
	float distance;
	if (slope == 1000000000)
		distance = std::abs(p3.x - b); //b here represents the x=const case.
	else
		distance = std::abs((p3.y - slope * p3.x - b))
				/ sqrt(slope * slope + 1);
	return distance;
}
float getSlope(Point p1, Point p2) {
	float slope;
	if (p1.x == p2.x) {
		slope = 1000000000.0;
	} else {
		slope = ((float) p2.y - (float) p1.y) / ((float) p2.x - (float) p1.x);
	}
	return slope;
}
float getYint(Point p1, float slope) {
	float b;
	if (slope == 1000000000) {
		b = p1.x; //should be inf but to represent case x=const, so b will be this constant
	} else {
		b = p1.y - slope * p1.x;
	}
	return b;
}
Mat addLine(Mat lines_img, float slope, float y_intercept) {
	Point point1, point2;

	if (slope == 1000000000) {
		point1.x = y_intercept;
		point1.y = 0;
		point2.x = y_intercept;
		point2.y = 1000;
	}

	else {
		point1.x = 0;
		point1.y = slope * point1.x + y_intercept;
		point2.x = 1000;
		point2.y = slope * point2.x + y_intercept;
	}

	line(lines_img, point1, point2, Scalar(0, 0, 255), 1, 8);
	return lines_img;
}
void LineRANSAC(Mat &img, Mat &imgColor, int thres){

		GaussianBlur(img,img,Size(3,3), 1, 1);
		Canny(img,img,thres,thres*3,3, true);
		// Identify all valid points post-canny filtering and store in a vector
		vector<cv::Point > imgPoints;
		for (int row = 0; row < img.rows; row++) {
			for (int col = 0; col < img.cols; col++) {
				if (img.at<unsigned char>(row, col) == 255) {
					imgPoints.push_back(Point(col, row)); // (x:col,y:row)
				}
			}
		}

		// Begin RANSAC

		// Parameters
		int iter = 1500;
		int minPoints = 100;
		int showLines = 15;
		int inlierCount;
		float maxDist = 3;
		float maxDist2 = 2;
		float slope, y_int, dist;

		// A matrix of dimensions iter x 3: stores # inliers, slope, y intercept per row
		Mat lines(iter, 3, CV_32F, Scalar(0.0));
		vector<Point > inliers; // Vector of points that are inliers

		cv::Point p1,p2,p3;
		RNG rand;

		for(int i = 0; i < iter; i++){
			inlierCount = 0;
			// Generate two random points

			p1 = imgPoints.at(rand.uniform(0, imgPoints.size()));
			p2 = imgPoints.at(rand.uniform(0, imgPoints.size()));

			float d1, d2;
			for (int j = 0; j < lines.rows; j++) {
				float temp_slope = lines.at<float>(j, 1);
				float temp_intercept = lines.at<float>(j, 2);
				d1 = getDist(p1, temp_slope, temp_intercept);
				d2 = getDist(p2, temp_slope, temp_intercept);
				if ((d1 < maxDist2) && (d2 < maxDist2))
					break;
			}
			if ((d1 < maxDist2) && (d2 < maxDist2))
				continue;

			// Get the slope & y-intercept of each one
			slope = getSlope(p1,p2);
			y_int = getYint(p1,slope);

			// Find the distance of valid points to line and count # of inliersS
			for(int j = 0; j < imgPoints.size(); j++){
				p3 = imgPoints.at(j);
				dist = getDist(p3, slope, y_int);
				if(dist <= maxDist){
					inlierCount++;
					inliers.push_back(p3);
				}
			}

			// If random line is above minimum points, do a better fit & store that information
			Vec4f lineVector; // Stored attributes of improved line fit
			if(inlierCount > minPoints){
				fitLine(inliers,lineVector,CV_DIST_L2,0,0.01,0.01);
				// lineVector = (vx, vy, x0, y0)

				Point2f pt1,pt2;

				// Update the two points to be on our better fit line
				pt1.x = lineVector[2];
				pt1.y = lineVector[3];

				pt2.x = lineVector[2] + 30*lineVector[0];
				pt2.y = lineVector[3] + 30*lineVector[1];

				// Get improved slope & intercept value
				slope = getSlope(pt1,pt2);
				y_int = getYint(pt1,slope);

				// Store information in line array
				lines.at<float>(i,0) = inlierCount;
				lines.at<float>(i,1) = slope;
				lines.at<float>(i,2) = y_int;
			}
			inliers.clear();
		}

		//Sort lines from best->worst

		Mat sortedIndecies(lines.size(), lines.type());
		Mat lines2 = lines.clone();
		sortIdx(lines2,sortedIndecies,CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
		int index;
		for (int n = 1; n <= showLines; n++) {
			index = sortedIndecies.at<int>(n, 0);
			float s = lines.at<float>(index, 1);
			float y = lines.at<float>(index, 2);
			if ((!(s == 0)) && (!(y == 0))) {
				imgColor = addLine(imgColor, s, y);
			}
		}
}

Point getCenter(Point p1, Point p2, Point p3){
	Point center;
	float a,b;
	float d11,d12,d21,d22,denom;

	d11 = p2.x*p2.x + p2.y*p2.y - (p1.x*p1.x + p1.y*p1.y);
	d12 = 2*(p2.y-p1.y);
	d21 = p3.x*p3.x + p3.y*p3.y - (p1.x*p1.x + p1.y*p1.y);
	d22 = 2*(p3.y - p1.y);

	denom = 4*( (p2.x-p1.x)*(p3.y-p1.y) - (p3.x-p1.x)*(p2.y-p1.y) );

	a = (d11*d22-d12*d21)/denom;

	d11 = 2*(p2.x-p1.x);
	d12 = p2.x*p2.x + p2.y*p2.y - (p1.x*p1.x + p1.y*p1.y);
	d21 = 2*(p3.x - p1.x);
	d22 = p3.x*p3.x + p3.y*p3.y - (p1.x*p1.x + p1.y*p1.y);

	b = (d11*d22-d12*d21)/denom;

	center.x = a;
	center.y = b;

	return center;
}
float getRadius(Point p, Point center){
	float r;
	r = sqrt( (p.x - center.x)*(p.x - center.x) + (p.y - center.y)*(p.y - center.y));
	return r;
}
float getCircleDist(Point p, Point center, float radius){
	float dist;
	dist = abs(sqrt((p.x - center.x)*(p.x - center.x) + (p.y - center.y)*(p.y - center.y)) - radius);
	return dist;
}

int main(int argc, char **argv)
{
	Mat img, imgColor;

	Mat highway, empire, seaside;
	Mat highwayColor, empireColor, seasideColor;

	// Import Documents in Greyscale
	highway = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 3\\highway.jpg",0);
	empire = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 3\\empire_state_building.jpg", 0);
	seaside = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 3\\seaside.jpg", 0);

	highwayColor = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 3\\highway.jpg",CV_LOAD_IMAGE_COLOR);
	empireColor = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 3\\empire_state_building.jpg", CV_LOAD_IMAGE_COLOR);
	seasideColor = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 3\\seaside.jpg", CV_LOAD_IMAGE_COLOR);

//	LineRANSAC(highway,highwayColor,55);
//	imshow("W1",highwayColor);
//
//	LineRANSAC(empire,empireColor,60);
//	imshow("W2",empireColor);
//
//	LineRANSAC(seaside,seasideColor,40);
//	imshow("W3",seasideColor);

	Mat circle, concircle, parliament;
	Mat circleColor, concircleColor, parliamentColor;

	circle = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 3\\circle.jpg",0);
	concircle = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 3\\concentric_circles.jpg",0);
	parliament = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 3\\parliament_clock.jpg",0);

	circleColor = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 3\\circle.jpg",CV_LOAD_IMAGE_COLOR);
	concircleColor = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 3\\concentric_circles.jpg",CV_LOAD_IMAGE_COLOR);
	parliamentColor = cv::imread("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 3\\parliament_clock.jpg",CV_LOAD_IMAGE_COLOR);
	// Circle RANSAC

//	img = circle.clone();
//	imgColor = circleColor.clone();

//	img = concircle.clone();
//	imgColor = concircleColor.clone();

	img = parliament.clone();
	imgColor = parliamentColor.clone();

	int thres = 80;
	GaussianBlur(img,img,Size(5,5), 2, 2);
	Canny(img,img,thres,thres*3,3, true);
	// Identify all valid points post-canny filtering and store in a vector
	vector<cv::Point > imgPoints;
	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			if (img.at<unsigned char>(row, col) == 255) {
				imgPoints.push_back(Point(col, row)); // (x:col,y:row)
			}
		}
	}

	imshow("W1",img);
	imshow("W2",imgColor);

	int iter = 4000;
	int minPoints = 100;
	int getBestCircles = 10;
	int showCircles = 1;
	int inlierCount;
	float maxDist = 1;
	float radius, dist;
	Mat circles(iter, 5, CV_32F, Scalar(0.0));
	vector<Point > inliers; // Vector of points that are inliers

	cv::Point p1,p2,p3,p4,c;
	RNG rand;

	for(int i = 0; i < iter; i++){
		inlierCount = 0;

		do{
			p1 = imgPoints.at(rand.uniform(0, imgPoints.size()));
			p2 = imgPoints.at(rand.uniform(0, imgPoints.size()));
			p3 = imgPoints.at(rand.uniform(0, imgPoints.size()));
			c = getCenter(p1,p2,p3);
			radius = getRadius(p1,c);
		}while(radius > min(img.rows/2,img.cols/2));

		for(int j = 0; j < imgPoints.size(); j++){
			p4 = imgPoints.at(j);
			dist = getCircleDist(p4, c, radius);
			if(dist <= maxDist){
				inlierCount++;
				inliers.push_back(p4);
			}
		}

		Point2f newCenter;
		float newRadius;

		if(inlierCount > minPoints){
			minEnclosingCircle(inliers,newCenter,newRadius);
			circles.at<float>(i,0) = inlierCount;
			circles.at<float>(i,1) = newCenter.x;
			circles.at<float>(i,2) = newCenter.y;
			circles.at<float>(i,3) = newRadius;
			circles.at<float>(i,4) = inliers.size()/(2*3.14159*newRadius);
			//cout << inliers.size()/(2*3.14159*newRadius) << endl;
		}
		inliers.clear();
	}

	Mat sortedIndecies(circles.size(), circles.type());
	Mat circles2(getBestCircles, 5, CV_32F, Scalar(0.0));
	sortIdx(circles,sortedIndecies,CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

	int index;
	for (int n = 0; n < getBestCircles; n++) {
		index = sortedIndecies.at<int>(n, 4);
		circles2.at<float>(n,0) = circles.at<float>(index,0);
		circles2.at<float>(n,1) = circles.at<float>(index,1);
		circles2.at<float>(n,2) = circles.at<float>(index,2);
		circles2.at<float>(n,3) = circles.at<float>(index,3);
		circles2.at<float>(n,4) = circles.at<float>(index,4);
	}
	sortIdx(circles2,sortedIndecies,CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

	for(int n = 0; n < showCircles; n++){
		index = sortedIndecies.at<int>(n, 3);
		float cx = circles2.at<float>(index, 1);
		float cy = circles2.at<float>(index, 2);
		int r =  circles2.at<float>(index, 3);
		Point c = Point(cx,cy);
		if (!(r == 0) && r < min(img.rows/2,img.cols/2)) {
			cv::circle(imgColor,c,r,Scalar(0,0,255),2,LINE_8,0);
		}
	}

	imshow("Resulted Image", imgColor);

	cv::waitKey();
	return 0;
}
