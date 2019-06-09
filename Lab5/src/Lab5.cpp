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

int main()
{
	VideoCapture video("C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 5\\wingsuit.mp4");

	int minHessian = 300;
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	Mat frame1, frame2, frameColor;


	Ptr<SURF> detector = SURF::create();
	detector->setHessianThreshold(minHessian);

	video >> frame2;
	cvtColor(frame2, frame2, CV_RGB2GRAY);
	detector->detectAndCompute(frame2, Mat(), keypoints2, descriptors2); // @suppress("Invalid arguments")

	int rows = frame2.rows;
	int cols = frame2.cols;

	Mat velocityImage = Mat::zeros(rows,cols,CV_8UC3);

	int index1, index2;
	float mag, angle;
	Point2f bluePt, redPt, disp;
	vector<float> magnitudes;
	vector<float> directions;
	vector<Point2f> p_set;

	while(1){

		frame1 = frame2.clone();
		keypoints1 = keypoints2;
		descriptors1 = descriptors2.clone();

		video >> frameColor;
		video >> frameColor;
		if(!frameColor.data) break;
		cvtColor(frameColor, frame2, CV_RGB2GRAY);

		detector->detectAndCompute(frame2, Mat(), keypoints2, descriptors2); // @suppress("Invalid arguments")

		BFMatcher matcher(NORM_L2,true); // @suppress("Abstract class cannot be instantiated")
		vector<DMatch> matches;
		matcher.match(descriptors2, descriptors1, matches); // @suppress("Invalid arguments")

		Subdiv2D subdiv(Rect(0,0,frame2.cols,frame2.rows));

		for(int i = 0; i < matches.size(); i++){

			if(matches[i].distance <= 0.25){
//				cout << matches[i].distance << endl;
				index2 = matches[i].queryIdx;
				index1 = matches[i].trainIdx;
				bluePt = keypoints2[index2].pt;
				redPt = keypoints1[index1].pt;

				disp = bluePt-redPt;
				mag = sqrt(disp.x*disp.x + disp.y+disp.y);
				angle = atan2(-disp.y,disp.x);
//				cout << disp << endl;
//				cout << mag << endl;
//				cout << angle << endl;
				if(mag < 15){
					circle(frameColor,bluePt,1,Scalar(0,0,255),1,8,0);
					circle(frameColor,redPt,1,Scalar(255,0,0),1,8,0);
					line(frameColor,redPt,bluePt,Scalar(0,255,255),1,8,0);
					magnitudes.push_back(mag);
					directions.push_back(angle);
					p_set.push_back(bluePt);
					subdiv.insert(bluePt);
				}

			}
		}
		vector<vector<Point2f> > facets;
		vector<Point2f> centers;
		subdiv.getVoronoiFacetList(vector<int>(), facets, centers); // @suppress("Invalid arguments")

		float h,s,v;
		float v_limit = 10;
		Scalar color;
		for (int i = 0; i < (int)facets.size(); i++) {
			vector<Point2f> facet;
			facet = facets[i];
			vector<Point> facet_pts;
			for (int j = 0; j < facet.size(); j++)
				facet_pts.push_back(facet[j]);
			mag = magnitudes[i];
			angle = directions[i]+3.14159;
//			cout << mag << endl;
//			cout << angle << endl;
			h = 180*angle/(2*M_PI);
			s = 255*mag/v_limit;
			v = 128*mag/v_limit + 127;
			color = Scalar(h, s, v);
//			cout << color << endl;
			fillConvexPoly(velocityImage, facet_pts, color, 8, 0); // @suppress("Invalid arguments")
		}
		cout << velocityImage.size << endl;
		Mat velocityImage2;
		cvtColor(velocityImage,velocityImage2,CV_HSV2BGR);
		cv::namedWindow("Window", WINDOW_AUTOSIZE);
		cv::imshow("Window",velocityImage);
		cv::imshow("video",frameColor);
		if(waitKey(30) >= 0) break;

		magnitudes.clear();
		directions.clear();
		p_set.clear();
	}

    return 0;
}
