#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

void addToDataSet(Mat &data, vector<string> &labels, Mat &newData, vector<string> &newLabels);
void resizeFace(Mat &src);
string recognizeFace(Mat query, Mat samples, vector<string> labels);
Mat detectFace ( const Mat &image , CascadeClassifier &faceDetector);

int main() {

	std::vector<String> names;
//	names.push_back("matthews.jpg");
//	names.push_back("nylander.jpg");
//	names.push_back("gardiner.jpg");
//	names.push_back("polak.jpg");
//	names.push_back("kadri.jpg");



	names.push_back("drakeTest.jpg");
	names.push_back("nylanderTest.jpg");
	names.push_back("demarTest.jpg");
	//names.push_back("littTest.jpg");
	//names.push_back("matthewsTest.jpg");

	Mat samples;
	std::vector<String> labels;

	String face_cascade_name = "C:\\opencv\\opencv_3.4.0_Contrib\\opencv-3.4.0\\samples\\winrt\\FaceDetection\\FaceDetection\\Assets\\haarcascade_frontalface_alt.xml";
	CascadeClassifier face_cascade;
	if( !face_cascade.load( face_cascade_name ) ){ printf("Can't Load Cascade Classifier\n"); return -1; };

	for(int i = 0; i<3; i ++){

		String url = "C:\\Users\\andre\\Documents\\School - 4th Year\\ELEC 474\\Lab 4\\";
		url = url + names[i];
		Mat faceImg = imread(url,CV_LOAD_IMAGE_GRAYSCALE);
		//imshow("url",faceImg);
		waitKey(0);
		Mat face;

		face = detectFace(faceImg , face_cascade);
		resizeFace(face);
		imshow("url",face);
		waitKey(0);
		samples.push_back(face.clone().reshape(1,1));
		labels.push_back(names[i].c_str());
	}

	FileStorage fs;
	fs.open("trainsamples.xml", FileStorage::WRITE);

	fs << "samples" << samples;
	fs << "labels" << labels;

	fs.release();

	return 0;

}

void resizeFace( Mat &src) {
	int newCols = 200;
	int newRows = 200;
	resize(src, src, Size(newCols, newRows));
}

void addToDataSet(Mat &samples, vector<string> &labels, Mat &newSamples,vector<string> &newLabels) {
	for(int i = 0; i < 5; i++){
		samples.push_back(newSamples.row(i));
		labels.push_back(newLabels[i]);
	}
}

Mat detectFace ( const Mat &image , CascadeClassifier &faceDetector)
{
	vector<Rect> faces;
	faceDetector.detectMultiScale(image, faces, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30,30));

	if(faces.size()==0)
	{
		cerr<< "ERROR: No Faces found" << endl;
		return Mat();
	}
	if(faces.size() > 1 )
	{
		cerr << "ERROR: Multiple Faces Found" << endl;
		return Mat();
	}

	return image (faces[0]).clone();
}
