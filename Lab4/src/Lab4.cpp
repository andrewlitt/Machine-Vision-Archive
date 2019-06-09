#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
using namespace cv;
using namespace std;

void addToDataSet(Mat &data, vector<String> &labels, Mat &newData, vector<String> &newLabels);
Mat norm_0_255(Mat src);
String recognizeFace(Mat query, Mat samples, vector<String> labels);

int main()
{
	// Load your data and combine it with the data set of several of your peers using:
	// addToDataSet

	FileStorage fs;
	Mat samples, tempSamples, trainSamples;
	vector<String> labels, tempLabels, trainLabels;

	fs.open("10150478.xml", FileStorage::READ);
	fs["samples"] >> samples;
	fs["labels"] >> labels;
	fs.release();

	fs.open("10149982.xml", FileStorage::READ);
	fs["samples"] >> tempSamples;
	fs["labels"] >> tempLabels;
	fs.release();

	addToDataSet(samples, labels, tempSamples, tempLabels);

	fs.open("10142394.xml", FileStorage::READ);
	fs["samples"] >> tempSamples;
	fs["labels"] >> tempLabels;
	fs.release();

	addToDataSet(samples, labels, tempSamples, tempLabels);

	fs.open("trainsamples.xml", FileStorage::READ);
	fs["samples"] >> trainSamples;
	fs["labels"] >> trainLabels;
	fs.release();


	cout << samples.size() << endl;
	// Perform PCA
	PCA pca(samples, Mat(), CV_PCA_DATA_AS_ROW);
	//Visualize Mean

    Mat meanFace = norm_0_255(pca.mean);

    // normalize and reshape mean
    imshow("meanFace", meanFace);
    waitKey();

    //Visualize Eigenfaces
    for(unsigned int i = 0; i < pca.eigenvectors.rows; i++)
    {
        Mat eigenface;
        eigenface = norm_0_255(pca.eigenvectors.row(i).clone());

        // normalize and reshape eigenface
        applyColorMap(eigenface, eigenface, COLORMAP_JET);

        imshow(format("eigenface_%d", i), eigenface);
        waitKey();
    }

    Mat output;
    // Project all samples into the Eigenspace
	pca.project(samples,output);
	cout << output.size() << endl;
	cout << output << endl;

	for(int k = 0; k < samples.rows; k++){
		Mat queryOutput;
		pca.project(samples.row(k).clone(),queryOutput);

		String guess = recognizeFace(queryOutput,output,labels);
		String answer = labels[k];
		cout << "Guess: " << guess << endl;
		cout << "Answer: " << answer << endl;
	}

//	for(int k = 0; k < trainSamples.rows; k++){
//
//		Mat queryOutput;
//		pca.project(trainSamples.row(k).clone(),queryOutput);
//
//		String guess = recognizeFace(queryOutput,output,trainLabels);
//		String answer = trainLabels[k];
//		cout << "Train Guess: " << guess << endl;
//		cout << "Train Answer: " << answer << endl;
//	}
	// ID Faces
	// code..

    return 0;
}

void addToDataSet(Mat &samples, vector<String> &labels, Mat &newSamples,vector<String> &newLabels) {
	for(int i = 0; i < 5; i++){
		samples.push_back(newSamples.row(i));
		labels.push_back(newLabels[i]);
	}
}

Mat norm_0_255(Mat src)
{
	cv::Mat tmp(200,200,CV_32F, Scalar());

	for(int i = 0; i < 200; i++){
		for(int j = 0; j < 200; j++){
			tmp.at<float>(i,j) = src.at<float>(0,i*200+j);
		}
	}
	//cout << src.size() << endl;

	float minValue = 1000;
	float maxValue = 0;
	float val;

	for(int i = 0; i < 200; i++){
		for(int j = 0; j < 200; j++){
			val = tmp.at<float>(i,j);
			if(val < minValue){
				minValue = val;
			}
			if(val > maxValue){
				maxValue = val;
			}
		}
	}
	//cout << minValue << endl;
	//cout << maxValue << endl;

	for(int i = 0; i < 200; i++){
		for(int j = 0; j < 200; j++){
			val = tmp.at<float>(i,j);
			tmp.at<float>(i,j) = 255*(val-minValue)/(maxValue-minValue);

		}
	}
	tmp.convertTo(tmp, CV_8UC3,1,0);
	return tmp;
}

String recognizeFace(Mat query, Mat samples,vector<String> labels)
{
	String ans = "test";
	Size q = query.size();
	Size s = samples.size();
//	cout << q << endl;
//	cout << s << endl;
	int numsamples = s.height;
	int features = s.width;

	int minIndex;
	float minDist = 99999;
	for (int i = 0; i < numsamples; i++){
		volatile float totalDist = 0;

		for(int j = 0; j < features; j++){
			totalDist += abs(samples.at<uchar>(i,j)-query.at<uchar>(0,j));
		}
		if(totalDist < minDist){
			minDist = totalDist;
			minIndex = i;
		}
	}

	ans = labels[minIndex];

	return ans;
}

