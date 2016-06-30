#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

int  main(int argc, char* argv[])
{
	// load images, convert them to gray images
	Mat img1 = imread ("gundam_1.jpg", 1);
	Mat img2 = imread ("gundam_2.jpg", 1);
	Mat grayimg1(img1.size(), CV_8UC1), grayimg2(img2.size(), CV_8UC1);
	cvtColor(img1, grayimg1, CV_BGR2GRAY);
	cvtColor(img2, grayimg2, CV_BGR2GRAY);
	
	//Mat grayimg1 = imread ("box.jpg", 0);
	//Mat grayimg2 = imread ("scene.jpg", 0);

	// SIFT feature extraction
	SIFT sift;
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	bool useProvidedKeypoints=false;
	sift.detect(grayimg1, keypoints1);
	sift.detect(grayimg2, keypoints2);
	sift.operator()(grayimg1, noArray(), keypoints1, descriptors1, true);
	sift.operator()(grayimg2, noArray(), keypoints2, descriptors2, true);

	Mat output (grayimg2.rows, grayimg1.cols+grayimg2.cols, CV_8UC3, Scalar::all(0)); // create output image
	// copy two images to the output image
	output.adjustROI(0, 0, 0, -grayimg1.cols);
	img1.copyTo(output);
	output.adjustROI(0, 0, -grayimg1.cols, grayimg2.cols);
	img2.copyTo(output);
	output.adjustROI(0, 0, grayimg2.cols, 0);

	/*Mat output (grayimg2.rows, (grayimg1.cols+grayimg2.cols), CV_8UC1, Scalar::all(0));
	output.adjustROI(0, grayimg1.rows - grayimg2.rows, 0, -grayimg2.cols);
	grayimg1.copyTo(output);
	output.adjustROI(0, grayimg2.rows - grayimg1.rows, -grayimg1.cols, grayimg2.cols);
	grayimg2.copyTo(output);
	output.adjustROI(0, 0, grayimg1.cols, 0);*/

	// find matching pairs
	/*cout<<keypoints1.size()<<"   "<<keypoints2.size()<<endl;
	FileStorage fs("keypoints.yml", FileStorage::WRITE);
	write( fs , "aNameYouLike", keypoints1 );
	fs.release();*/
	cout<<descriptors1.rows<<"   "<<keypoints1.size()<<endl;
	vector<int> pair1(descriptors1.rows,0);
	vector<int> pair2(descriptors2.rows,0);
	vector<double> min_distance1(descriptors1.rows,0);
	vector<double> min_distance2(descriptors2.rows,0);


	for (int i = 0; i < descriptors1.rows; i ++)
	{
		uchar* row1i = descriptors1.ptr<uchar>(i); // the ith row of descriptors 1
		double mindis = 0.0; // min distance
		double diff = 0.0;
		int minrow = 0;
		// distance with the 0th row of descriptors2
		//uchar* row20 = descriptors2.ptr<uchar>(0);
		for (int j = 0; j < 128; j++)
			//diff += pow( (double) row1i[j] - row20[j], 2.0);
			diff += pow((double)descriptors1.at<float>(i,j) - descriptors2.at<float>(0,j), 2.0);
		mindis = sqrt(diff);
		min_distance1[i] = mindis;
		// find the row number has the min distance
		for (int k = 1; k < descriptors2.rows; k++)
		{
			diff = 0.0;
			//uchar* row2k = descriptors2.ptr<uchar>(k);
			for (int j = 0; j < 128; ++ j)
				//diff += pow( (double) row1i[j] - row2k[j], 2.0);
				diff += pow((double)descriptors1.at<float>(i,j) - descriptors2.at<float>(k,j), 2.0);
			if (mindis >= sqrt(diff))
			{
				mindis = sqrt(diff);
				min_distance1[i] = mindis;
				minrow = k; // row i in descriptors1 has min distance to row k in descriptor2
			}
		}
		pair1[i] = minrow;
	}

	for (int i = 0; i < descriptors2.rows; i++)
	{
		//uchar* row2i = descriptors2.ptr<uchar>(i);
		double mindis = 0.0; // min distance
		double diff = 0.0;
		int minrow = 0;
		// distance with the 0th row of descriptors1
		uchar* row10 = descriptors1.ptr<uchar>(0);
		for (int j = 0; j < 128; j++)
			//diff += pow((double)row2i[j] - row10[j], 2.0);
			diff += pow((double)descriptors2.at<float>(i,j) - descriptors1.at<float>(0,j), 2.0);
		mindis = sqrt(diff);
		min_distance2[i] = mindis;
		// find the row number has the min distance
		for (int k = 1; k < descriptors1.rows; k++)
		{
			diff = 0.0;
			//uchar* row1k = descriptors1.ptr<uchar>(k);
			for (int j = 0; j < 128; ++ j)
				//diff += pow((double)row2i[j] - row1k[j], 2.0);
				diff += pow((double)descriptors2.at<float>(i,j) - descriptors1.at<float>(k,j), 2.0);
			if (mindis >= sqrt(diff))
			{
				mindis = sqrt(diff);
				min_distance2[i] = mindis;
 				minrow = k; // row i in descriptors1 has min distance to row k in descriptor2
			}
		}
		pair2[i] = minrow;
	}
	
	/*double thres1 = min_distance1[0];
	for(int i=0; i<grayimg1.rows; i++)
		if(thres1 >= min_distance1[i])
			thres1 = min_distance1[i];
	double thres2 = min_distance2[0];
	for(int i=0; i<grayimg2.rows; i++)
		if(thres2 >= min_distance2[i])
			thres2 = min_distance2[i];*/

	// connect row i and row k
	//Mat output2;
	Vector<int> match;
	//cvtColor(output, output2, CV_GRAY2BGR);
	for (int i = 0; i< descriptors1.rows; i++)
	{
		if (i == pair2[pair1[i]])
		{
			match.push_back(i);
			//keypoints2[pair1[i]].pt.x = keypoints2[pair1[i]].pt.x + (float)grayimg1.cols;
			//line(output2, keypoints1[i].pt, keypoints2[pair1[i]].pt, CV_RGB(255,0,0),1,8,0);
		}
	}
	double thres = min_distance1[match[0]];
	for(int i=0; i<match.size(); i++)
		if(thres >= min_distance1[match[i]])
			thres = min_distance1[match[i]];

	for(int i=0; i<descriptors1.rows; i++)
		if(min_distance1[i] < 3*thres && i == pair2[pair1[i]])
		{
			keypoints2[pair1[i]].pt.x = keypoints2[pair1[i]].pt.x + (float)grayimg1.cols;
			line(output, keypoints1[i].pt, keypoints2[pair1[i]].pt, CV_RGB(255,0,0),1,8,0);
		}



	namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
    imshow( "Display window", output );
	imwrite("gundam_result.jpg", output );
	waitKey(0);
    return 0;
}