#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

int  main(int argc, char* argv[])
{
	int numIter = 50;
	int NUM_SAMP = 4;
	int maxInlier = 0;
	double SQR_TOL = 10000.0;
	// load images, convert them to gray images
	Mat img1 = imread ("bookshelf_1.jpg", 1);
	Mat img2 = imread ("bookshelf_2.jpg", 1);
	Mat grayimg1, grayimg2;
	cvtColor(img1, grayimg1, CV_BGR2GRAY);
	cvtColor(img2, grayimg2, CV_BGR2GRAY);
	
	// for box.jpg & scene.jpg
	/*Mat grayimg1 = imread ("box.jpg", 0);
	Mat grayimg2 = imread ("scene.jpg", 0);*/

	// SIFT feature extraction
	SIFT sift;
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	bool useProvidedKeypoints=false;
	sift.detect(grayimg1, keypoints1);
	sift.detect(grayimg2, keypoints2);
	sift.operator()(grayimg1, noArray(), keypoints1, descriptors1, true);
	sift.operator()(grayimg2, noArray(), keypoints2, descriptors2, true);

	// find matching pairs
	vector<int> pair1(descriptors1.rows,0);
	vector<int> pair2(descriptors2.rows,0);
	vector<double> min_distance1(descriptors1.rows,0);
	vector<double> min_distance2(descriptors2.rows,0);
	vector<double> second_min_distance1(descriptors1.rows,0);
	vector<double> second_min_distance2(descriptors2.rows,0);

	for (int i = 0; i < descriptors1.rows; i ++)
	{
		uchar* row1i = descriptors1.ptr<uchar>(i); // the ith row of descriptors 1
		double mindis = 0.0; // min distance
		double diff = 0.0;
		int minrow = 0;
		// distance with the 0th row of descriptors2
		for (int j = 0; j < 128; j++)
			diff += pow((double)descriptors1.at<float>(i,j) - descriptors2.at<float>(0,j), 2.0);
		mindis = sqrt(diff);
		min_distance1[i] = mindis;
		second_min_distance1[i] = mindis;
		// find the row number has the min distance
		for (int k = 1; k < descriptors2.rows; k++)
		{
			diff = 0.0;
			for (int j = 0; j < 128; ++ j)
				diff += pow((double)descriptors1.at<float>(i,j) - descriptors2.at<float>(k,j), 2.0);
			if (second_min_distance1[i] >= sqrt(diff))
			{
				second_min_distance1[i] = sqrt(diff);
				if (second_min_distance1[i] < min_distance1[i])
				{
					double tmp = second_min_distance1[i];
					second_min_distance1[i] = min_distance1[i];
					min_distance1[i] = tmp;
					minrow = k;
				}
				//minrow = k; // row i in descriptors1 has min distance to row k in descriptor2
			}
		}
		pair1[i] = minrow;
	}

	for (int i = 0; i < descriptors2.rows; i++)
	{
		double mindis = 0.0; // min distance
		double diff = 0.0;
		int minrow = 0;
		// distance with the 0th row of descriptors1
		uchar* row10 = descriptors1.ptr<uchar>(0);
		for (int j = 0; j < 128; j++)
			diff += pow((double)descriptors2.at<float>(i,j) - descriptors1.at<float>(0,j), 2.0);
		mindis = sqrt(diff);
		min_distance2[i] = mindis;
		second_min_distance2[i] = mindis;
		// find the row number has the min distance
		for (int k = 1; k < descriptors1.rows; k++)
		{
			diff = 0.0;
			for (int j = 0; j < 128; ++ j)
				diff += pow((double)descriptors2.at<float>(i,j) - descriptors1.at<float>(k,j), 2.0);
			if (second_min_distance2[i] >= sqrt(diff))
			{
				second_min_distance2[i] = sqrt(diff);
				if (second_min_distance2[i] < min_distance2[i])
				{
					double tmp = second_min_distance2[i];
					second_min_distance2[i] = min_distance2[i];
					min_distance2[i] = tmp;
					minrow = k;
				}
				//minrow = k; // row i in descriptors1 has min distance to row k in descriptor2
			}
		}
		pair2[i] = minrow;
	}

	// connect row i and row k
	vector<int> match;
	//Mat output;
	//cvtColor(grayimg2, output, CV_GRAY2BGR);
	for (int i = 0; i< descriptors1.rows; i++)
	{
		if (i == pair2[pair1[i]])
		{
			if (second_min_distance1[i] / min_distance1[i] > 1.5) 
				match.push_back(i);
		}
	}
	cout<<"number of match pairs: "<<match.size()<<endl;

	// RANSAC Homography
	int iteration=0; Mat homo;
	srand (time(0));
	while(iteration < numIter)
	{
		// sample random pairs
		//srand (time(0));
		int ran_num[4];
		for (int i=0; i<NUM_SAMP; i++)
			ran_num[i] = rand() % (match.size());
		// solve for H
		//Mat b = (Mat_<double>(12, 1) << 0,0,0,0,0,0,0,0,-1,-1,-1,-1);
		
		double x[4], y[4], u[4], v[4];
		for (int i=0; i<NUM_SAMP; i++)
		{
			x[i] = keypoints1[match[ran_num[i]]].pt.x;// - grayimg1.cols/2;
			y[i] = keypoints1[match[ran_num[i]]].pt.y;// - grayimg1.rows/2;
			u[i] = keypoints2[pair1[match[ran_num[i]]]].pt.x;// - grayimg1.cols/2;
			v[i] = keypoints2[pair1[match[ran_num[i]]]].pt.y;// - grayimg1.rows/2;
		}
		Mat b = (Mat_<double>(8, 1) << u[0], v[0], u[1], v[1], u[2], v[2], u[3], v[3]);
		/*Mat a = (Mat_<double>(12, 12) <<
					 x[0], y[0],   1,    0,    0,   0,    0,    0, -u[0],     0,     0,     0,
					 x[1], y[1],   1,    0,    0,   0,    0,    0,     0, -u[1],     0,     0,
					 x[2], y[2],   1,    0,    0,   0,    0,    0,     0,     0, -u[2],     0,
					 x[3], y[3],   1,    0,    0,   0,    0,    0,     0,     0,     0, -u[3],
						0,    0,   0, x[0], y[0],   1,    0,    0, -v[0],     0,     0,     0,
						0,    0,   0, x[1], y[1],   1,    0,    0,     0, -v[1],     0,     0,
						0,    0,   0, x[2], y[2],   1,    0,    0,     0,     0, -v[2],     0,
						0,    0,   0, x[3], y[3],   1,    0,    0,     0,     0,     0, -v[3],
						0,    0,   0,    0,    0,   0, x[0], y[0],    -1,     0,     0,     0,
						0,    0,   0,    0,    0,   0, x[1], y[1],     0,    -1,     0,     0,
						0,    0,   0,    0,    0,   0, x[2], y[2],     0,     0,    -1,     0,
						0,    0,   0,    0,    0,   0, x[3], y[3],     0,     0,     0,    -1);*/
		Mat a = (Mat_<double>(8, 8) <<
					 x[0], y[0],   1,    0,    0,   0, -u[0]*x[0], -u[0]*y[0], 
					    0,    0,   0, x[0], y[0],   1, -v[0]*x[0], -v[0]*y[0], 
					 x[1], y[1],   1,    0,    0,   0, -u[1]*x[1], -u[1]*y[1], 
					    0,    0,   0, x[1], y[1],   1, -v[1]*x[1], -v[1]*y[1], 
					 x[2], y[2],   1,    0,    0,   0, -u[2]*x[2], -u[2]*y[2], 
					    0,    0,   0, x[2], y[2],   1, -v[2]*x[2], -v[2]*y[2], 
					 x[3], y[3],   1,    0,    0,   0, -u[3]*x[3], -u[3]*y[3], 
					    0,    0,   0, x[3], y[3],   1, -v[3]*x[3], -v[3]*y[3] );
		Mat H (8, 1, CV_64F);
		solve (a, b, H, DECOMP_LU);
		/*for (int ii=0; ii<8; ii++)
			cout<<"H"<<ii<<": "<<H.at<double>(ii)<<endl;*/

		/*double thres = min_distance1[match[0]];
		for(int i=0; i<match.size(); i++)
			if(thres >= min_distance1[match[i]])
				thres = min_distance1[match[i]];*/
		int numInlier = 0; 
		for(int i=0; i<match.size(); i++)
		{
			int xh, yh; double w, d=0;
			w = keypoints1[match[i]].pt.x * H.at<double>(6) + keypoints1[match[i]].pt.y * H.at<double>(7) + 1;
			xh = (int) (keypoints1[match[i]].pt.x * H.at<double>(0) + keypoints1[match[i]].pt.y * H.at<double>(1) + H.at<double>(2)) / w;
			yh = (int) (keypoints1[match[i]].pt.x * H.at<double>(3) + keypoints1[match[i]].pt.y * H.at<double>(4) + H.at<double>(5)) / w;
		    //cout<<"xh: "<<xh<<" yh: "<<yh<<endl;
			//cout<<"w: "<<w<<endl;
			for(int j=0; j<match.size(); j++)
			{
				if (-0.1 < keypoints2[pair1[match[j]]].pt.x - xh < 0.1 && -0.1 < keypoints2[pair1[match[j]]].pt.y - yh < 0.1)
				{
					for (int m = 0; m < 128; m ++)
						d += pow((double)descriptors2.at<float>(pair1[match[i]], m) - descriptors2.at<float>(pair1[match[j]], m), 2.0);
					if (d < SQR_TOL)
						numInlier ++;
					//cout<<d<<endl;
					break;
				}
			}
		}
		if (numInlier > maxInlier)
		{maxInlier = numInlier; homo = H;}
		iteration ++;
	}

	// display
	for(int i=0; i<match.size(); i++)
	{
		int x = keypoints2[pair1[match[i]]].pt.x;
		int y = keypoints2[pair1[match[i]]].pt.y;
		double w = keypoints1[match[i]].pt.x * homo.at<double>(6) + keypoints1[match[i]].pt.y * homo.at<double>(7) + 1;
		int xh = (int) (keypoints1[match[i]].pt.x * homo.at<double>(0) + keypoints1[match[i]].pt.y * homo.at<double>(1) + homo.at<double>(2)) / w;
		int yh = (int) (keypoints1[match[i]].pt.x * homo.at<double>(3) + keypoints1[match[i]].pt.y * homo.at<double>(4) + homo.at<double>(5)) / w;

		if (yh-1>-1 && yh+1<grayimg2.rows && xh-1>-1 && xh+1<grayimg2.cols)
		{
			line(img2, Point(xh, yh), keypoints2[pair1[match[i]]].pt, CV_RGB(255,0,0),1,8,0);
			for(int l=-1; l<2; l++)
				for(int m=-1; m<2; m++)
				{
				img2.at<Vec3b>(y+l, x+m )[0] = 255;
				img2.at<Vec3b>(y+l, x+m )[1] = 0;
				img2.at<Vec3b>(y+l, x+m )[2] = 0;

				img2.at<Vec3b>(yh+l, xh+m)[0] = 0;
				img2.at<Vec3b>(yh+l, xh+m)[1] = 255;
				img2.at<Vec3b>(yh+l, xh+m)[2] = 0;
				}
		}
	}
	/*for(int i=0; i<descriptors1.rows; i++)
		if(min_distance1[i] < 3*thres && i == pair2[pair1[i]])
		{
			keypoints2[pair1[i]].pt.x = keypoints2[pair1[i]].pt.x + (float)grayimg1.cols;
			line(output, keypoints1[i].pt, keypoints2[pair1[i]].pt, CV_RGB(255,0,0),1,8,0);
		}*/

	for (int i=0; i<8; i++)
		cout<<homo.at<double>(i)<<endl;

	namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
    imshow( "Display window", img2 );
	imwrite("bookshelf_result.jpg", img2 );
	waitKey(0);
    return 0;
}