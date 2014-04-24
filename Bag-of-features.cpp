#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	// get PCA-SIFT featrures
	Mat siftFeatures;
	int siftNum[100];
	int ct=0;
	string name;

	while (ct!=5)
	{
		switch(ct) 
		{
		case 0:
			name = "train_car_";
			break;
		case 1:
			name = "train_face_";
			break;
		case 2:
			name = "train_laptop_";
			break;
		case 3:
			name = "train_motorbike_";
			break;
		case 4:
			name = "train_pigeon_";
			break;
		}

		for (int i=1; i!=21; i++)
		{
			ostringstream oss;
			string file;

			if(i < 10)
			{
				oss << i;
				file = name + "0" + oss.str() + ".jpg";
			}
			else
			{
				oss << i;
				file = name + oss.str() + ".jpg";
			}
		
			Mat grayimg;
			if(ct!=0)
			{
				Mat img = imread(file, 1);
				cvtColor(img, grayimg, CV_BGR2GRAY);
			}
			else
			{
				grayimg = imread(file, 0);
			}

			SIFT sift;
			vector<KeyPoint> keypoints;
			Mat descriptors;
			sift.operator()(grayimg, noArray(), keypoints, descriptors, false);
			siftFeatures.push_back(descriptors);
			siftNum[ct*20 + i -1] = descriptors.rows;
		}
		ct++;
	}



	Mat covar, mean;
	//const Mat *samples = &siftFeatures;
	calcCovarMatrix(siftFeatures, covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_64F);
	Mat eigenvalues, eigenvectors;
	eigen(covar, eigenvalues, eigenvectors);
	Mat priDir;



	for(int i=0; i<20; i++)
		priDir.push_back(eigenvectors.row(i));
	Mat priDir_t = priDir.t();
	Mat sift64f;
	siftFeatures.convertTo(sift64f, CV_64F);
	Mat pcaSift = sift64f * priDir_t;
	

	// k-means clustering
	Mat bestLabels;
	Mat pcaSift32f;
	pcaSift.convertTo(pcaSift32f, CV_32F);
	Mat centers;
	const int K = 120;
	kmeans(pcaSift32f, K, bestLabels, TermCriteria( CV_TERMCRIT_EPS, 100, .5), 5, KMEANS_PP_CENTERS, centers);



	// compute histogram
	int sum[100][K] = {0};
	int step = 0;
	for(int i=0; i<100; i++)
	{
		for(int j=0; j<siftNum[i]; j++)
		{
			if(i==0)
				sum[i][bestLabels.at<int>(j)]++;
			else
				sum[i][bestLabels.at<int>(j + step)]++;
			//cout<<bestLabels.at<int>(j);
		}
		step = step + siftNum[i];
	}


	// Object  Recognition
	for (int ii=1; ii!=11; ii++)
	{
		ostringstream oss;
		string file;

		if(ii < 10)
		{
			oss << ii;
			file = "test_face_0" + oss.str() + ".jpg"; //
		}
		else
		{
			oss << ii;
			file = "test_face_" + oss.str() + ".jpg";
		}

		//Mat testImg = imread("test_car_10.jpg", 0);
		Mat testImg = imread(file, 1);
		Mat grayTestImg;
		cvtColor(testImg, grayTestImg, CV_BGR2GRAY);
		SIFT sift;
		vector<KeyPoint> keypoints;
		Mat descriptors;
		sift.operator()(grayTestImg, noArray(), keypoints, descriptors, false);
		Mat feature;
		Mat descriptors64f;
		descriptors.convertTo(descriptors64f, CV_64F);
		feature = descriptors64f * priDir_t;
	
		double **distance;  // i: each feature; j: to each center
		distance = new double * [feature.rows];
		for(int i=0; i<feature.rows; i++){
			distance[i] = new double [K];
			for (int j=0; j< K; j++)
				distance[i][j] = 0;
		}

		for(int i=0; i<feature.rows; i++)
			for(int j=0; j<K; j++){
				for(int k=0; k<20; k++)
					distance[i][j] += pow(feature.at<double>(i,k) - centers.at<float>(j,k), 2.0);
				distance[i][j] = sqrt(distance[i][j]);
			}

		int histo[K] = {0};
		vector<int> mindis;
		for(int i=0; i<feature.rows; i++)
		{
			int temp=0;
			for(int j=1; j<K; j++)
			{
				if(distance[i][temp] > distance[i][j])
					temp = j;
			}
			mindis.push_back(temp);
		}

		for(int i=0; i<feature.rows; i++)
		{
			histo[mindis[i]]++;
		}

		/*for(int i=0; i<feature.rows; i++)
		{
			double mindis = distance[i][0];
			int temp=0;
			for(int j=0; j<K; j++)
			{
				if(mindis > distance[i][j]) {
					mindis = distance[i][j];
					temp = j;
				}
			}
			histo[temp] ++;
		}*/

		

		double diff[100] = {0};
		for(int i=0; i<100; i++){
			for(int j=0; j<K; j++){
				diff[i] += (histo[j] - sum[i][j]) * (histo[j] - sum[i][j]);
			}
			diff[i] = sqrt(diff[i]);
		}

		const int neib = 10; // number of neighbors
		Mat dst;
		Mat src(1,100,CV_64F,&diff);
		sortIdx(src,dst,CV_SORT_EVERY_ROW | CV_SORT_ASCENDING );

		double vote[5] = {0};
		for(int i=0; i<neib; i++)
		{
			if(dst.at<int>(i) >= 0 && dst.at<int>(i) < 20)
				vote[0] += 1/diff[dst.at<int>(i)];
			else if(dst.at<int>(i) >= 20 && dst.at<int>(i) < 40)
				vote[1] += 1/diff[dst.at<int>(i)];
			else if(dst.at<int>(i) >= 40 && dst.at<int>(i) < 60)
				vote[2] += 1/diff[dst.at<int>(i)];
			else if(dst.at<int>(i) >= 60 && dst.at<int>(i) < 80)
				vote[3] += 1/diff[dst.at<int>(i)];
			else
				vote[4] += 1/diff[dst.at<int>(i)];
			cout<<dst.at<int>(i)<<"    "<<diff[dst.at<int>(i)]<<endl;
		}
		int max=0;
		for(int i=0; i<5; i++)
		{
			if(vote[max] < vote[i])
				max=i;
		}
		switch(max) 
			{
			case 0:
				cout<<ii<<" is: car"<<endl;
				break;
			case 1:
				cout<<ii<<" is: face"<<endl;
				break;
			case 2:
				cout<<ii<<" is: laptop"<<endl;
				break;
			case 3:
				cout<<ii<<" is: motorbike"<<endl;
				break;
			case 4:
				cout<<ii<<" is: pigeon"<<endl;
				break;
			}


		delete[] distance;
	}
	system("pause");
	return 0;
}
