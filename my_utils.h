#ifndef MY_UTILS_H
#define MY_UTILS_H

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/sfm/fundamental.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/sfm/triangulation.hpp>
#include <vector>
#include <opencv2/FeatureDetectionClass.h>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/QR>
#include <cmath>
#include <dirent.h>
#include <sys/types.h>
#include <map>
#include <algorithm>
#include <chrono>
//#include <boost/thread/thread.hpp>
//#include <boost/algorithm/algorithm.hpp>
//#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Dense>
#include <set>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

vector<string> read_directory(const string &path);
vector<char*> readFiles(char* dir, int num);

Mat multiplyMatrix(Mat& mat1, Mat& mat2);
Mat plusMatrix(Mat& mat1, Mat& mat2);
Mat multiplyElement(Mat& mat1, Mat& mat2);
Mat diagMatrix(Mat& mat1);
Mat doubleMatrixElement(Mat& mat1, int row, int col);
Mat divideMatrixElement(Mat& mat1, Mat& mat2);
void splitMatrix(Mat& mat, int startrow, Mat& dst);

void epipolarLineImage(Mat& resultImage, Mat& image1, Mat& image2, vector<Point2f>& points1, vector<Point2f>& points2, Mat fundamentalMatrix, int whichImage); // draw epi
vector<Point3d> getRGB(Mat& image1, vector<Point2d>& points1); // get RGB of 3D points

//void SavePointCloudPLY(string fileName, string path, vector<Point3d> object3Dpoints, vector<Point3d> objectRGB);
void SavePointCloudPLY(string fileName, string path, vector<Point3d> object3Dpoints, vector<Point3d> objectRGB, vector<Point3d> cam3Dpts, vector<Point3d> camRGB); // save 3d points

#endif