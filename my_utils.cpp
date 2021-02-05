#include "my_utils.h"


vector<string> read_directory(const string &path) {
  vector<string> result;
  dirent *de;
  DIR *dp;
  errno = 0;
  dp = opendir(path.empty() ? "." : path.c_str());
  if (dp) {
    while (true) {
      errno = 0;
      de = readdir(dp);
      if (de == NULL) break;
	  if (string(de->d_name).find("JPG")!=string::npos){
		  result.push_back(string(de->d_name));
	  }
    }
    closedir(dp);
    sort(result.begin(), result.end());
  }

  result.push_back(path);
  return result;
}

// end element : dir
vector<char*> readFiles(char* dir, int num){
	vector<char*> files;
	DIR *dr;
    struct dirent *en;
    dr = opendir(dir); //open all directory
    if (dr) {
        while ((en = readdir(dr)) != NULL) {
		   files.push_back(en->d_name);
        }
        closedir(dr); //close all directory
    }

	files.erase(files.begin(), files.begin()+2);
	sort(files.begin(), files.end());

	files.push_back(dir);

	return files;
}

Mat multiplyMatrix(Mat& mat1, Mat& mat2)
{
	int M = mat1.rows;
	int l = mat1.cols;
	int N = mat2.cols;
	//cout << "N : " << N << endl;
	Mat res = Mat::zeros(M,N, CV_64F);
    if (l != mat2.rows){
		cout << "matrix shape error. check your matrix shape!!!!!!!!!!!!!!!!!!!!!!!1" << endl;
	} else {
		for (int i=0; i<M; i++){
			for (int j=0; j<N; j++){
				float element = 0;
				for (int k=0; k<l; k++){
                    //element += mat1.at<float>(k,i)*mat2.at<float>(j,k);
					vector<float> e1, e2;
					mat1.row(i).col(k).copyTo(e1);
					mat2.row(k).col(j).copyTo(e2);
					element += e1[0]*e2[0];
				}
				res.row(i).col(j) = element;
			}
		}
	}
	return res;
}

Mat plusMatrix(Mat& mat1, Mat& mat2){
    int M = mat1.rows;
    int N = mat1.cols;
    Mat res = Mat::zeros(M,N, CV_64F);
    if (M!=mat2.rows || N!=mat2.cols){
        cout << "matrix shape error. check your matrix shape!!!!!!!!!!!!!!!!!!!!!!!1" << endl;
    }else {
		for (int i=0; i<M; i++){
			for (int j=0; j<N; j++){
				float element = 0;
				vector<float> e1, e2;
				mat1.row(i).col(j).copyTo(e1);
				mat2.row(i).col(j).copyTo(e2);
				element = e1[0]+e2[0];
				res.row(i).col(j) = element;
			}
		}
	}
	return res;
}


void epipolarLineImage(Mat& resultImage, Mat& image1, Mat& image2, vector<Point2f>& points1, vector<Point2f>& points2, Mat fundamentalMatrix, int whichImage){
	vector<Vec3f> lines;

	if (whichImage == 1){
		computeCorrespondEpilines(Mat(points1), whichImage, fundamentalMatrix, lines);
		//circle(resultImage, points1[0], 5, Scalar(255,0,0));
	}
	else{ //2
		computeCorrespondEpilines(Mat(points2), whichImage, fundamentalMatrix, lines);
		//circle(resultImage, points2[0], 5, Scalar(255,0,0));
	}
	
	vector<Vec3f>::iterator it=lines.begin()+1;
	for (vector<Vec3f>::iterator it=lines.begin(); it!=lines.end(); ++it){
        //cout << *it << endl;
		line(resultImage, Point(0,-(*it)[2]/(*it)[1]), Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]), Scalar(255,255,255));
	}
}





vector<Point3d> getRGB(Mat& image1, vector<Point2d>& points1){
	vector<Point3d> res;
	int Num = points1.size();

	for (int i=0; i<Num; i++){
		float x = points1[i].x;
		float y = points1[i].y;

		vector<Point3d> RGB;
		image1.row(y).col(x).copyTo(RGB);
		res.push_back(RGB[0]);
	}

	return res;
} // need to check x, y ***********************

/*
void SavePointCloudPLY(string fileName, string path, vector<Point3d> object3Dpoints, vector<Point3d> objectRGB) {

	cout << "check rgb and 3d point size : " << objectRGB.size() << ", " << object3Dpoints.size() << endl;
    int featuresNum = object3Dpoints.size();
	//int camNum = cam3Dpts.size();

	string saveFile = path+fileName;
	const char* f = saveFile.c_str();
    FILE *out=fopen(f,"wb");
    fprintf(out,"ply\n");
    fprintf(out,"format ascii 1.0\n");
    fprintf(out,"element vertex %d\n",featuresNum);
    fprintf(out,"property float x\n");
    fprintf(out,"property float y\n");
    fprintf(out,"property float z\n");
    fprintf(out,"property uchar diffuse_blue\n");
    fprintf(out,"property uchar diffuse_green\n");
    fprintf(out,"property uchar diffuse_red\n");
    fprintf(out,"end_header\n");

	vector<Point3d> tempRGB;
	if (objectRGB.size() == 0){
		for (int i=0; i<featuresNum; i++){
			Point3d one = {255, 255, 255};
			tempRGB.push_back(one);
		}

		objectRGB = tempRGB;
	}

    for(int i=0; i<featuresNum; i++){
        fprintf(out, "%lf %lf %lf %d %d %d\n", object3Dpoints[i].x, object3Dpoints[i].y, object3Dpoints[i].z, int(objectRGB[i].x), int(objectRGB[i].y), int(objectRGB[i].z));
    }

    fclose(out);

    cout << "save 3D points complete." << endl;
}
*/

// camerapoints is in object3Dpoints
void SavePointCloudPLY(string fileName, string path, vector<Point3d> object3Dpoints, vector<Point3d> objectRGB, vector<Point3d> cam3Dpts, vector<Point3d> camRGB) {

	cout << "check rgb and 3d point size : " << objectRGB.size() << ", " << object3Dpoints.size() << endl;
    int featuresNum = object3Dpoints.size();
	int camNum = cam3Dpts.size();

	string saveFile = path+fileName;
	const char* f = saveFile.c_str();
    FILE *out=fopen(f,"wb");
    fprintf(out,"ply\n");
    fprintf(out,"format ascii 1.0\n");
    fprintf(out,"element vertex %d\n",featuresNum+camNum);
    fprintf(out,"property float x\n");
    fprintf(out,"property float y\n");
    fprintf(out,"property float z\n");
    fprintf(out,"property uchar diffuse_blue\n");
    fprintf(out,"property uchar diffuse_green\n");
    fprintf(out,"property uchar diffuse_red\n");
    fprintf(out,"end_header\n");

	vector<Point3d> tempRGB;
	if (objectRGB.size() == 0){
		for (int i=0; i<featuresNum; i++){
			Point3d one = {255, 255, 255};
			tempRGB.push_back(one);
		}

		objectRGB = tempRGB;
	}

    for(int i=0; i<featuresNum; i++){
        fprintf(out, "%lf %lf %lf %d %d %d\n", object3Dpoints[i].x, object3Dpoints[i].y, object3Dpoints[i].z, int(objectRGB[i].x), int(objectRGB[i].y), int(objectRGB[i].z));
    }

	for(int i=0; i<camNum; i++){
		fprintf(out, "%lf %lf %lf %d %d %d\n", cam3Dpts[i].x, cam3Dpts[i].y, cam3Dpts[i].z, int(camRGB[i].x), int(camRGB[i].y), int(camRGB[i].z));
	}

    fclose(out);

    cout << saveFile << " save 3D points complete.(object points : " << featuresNum << ", camNum : " << camNum << ")." << endl;
}


Mat diagMatrix(Mat& mat1){ // M=N
	int M = mat1.rows;
	int N = mat1.cols;
	Mat temp = Mat::zeros(M, M, CV_64F);

	for (int i=0; i<M; i++){
		for (int j=0; j<N; j++){
			if (i==j){
				temp.row(i).col(i) = mat1.row(i).col(i);
			}
		}
	}
	return temp;
}

Mat multiplyElement(Mat& mat1, Mat& mat2){
	int M = mat1.rows;
	int N = mat1.cols;
	Mat temp = Mat::zeros(M, N, CV_64F);

	for (int i=0; i<M; i++){
		for (int j=0; j<N; j++){
			temp.row(i).col(j) = mat1.row(i).col(j) * mat2.row(i).col(j);
		}
	}

	return temp;
}

Mat doubleMatrixElement(Mat& mat1, int row, int col){
// -1 : non
// 0 ~ n : specific row, col
    int M = mat1.rows; // 3
	int N = mat1.cols; 
	
	if (row!=-1){
		Mat res = Mat::zeros(1,N, CV_64F);
		for(int i=0; i<N; i++){
			res.row(0).col(i) = mat1.row(row).col(i)*mat1.row(row).col(i);
		}
		return res;
	}

	if (col!=-1){
		Mat res = Mat::zeros(M,1, CV_64F);
        for(int i=0; i<M; i++){
			res.col(0).row(i) = mat1.col(col).row(i)*mat1.col(col).row(i);
			
		}
		return res;
	}
}

Mat divideMatrixElement(Mat& mat1, Mat& mat2){
	int M = mat1.rows;
	int N = mat1.cols;
	Mat res = Mat::zeros(M,N,CV_64F);

	for (int i=0; i<M; i++){
		for (int j=0; j<N; j++){
			res.row(i).col(j) = mat1.at<float>(i,j)/mat2.at<float>(i,j);
		}
	}

	return res;
}

void splitMatrix(Mat& mat, int startrow, Mat& dst){
	Mat temp = Mat::eye(3,3,CV_64F);
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			temp.row(i).col(j) = mat.row(i+startrow).col(j);
		}
	}

	dst = temp;
}

// ** print type ** // 
//vector <Point2d> p(1);
//p[0] = Point2d(0.3, 0);
//cout << typeid(p[0].x).name() << endl;



// matrix 
// Mat::zeros(N, M)
// size -> M, N 
// row = N, col = M;
// resize( Size(2,1)) -> [1,0]





/*
Mat diagMatrix(Mat& mat1){ // M=N
	int M = mat1.rows;
	int N = mat1.cols;
	Mat temp = Mat::zeros(M, M, CV_64F);

	for (int i=0; i<M; i++){
		for (int j=0; j<N; j++){
			if (i==j){
				temp.row(i).col(i) = mat1.row(i).col(i);
			}
		}
	}
	return temp;
}

Mat doubleMatrixElement(Mat& mat1, int row, int col){
// -1 : non
// 1 ~ n : specific row, col
    int M = mat1.rows;
	int N = mat1.cols;


    
	if (row!=-1){
		for(int i=0; i<N; i++){
			Mat res = Mat::zeros(M,N, CV_64F);
			vector<float> e1;
			mat1.row(row).col(i).copyTo(e1);
            res.row(row).col(i) = pow(e1[0],2);
		}
	}

	if (col!=-1){
        for(int i=0; i<M; i++){
			Mat res = Mat::zeros(M,N, CV_64F);
			vector<float> e1;
			mat1.col(col).row(i).copyTo(e1);
			res.col(col).row(i) = pow(e1[0],2);
		}
	}

	return res;
}

Mat divideMatrixElement(Mat& mat1, Mat& mat2){

}

void removeNonMatch(vector<DMatch> matches, vector<KeyPoint> f1, vector<KeyPoint> f2){
    vector<Point2f> Point2df1, Point2df2;
	KeyPoint::convert(f1, Point2df1);
	KeyPoint::convert(f2, Point2df2);

	//queryIdx, int _trainIdx

	vector<Point2f> matchf1, matchf2;

	int num = matches.size();
	for (int i=0; i<num; i++){
		int f1Num = matches[i].queryIdx;
		int f2Num = matches[i].trainIdx;

		
	}
}

void removeOutlier(vector<Point2f>& points1, vector<Point2f>& points2, vector<DMatch>& matches, Mat& FundamentalMat){
	int num=matches.size();
    
	vector<Point2f> newPoints1, newPoints2;

	vector<Point3d> homoPoints1, homoPoints2;
	
	for(vector<Point2f>::iterator it=points1.begin(); it!=points1.end(); ++it){
		Point3d homoPoint1((*it).x, (*it).y, 1.f);
		homoPoints1.push_back(homoPoint1);
	}

	for(vector<Point2f>::iterator it=points2.begin(); it!=points2.end(); ++it){
		Point3d homoPoint2((*it).x, (*it).y, 1.f);
		homoPoints2.push_back(homoPoint2);
	}

	Mat matHomoPoints1 = Mat(homoPoints1);
	Mat matHomoPoints2 = Mat(homoPoints2);

	Mat transFundamental;
	transposeMatrix(FundamentalMat, transFundamental);

	Mat X2Line, X1Line;
	X2Line = multiplyMatrix(FundamentalMat, matHomoPoints1);
	X1Line = multiplyMatrix(transFundamental, matHomoPoints2);

    Mat transX2;
	transposeMatrix(matHomoPoints2, transX2);

	Mat Ck = multiplyMatrix(transX2, X2Line);
	Mat diagCk = diagMatrix(Ck);
	Mat CkEpi = multiplyMatrix(diagCk, diagCk);

	Mat X2Line0RD = doubleMatrixElement(X2Line, 0, -1);
	Mat X2Line1RD = doubleMatrixElement(X2Line, 1, -1);
	Mat X1Line0RD = doubleMatrixElement(X1Line, -1, 0);
	Mat X1Line1RD = doubleMatrixElement(X1Line, -1, 1);
	Mat denom = X2Line0RD+X2Line1RD +X1Line0RD+X1Line1RD;


}
*/