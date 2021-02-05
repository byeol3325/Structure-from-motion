#include "SFM.h"

Matx<float, 3, 3> intrin0(1698.873755, 0.000000, 971.7497705, 0.000000, 1698.8796645, 647.7488275, 0.000000, 0.000000, 1.000000); // intrinsic matrix
Mat intrin(intrin0);
vector<float> distortion = {0.0, 0.0, 0.0, 0.0, 0.0}; // lens distortion

Matx<float, 3, 3> intrinp(5301.333, 0.000000, 3926, 0.000000, 5301.333, 2552, 0.000000, 0.000000, 1.000000); // intrinsic matrix
Mat intrin_pro(intrinp);
//vector<float> distortion = {0.0, 0.0, 0.0, 0.0, 0.0}; // lens distortion

SFM::SFM(){

}

void SFM::setdirPATH(int N, string dir){
    this->dirPATH = dir;
    this->Num = N;

    vector<string> fileN = read_directory(dir);
    vector<string> temp;

    for (int i=0; i<N; i++){
        string imagePath = fileN[i];
        temp.push_back(imagePath);
        cout << imagePath << " file is added" << endl;
    }
    this->pathImages = temp;

    cout << "============================ set dir complete ============================" << endl;
    cout << "\n";
}


void SFM::loadAndGetfeatures(){
    vector<vector<KeyPoint>> tempKpoints;
    vector<Mat> tempDesc;
    vector<vector<Point2d>> temp2Dpts;
    vector<Matx34d> tempCampose;
    vector<Mat> tempMatColorImages;
    vector<Mat> tempMatGrayImages;

    if (this->dirPATH == "./data/"){
        this->instrinsic = intrin_pro;
    } else if (this->dirPATH == "./testData/")
    {
        this->instrinsic = intrin;
    }

    this->lensDistortion = distortion;
    
    this->nCameraPose.resize(this->MatGrayImages.size());
    this->imagesKeypoints.resize(this->MatGrayImages.size(), vector<KeyPoint>());
    this->imagesDescriptors.resize(this->MatGrayImages.size(), Mat());
    this->imagesPts2D.resize(this->MatGrayImages.size(), vector<Point2d>());

    cout << "intrinsic : " << this->instrinsic << endl;
    cout << "distortion : ";
    for (int i=0; i<this->lensDistortion.size(); i++){
        cout << this->lensDistortion[i] << ", ";
    }

    cout << "\n";
    for (int i=0; i<this->Num; i++){
        Mat colorImage = imread(this->dirPATH + this->pathImages[i], IMREAD_COLOR);
        Mat grayImage = imread(this->dirPATH + this->pathImages[i], IMREAD_GRAYSCALE);

        tempMatColorImages.push_back(colorImage);
        tempMatGrayImages.push_back(grayImage);
        Ptr<Feature2D> sift = SIFT::create(); // if you want to add variant
	    vector<KeyPoint> f1;
        Mat d1;
	    Mat mask;
	    sift->detectAndCompute(grayImage, mask, f1, d1);

        cout << this->pathImages[i] << " has " << f1.size() << " size features (SIFT)" << endl;

        tempKpoints.push_back(f1);
        tempDesc.push_back(d1);

        vector<Point2f> pts2Df;
        KeyPoint::convert(f1, pts2Df);
        
        vector<Point2d> pts2D;
        for (int i=0; i<pts2Df.size(); i++){
            pts2D.push_back((Point2d)pts2Df[i]);
        }

        temp2Dpts.push_back(pts2D);
        tempCampose.push_back(Matx34d::eye());
    }
    this->MatColorImages = tempMatColorImages;
    this->MatGrayImages = tempMatGrayImages;
    this->imagesKeypoints = tempKpoints;
    this->imagesDescriptors = tempDesc;
    this->imagesPts2D = temp2Dpts;
    this->nCameraPose = tempCampose;


    cout << "============================ load and get features complete ============================" << endl;
    cout << "\n";
}

void SFM::getMatching(int i1, int i2, vector<DMatch>& goodMatches){
    BFMatcher* matcher = new BFMatcher(NORM_L2, false);
    vector<vector<DMatch>> knnMatches;

    Mat desc1 = this->imagesDescriptors[i1];
    Mat desc2 = this->imagesDescriptors[i2];

    matcher->knnMatch(desc1, desc2, knnMatches, 2);

    const float ratio_thresh = 0.55f;
    for(size_t i=0; i<knnMatches.size(); i++){
        if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance){
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    cout << i1 << ", " << i2 << " is getMatching." << endl;
}


void SFM::findBestPair(){
    map<int, pair<int, int>, greater<int>> numInlier;

    for (int i1=0; i1<this->Num; i1++){
        for (int i2=i1+1; i2<this->Num; i2++){
            vector<DMatch> good_matches;
            getMatching(i1, i2, good_matches);
            vector<KeyPoint> good_f1, good_f2; // good match features in all
            vector<KeyPoint> f1 = this->imagesKeypoints[i1];
            vector<KeyPoint> f2 = this->imagesKeypoints[i2];
            for (vector<DMatch>::iterator it=good_matches.begin(); it!=good_matches.end(); ++it){
                good_f1.push_back(f1[(*it).queryIdx]);
                good_f2.push_back(f2[(*it).trainIdx]);
            }

            vector<Point2f> source2f, target2f;
            KeyPoint::convert(good_f1, source2f);
            KeyPoint::convert(good_f2, target2f);

            vector<Point2d> sourcePoints, targetPoints; //////////////
            for (int i=0;i<source2f.size();i++){
                sourcePoints.push_back((Point2d)source2f[i]);
                targetPoints.push_back((Point2d)target2f[i]);
            }

            vector<uchar> inliers(good_matches.size(),0);

            if (good_matches.size() < 80){
                continue;
            }
            Mat fundamental_matrix = findFundamentalMat(sourcePoints, targetPoints, inliers, FM_7POINT, 1, 0.99);


            /*
            if (fundamental_matrix.rows ==0){
                cout << "what ??? " << endl;
            }
            if (fundamental_matrix.rows !=3){
                int inlierPtsNum = 0;
                numInlier[inlierPtsNum] = make_pair(i1, i2);
                continue;
                
                int max = 0;
                Mat tempF;
                for (int m=0; m<fundamental_matrix.rows/3; m++){
                    Mat oneF;
                    vector<DMatch> temp_matches = good_matches;
                    vector<Point2d> tempf1 = sourcePoints;
                    vector<Point2d> tempf2 = targetPoints;
                    splitMatrix(fundamental_matrix, 3*m, oneF);
                    cout << "oneF.size : " << oneF.size() << endl;
                    removeOutlier(temp_matches, tempf1, tempf2, oneF);
                    if (temp_matches.size()>max){
                        tempF = oneF;
                    }
                }
                cout << "max : " << max << endl;
                fundamental_matrix = tempF;
            }
            */

            Mat intrinsic = Mat(this->instrinsic);
            Mat Essential_matrix(3,3,CV_64F, Scalar(0));
            //cout << instrinsic.size() << endl;
            cout << "fundamental size : " << fundamental_matrix.rows << ", " << fundamental_matrix.cols << endl;
            sfm::essentialFromFundamental(fundamental_matrix, intrinsic, intrinsic, Essential_matrix);

            //cout << i1 << ", " << i2 << " has "<< good_matches.size() << " match inlier (before remove outlier)" << endl;
            //removeOutlier(good_matches, sourcePoints, targetPoints, fundamental_matrix);
            //cout << i1 << ", " << i2 << " has "<< good_matches.size() << " match inlier (after remove outlier)" << endl;

            int inlierPtsNum = good_matches.size();

            numInlier[inlierPtsNum] = make_pair(i1, i2);
        }
    }

    this->orderPair = numInlier;

    cout << "============================ find pair map complete ============================" << endl;
    cout << "\n";
}


void SFM::removeOutlier(vector<DMatch>& good_matches, vector<Point2d>& f1, vector<Point2d>& f2, Mat& fundamental){
	int num=good_matches.size();
    
	vector<Point2d> newPoints1, newPoints2;

	vector<Point3d> homoPoints1, homoPoints2;

    vector<Point2d> points1 = f1;
    vector<Point2d> points2 = f2;

    Mat FundamentalMat = fundamental;
	
	for(vector<Point2d>::iterator it=points1.begin(); it!=points1.end(); ++it){
		Point3d homoPoint1((*it).x, (*it).y, 1);
		homoPoints1.push_back(homoPoint1);
	}

	for(vector<Point2d>::iterator it=points2.begin(); it!=points2.end(); ++it){
		Point3d homoPoint2((*it).x, (*it).y, 1);
		homoPoints2.push_back(homoPoint2);
	}
    Mat matHomoPoints1(3, homoPoints1.size(), CV_64F);
    for (size_t i=0, end=homoPoints1.size(); i<end; ++i){
        matHomoPoints1.at<double>(0,i) = homoPoints1[i].x;
        matHomoPoints1.at<double>(1,i) = homoPoints1[i].y;
        matHomoPoints1.at<double>(2,i) = homoPoints1[i].z;
    }
    Mat matHomoPoints2(3, homoPoints2.size(), CV_64F);
    for (size_t i=0, end=homoPoints2.size(); i<end; ++i){
        matHomoPoints2.at<double>(0,i) = homoPoints2[i].x;
        matHomoPoints2.at<double>(1,i) = homoPoints2[i].y;
        matHomoPoints2.at<double>(2,i) = homoPoints2[i].z;
    }

    Mat X2Line = FundamentalMat*matHomoPoints1;
    Mat X1Line = FundamentalMat.t()*matHomoPoints2;
    Mat Ck = matHomoPoints2.t()*X2Line;
	Mat diagCk = diagMatrix(Ck);
	Mat CkEpi = multiplyElement(diagCk, diagCk);
    Mat numer = CkEpi.t();


    Mat X2Line0R2 = doubleMatrixElement(X2Line, 0, -1);
    Mat X2Line1R2 = doubleMatrixElement(X2Line, 1, -1);
    Mat X1Line0R2 = doubleMatrixElement(X1Line, 0, -1);
    Mat X1Line1R2 = doubleMatrixElement(X1Line, 1, -1);
    Mat denom = X2Line0R2+X2Line1R2+X1Line0R2+X1Line1R2;
    Mat Cost = divideMatrixElement(numer, denom);
    int check_N = Cost.cols;
    
    vector<int> inlierNumbers;

    for(int i=0; i<check_N; i++){
        double checkOne = Cost.at<double>(0, i);
        if (checkOne < 1.5){ // threshold 1.5 pixel
            inlierNumbers.push_back(i);
        }
    }
    vector<Point2d> inf1, inf2;
    vector<DMatch> in_good_matches;

    for(int i=0; i<inlierNumbers.size(); i++){
        inf1.push_back(f1[inlierNumbers[i]]);
        inf2.push_back(f2[inlierNumbers[i]]);
        in_good_matches.push_back(good_matches[inlierNumbers[i]]);
    }

    f1 = inf1;
    f2 = inf2;
    good_matches = in_good_matches;

}

/*
for (int i=0; i<Num; i++){
		float x = points1[i].x;
		float y = points1[i].y;

		vector<Point3d> RGB;
		image1.row(y).col(x).copyTo(RGB);
		res.push_back(RGB[0]);
	}
*/

void SFM::getColor(int imageNum, Point2d pts2d){
    double x = pts2d.x;
    double y = pts2d.y;

    vector<Point3d> oneRGB;
    this->MatColorImages[imageNum].row(y).col(x).copyTo(oneRGB);
    this->RGB.push_back(oneRGB[0]);
}

bool SFM::baseRecon(){
    map<int, pair<int, int>, greater<int>> bestViews = this->orderPair;
    cout << " ================== start base recon ================== " << endl;
    cout << " best views size : " << bestViews.size() << endl;

    int max = 0;
    for (auto it=bestViews.begin(); it!=bestViews.end(); it++){
        if ((*it).first > max){
            max = (*it).first;
        }
    }

    cout << "max pair number : " << max << " , pair : " << bestViews[max].first << ", " << bestViews[max].second << endl;
    this->world = bestViews[max].first;
    cout << "The world : " << this->world << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    pair<int, int> bestpair = bestViews[max];
    
    int i1 = bestpair.first;
    int i2 = bestpair.second;

    vector<DMatch> goodMatches;
    getMatching(i1, i2, goodMatches);
    cout << i1 << ", " << i2  << " has " << goodMatches.size() << " matching points" << endl;

    Matx34d P1 = Matx34d::eye();
    Matx34d P2 = Matx34d::eye();

    bool success = this->getCameraPose(i1, i2, goodMatches, this->imagesPts2D[i1], this->imagesPts2D[i2], P1, P2);

    if (!success){
        cout << i1 << ", " << i2 << " cannot matching!!!!!!!!!!!!" << endl;
    }

    vector<Point3D> pointcloud;
    vector<Point2d> aligned1forColor;

    bool successTri = this->triangulateViews(this->imagesPts2D[i1], this->imagesPts2D[i2], P1, P2, goodMatches, pair<int,int>(i1, i2), pointcloud, aligned1forColor);
    cout << "point cloud size : " << pointcloud.size() << endl;

    for (int i=0; i<aligned1forColor.size(); i++){
        this->getColor(i1, aligned1forColor[i]);
    }

    cout << "RGB size : " << this->RGB.size() << endl;

    if(!successTri){
        cout << i1 << ", " << i2 << " cannot do triangulate image!!!!!!!!!!!!!!!!!!!!" << endl;
    }

    this->recon3Dpts = pointcloud;
    cout << "P1 : " << P1 << endl;
    cout << "P2 : " << P2 << endl;
    this->nCameraPose[i1] = P1;
    this->nCameraPose[i2] = P2;

    this->DoneViews.insert(i1);
    this->DoneViews.insert(i2);
    //cout << "here " << endl;

    this->GoodViews.insert(i1);
    this->GoodViews.insert(i2);
    cout << i1 << ", " << i2 << " has recon !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << endl;

    cout << "============================ base reconstruction complete ============================" << endl;
    cout << "\n";
    //this->adjustCurrentBundle();
    return true;
}

bool SFM::triangulateViews(vector<Point2d>& f1, vector<Point2d>& f2, Matx34d& P1, Matx34d& P2, vector<DMatch> matches, pair<int, int> imagePair, vector<Point3D>& pointcloud, vector<Point2d>& aligned1forcolor){
    pointcloud.clear();

    vector<Point2d> aligned1, aligned2;
    vector<int> Refer1, Refer2;
    this->alignedPoints(f1, f2, matches, aligned1, aligned2, Refer1, Refer2);

    cout <<"size compare : " << f1.size() << ", " << aligned1.size() << ", " << Refer1.size() << endl;

    Mat normalizedPts1, normalizedPts2;
    undistortPoints(aligned1, normalizedPts1, this->instrinsic, this->lensDistortion);
    undistortPoints(aligned2, normalizedPts2, this->instrinsic, this->lensDistortion);

    Mat pts3dHomo;
    triangulatePoints(P1, P2, normalizedPts1, normalizedPts2, pts3dHomo);

    Mat pts3d;
    convertPointsFromHomogeneous(pts3dHomo.t(), pts3d);

    Mat rvec1;
    Rodrigues(P1.get_minor<3,3>(0,0), rvec1);
    Mat tvec1(P1.get_minor<3,1>(0,3).t());

    vector<Point2d> projection1(aligned1.size());
    projectPoints(pts3d, rvec1, tvec1, this->instrinsic, this->lensDistortion, projection1);

    Mat rvec2;
    Rodrigues(P2.get_minor<3,3>(0,0), rvec2);
    Mat tvec2(P2.get_minor<3,1>(0,3).t());

    vector<Point2d> projection2(aligned2.size());
    projectPoints(pts3d, rvec2, tvec2, this->instrinsic, this->lensDistortion, projection2);

    vector<Point2d> newaligned1forColor;
    const float MIN_REPROJECTION_ERROR = 6.0;
    for (int i=0;i<pts3d.rows;i++){
        const double p1Error = norm(projection1[i]-aligned1[i]);
        const double p2Error = norm(projection2[i]-aligned2[i]);

        if (MIN_REPROJECTION_ERROR<p1Error or MIN_REPROJECTION_ERROR<p2Error){
            continue;
        }

        Point3D p;
        p.pt = Point3d(pts3d.at<double>(i,0), pts3d.at<double>(i,1), pts3d.at<double>(i,2));

        p.idxImage[imagePair.first] = Refer1[i];
        p.idxImage[imagePair.second] = Refer2[i];

        p.pt2D[imagePair.first] = this->imagesPts2D.at(imagePair.first).at(Refer1[i]);
        p.pt2D[imagePair.second] = this->imagesPts2D.at(imagePair.second).at(Refer2[i]);

        pointcloud.push_back(p);
        newaligned1forColor.push_back(aligned1[i]);
        //this->getColor(imagePair.first, aligned1[i]);
    }

    aligned1forcolor = newaligned1forColor;

    cout << "============================" << imagePair.first << ", " << imagePair.second << " triangulate complete ============================" << endl;
    cout << "\n";
    return true;
}

bool SFM::getCameraPose(const int& i1, const int& i2, vector<DMatch>& matches, vector<Point2d>& pts1, vector<Point2d>& pts2, Matx34d& P1, Matx34d& P2){
    vector<DMatch> prunedMatch;
    this->prundMatchingHomography(i1, i2, matches, prunedMatch);

    vector<Point2d> aligned1, aligned2;
    this->alignedPointsFromMatch(pts1, pts2, matches, aligned1, aligned2);

    Mat mask;
    Mat intrinsic = this->instrinsic;
    Mat E = findEssentialMat(aligned1, aligned2, intrinsic, RANSAC, 0.999, 1.0, mask);

    Mat R, T;
    float fx = intrinsic.at<float>(0,0);
    float cx = intrinsic.at<float>(0,2);
    float cy = intrinsic.at<float>(1,2);
    Point2d pp = Point2d(cx, cy);

    recoverPose(E, aligned1, aligned2, R, T, fx, pp, mask);
    bool success = this->CheckCoherentR(R);

    Matx34d tempP1 = Matx34d::eye();
    Mat tP2;
    hconcat(R,T,tP2);
    Matx34d tempP2 = Matx34d(tP2);

    P1 = tempP1;
    P2 = tempP2;

    cout << i1 << ", " << i2 << " get Camera pose complete" <<endl;
    return success;
}

bool SFM::CheckCoherentR(Mat& R){
    if (fabs(determinant(R)-1.0>1e-07)){
        cout << "det(R) is not +-1.0!!!!!!!!!!!!!!!!!!" << endl;
        return false;
    }
    return true;
}

void SFM::alignedPointsFromMatch(vector<Point2d>& f1, vector<Point2d>& f2, vector<DMatch> matches, vector<Point2d>& aligned1, vector<Point2d>& aligned2){
    vector<int> Id1, Id2;
    this->alignedPoints(f1, f2, matches, aligned1, aligned2, Id1, Id2);
}


void SFM::alignedPoints(vector<Point2d>& f1, vector<Point2d>& f2, vector<DMatch> matches, vector<Point2d>& aligned1, vector<Point2d>& aligned2, vector<int>& Id1, vector<int>& Id2){
    for (int i=0; i<matches.size(); i++){
        aligned1.push_back(f1[matches[i].queryIdx]);
        aligned2.push_back(f2[matches[i].trainIdx]);

        Id1.push_back(matches[i].queryIdx);
        Id2.push_back(matches[i].trainIdx);
    }
}


void SFM::prundMatchingHomography(const int& i1, const int& i2, vector<DMatch>& matches, vector<DMatch>& prunedMatch){
    vector<KeyPoint> matched1, matched2;
    vector<KeyPoint> kf1 = this->imagesKeypoints[i1];
    vector<KeyPoint> kf2 = this->imagesKeypoints[i2];
    cout << "before kf1, kf2 : " << kf1.size() << ", " << kf2.size() << endl;

    for(unsigned i=0; i<matches.size(); i++){
        matched1.push_back(kf1[matches[i].queryIdx]);
        matched2.push_back(kf2[matches[i].trainIdx]);
    }

    cout << "after kf1, kf2 : " << matched1.size() << ", " << matched2.size() << endl;

    vector<Point2f> f1_2f, f2_2f;
    KeyPoint::convert(matched1, f1_2f);
    KeyPoint::convert(matched2, f2_2f);

    vector<Point2d> f1, f2;
    for (int i=0;i<f1_2f.size();i++){
        f1.push_back((Point2d)f1_2f[i]);
        f2.push_back((Point2d)f2_2f[i]);
    }

    const double ransac_thresh = 2.5;

    if (f1 == f2){
        cout << "f1 and f2 is same" << endl;
    }
    cout << "all matching features number is " << f1.size() << ", " << f2.size() << endl;
    Mat inlier_mask, homography;
    homography = findHomography(f1, f2, inlier_mask, RANSAC, ransac_thresh);

    cout << "Homography inlier mask : " << inlier_mask.rows << " inliers " << endl;

    for(int i=0; i<inlier_mask.rows; i++){
        if (inlier_mask.at<uchar>(i)){
            prunedMatch.push_back(matches[i]);
        }
    }
}


bool SFM::addMoreViews(){
    vector<Point3d> points3D;
    vector<Point2d> points2D;

    cout << "we have done " << this->DoneViews.size() << "/" << this->MatGrayImages.size() << " recon " << endl;
    cout << "DoneViews : ";
    for (auto it=this->DoneViews.begin(); it!=this->DoneViews.end(); it++){
        cout << *it << " ";
    }
    cout << "\n" << endl;

    while (this->DoneViews.size() != this->MatGrayImages.size()){
        set<int> newFrames;
        for (int newViewstoAdd:this->DoneViews){
            int i;
            int j;

            if (newViewstoAdd==0){
                i=newViewstoAdd;
                j=abs(newViewstoAdd+1);
            }else if(newViewstoAdd+1==this->Num){
                i=abs(newViewstoAdd-1);
                j=newViewstoAdd;
            }else{
                i = abs(newViewstoAdd-1);
                j= abs(newViewstoAdd+1);
            }

            if(this->DoneViews.count(i)==1){
                if(this->DoneViews.count(j)==1){
                    continue;
                }else{
                    newFrames.insert(j);
                }
            }else {
                newFrames.insert(i);
                if(this->DoneViews.count(j)==1){
                    continue;
                }else{
                    newFrames.insert(j);
                }
            }
        }

        cout << "we will add " << newFrames.size() << " size more " << endl;
        cout << "will add frames : ";
        for (auto it=newFrames.begin(); it!=newFrames.end(); it++){
            cout << *it << " ";
        }
        cout << "\n" << endl;

        for (int NEWVIEW:newFrames){
            if (this->DoneViews.find(NEWVIEW)!=this->DoneViews.end()){
                continue;
            }
            cout << "adding " << NEWVIEW  << " now..........." << endl;
            vector<DMatch> goodMatches;
            int DONEVIEW;
            this->find2D3DMatches(NEWVIEW, points3D, points2D, goodMatches, DONEVIEW);
            cout << "points3D size : " << points3D.size() << ", " << "points2D size: " << points2D.size() << "(" << NEWVIEW << ", "<< DONEVIEW << ")" << endl;

            cout << "Adding " << NEWVIEW << " to existing " << Mat(vector<int>(this->DoneViews.begin(), this->DoneViews.end())).t() << endl;
            this->DoneViews.insert(NEWVIEW);

            cout << "estimating camera pose... " << endl;
            Matx34d newCamPose = Matx34d::eye();
            //bool success = this->findCameraPosePNP(points3D, points2D, newCamPose, DONEVIEW, NEWVIEW); // change
            bool success = this->findCameraPosePNP(points3D, points2D, newCamPose, NEWVIEW, DONEVIEW);

            if (!success){
                cout << "Failed to get a camera pose. skip "<< NEWVIEW << " view!!!!!!!!!!!!!!!!1" << endl;
                this->SkipViews.insert(NEWVIEW);
                continue;
            }
            
            this->nCameraPose[NEWVIEW] = newCamPose; ////////////////////// need to fix for more views

            vector<Point3D> newTriangulated;

            for (int GOODVIEW:this->GoodViews){
                int i1, i2;
                i1 = GOODVIEW;
                i2 = NEWVIEW;
                /*
                if(NEWVIEW < GOODVIEW){
                    i1 = NEWVIEW;
                    i2 = GOODVIEW;
                }else{
                    i1 = GOODVIEW;
                    i2 = NEWVIEW;
                }
                */

                vector<DMatch> matches;
                this->getMatching(i1, i2, matches);
                if (matches.size()==0){
                    cout << "skip " << i1 << ", " << i2 << " view " << endl;
                    continue;
                }
                vector<Point2d> aligned1forColor;
                bool goodTriangulation = this->triangulateViews(this->imagesPts2D[i1], this->imagesPts2D[i2], this->nCameraPose[i1], this->nCameraPose[i2], matches, pair<int,int>(i1, i2), newTriangulated, aligned1forColor);

                if (!goodTriangulation){
                    continue;
                }

                cout << "Before " << NEWVIEW << " triangulation : " << this->recon3Dpts.size() << endl;
                this->mergeNewPoints(newTriangulated, i1, aligned1forColor);
                cout << "After " << NEWVIEW << " triangulation : " << this->recon3Dpts.size() << ", " << this->RGB.size() << endl;
            }

            this->GoodViews.insert(NEWVIEW);
            //this->adjustCurrentBundle();
        }
        continue;
    }
    cout << "\n";
    cout << " =============================================================================================================================== " << endl;
    cout << "Images processed (total: " << this->DoneViews.size() << ", skip : " << this->SkipViews.size() << ")" << endl;
    cout << "Skip views : ";
    for (set<int>::iterator it=this->SkipViews.begin(); it!=this->SkipViews.end(); it++){
        cout << *it << ", ";
    }
    cout <<"\n";
    cout << "PointCloud size = " << this->recon3Dpts.size() << " pts3D" << endl;

    return true;
}




void SFM::mergeNewPoints(vector<Point3D>& newPointCloud, int imageNum, vector<Point2d>& aligned1forColor){
    const float MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE   = 0.01;

    size_t newPoints = 0;
    int countForColor = 0;
    for (const Point3D& p: newPointCloud){
        Point3d newPoint = p.pt;
        bool foundAnyMatchingExstingViews = false;
        bool foundMatching3DPoint = false;

        for(Point3D& existingPoint: this->recon3Dpts){
            if(norm(existingPoint.pt-newPoint)<MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE){
                foundMatching3DPoint = true;
                break;
            }
        }

        if (not foundAnyMatchingExstingViews and not foundMatching3DPoint){
            this->recon3Dpts.push_back(p);
            this->getColor(imageNum, aligned1forColor[countForColor]);
            newPoints++;
        }

        countForColor++;
    }

    cout << "Adding New points count : " << newPoints << endl;
}



void SFM::find2D3DMatches(int& NEWVIEW, vector<Point3d>& points3D, vector<Point2d>& points2D, vector<DMatch>& bestMatches, int& DONEVIEW){
    points3D.clear(); points2D.clear();
    int i1, i2;
    int bestMatchesNum = 0;
    vector<DMatch> bestmatch;

    for(int doneView:this->DoneViews){
        /*
        if(NEWVIEW < doneView){
            i1 = NEWVIEW;
            i2 = doneView;
        }else{
            i1 = doneView;
            i2 = NEWVIEW;
        }
        */
        i1 = doneView;
        i2 = NEWVIEW;

        //i1 = NEWVIEW;
        //i2 = doneView;

        vector<DMatch> matches;
        getMatching(i1, i2, matches);

        int numMatches = matches.size();
        if (numMatches>bestMatchesNum){
            bestMatchesNum = numMatches;
            bestmatch = matches;
            DONEVIEW = doneView;
        }
    }

    cout <<"(in find2D3DMatches) " << DONEVIEW << " is best pair with " << NEWVIEW << endl;
    bestMatches = bestmatch;

    for(const Point3D& cloudPoint:this->recon3Dpts){
        bool found2DPoint = false;
        for (const pair<const int, int>& origViewAndPoint:cloudPoint.idxImage){
            const int originatingViewIdx = origViewAndPoint.first;
            const int orignatingViewFeatureIdx = origViewAndPoint.second;
            //cout << "origViewAndPoint : " << originatingViewIdx << ", " << orignatingViewFeatureIdx << endl;

            if (originatingViewIdx != DONEVIEW)continue;

            for (const DMatch& m: bestmatch){
                //cout << " HHHHHHHHHHHHHHHHHHHHEEEEEEEEEEEEERRRRRRRRRRRRRRREEEEEEEEEEEEEEEE" << endl;
                int matched2DPointInNewView = -1;
                if (m.queryIdx == orignatingViewFeatureIdx){
                    matched2DPointInNewView = m.trainIdx;
                }
                //if (m.trainIdx == orignatingViewFeatureIdx){
                //    matched2DPointInNewView = m.queryIdx;
                //}

                /*
                if (originatingViewIdx < NEWVIEW){
                    if (m.queryIdx == orignatingViewFeatureIdx){
                        matched2DPointInNewView = m.trainIdx;
                    }
                }else{
                    if (m.trainIdx == orignatingViewFeatureIdx){
                        matched2DPointInNewView = m.queryIdx;
                    }
                }
                */

                if (matched2DPointInNewView>=0){
                    const vector<Point2d> newViewFeatures = this->imagesPts2D[NEWVIEW];
                    points2D.push_back(newViewFeatures.at(matched2DPointInNewView));
                    points3D.push_back(cloudPoint.pt);
                    found2DPoint = true;
                    break;
                }
            }

            if (found2DPoint){
                break;
            }
        }
    }

    cout <<" best pair !!!!!" << DONEVIEW << " is best pair with " << NEWVIEW << endl;
}


bool SFM::findCameraPosePNP(vector<Point3d>& pts3D, vector<Point2d>& pts2D, Matx34d& P, int NEWVIEW, int DONEVIEW){
    if (pts3D.size() <= 7 || pts2D.size() <= 7 || pts3D.size() != pts2D.size()){
        cout << "couldn't do CAMEARA PNP!!!!!!!!!!" << endl;
        return false;
    }

    vector<DMatch> good_matches;
    getMatching(DONEVIEW, NEWVIEW, good_matches);
    Matx34d P1 = Matx34d::eye();
    Matx34d P2 = Matx34d::eye();
    bool success = this->getCameraPose(DONEVIEW, NEWVIEW, good_matches, this->imagesPts2D[DONEVIEW], imagesPts2D[NEWVIEW], P1, P2);
    cout << "findCameraPosePNP : " << DONEVIEW << ", " << NEWVIEW << endl;

    Mat rot_(P2.get_minor<3,3>(0,0));
    Mat t_(P2.get_minor<3,1>(0,3));
    Mat rot0(this->nCameraPose[DONEVIEW].get_minor<3,3>(0,0));
    Mat t0(this->nCameraPose[DONEVIEW].get_minor<3,1>(0,3));

    Mat rot = multiplyMatrix(rot0,rot_);
    Mat t = plusMatrix(t0, t_);
    
    Mat rvec, T, rvec0, T0;
    Rodrigues(rot, rvec);
    Rodrigues(rot, rvec0);
    T0 = t;
    T = t;

    cout << "HERE : " << pts3D.size() <<", " << pts2D.size() << ", " << P << endl;
    //Mat rvec, T;
    vector<int> inlier;

    double minVal, maxVal;
    minMaxIdx(pts2D, &minVal, &maxVal);
    solvePnPRansac(pts3D, pts2D, this->instrinsic, this->lensDistortion, rvec, T, true, 1000, 0.006*maxVal, 0.99, inlier, SOLVEPNP_P3P);
    cout << "rvec : " << rvec << endl;
    cout << "T : " << T << endl;

    vector<Point2d> projected3D;
    projectPoints(pts3D, rvec, T, this->instrinsic, this->lensDistortion, projected3D);

    if (inlier.size()==0){
        for (int i=0; i<projected3D.size();i++){
            if (norm(projected3D[i]-pts2D[i]) < 8.0){
                inlier.push_back(i);
            }
        }
    }
    cout << "inlier size : " << inlier.size() << endl;

    if (norm(T)>200.0){
        cout << "camera movement is too big!!!!!!!!!" << endl;
        return false;
    }

    Mat R, R0;
    Rodrigues(rvec, R);
    Rodrigues(rvec0, R0);

    if(!this->CheckCoherentR(R)){
        cout << "rotation is incoherent!!!!!!!!!!!!!" << endl;
        return false;
    }

/*
    Mat tempP;
    hconcat(R0, T0, tempP);
    Matx34d tempP_ = Matx34d(tempP);
    P = tempP_;
*/

    Mat tempP;
    hconcat(R, T, tempP);
    Matx34d tempP_ = Matx34d(tempP);
    P = tempP_;

    return true;
}


bool SFM::map3D(){
    cout << "**********************************************************************************************" << endl;
    cout << "****************************************START 3D MAPPING**************************************" << endl;
    cout << "**********************************************************************************************" << endl;

    // extract features & descriptor //
    this->loadAndGetfeatures();
    this->findBestPair();
    // base reconstruction // 
    for(map<int, pair<int, int>, greater<int>>::iterator it=this->orderPair.begin(); it!=this->orderPair.end(); it++){
        cout << it->second.first << ", " << it->second.second << " pair has " << it->first << " features." << endl;
    }
    bool success_baseRecon = this->baseRecon();
    if (not success_baseRecon){
        cout << "------------------------------ WE CAN'T DO 3D MATPPING -------------------------" << endl;
        return false;
    }

    // add views // 
    bool success_add = this->addMoreViews();
    if (not success_add){
        cout << "----------------------------- WE CAN'T ADD MORE VIEWS --------------------------" << endl;
        return false;
    }


    cout << "******************************************" << endl;
    

    return true;
}


void SFM::drawEpilineIn2views(int img1, int img2){
    Mat epiline1 = this->MatColorImages[img1];
    Mat epiline2 = this->MatColorImages[img2];
    vector<KeyPoint> f1 = this->imagesKeypoints[img1];
    vector<KeyPoint> f2 = this->imagesKeypoints[img2];

    Mat sourceGray = this->MatGrayImages[img1];
    Mat targetGray = this->MatGrayImages[img2];

    vector<DMatch> matches;
    this->getMatching(img1, img2, matches);

    vector<KeyPoint> good_f1, good_f2;

    for (vector<DMatch>::iterator it=matches.begin(); it!=matches.end(); it++){
        good_f1.push_back(f1[(*it).queryIdx]);
        good_f2.push_back(f2[(*it).trainIdx]);
    }

    vector<uchar> inliers(matches.size(),0);

    if (matches.size() < 80){
        cout << "can't draw epipolar line " << endl;

    }
    vector<Point2f> f1pts, f2pts;
    KeyPoint::convert(good_f1, f1pts);
    KeyPoint::convert(good_f2, f2pts);
    Mat fundamental_matrix = findFundamentalMat(f1pts, f2pts, inliers, FM_7POINT, 1, 0.99);
    epipolarLineImage(epiline2, sourceGray, targetGray, f1pts, f2pts, fundamental_matrix, 1);
    epipolarLineImage(epiline1, sourceGray, targetGray, f1pts, f2pts, fundamental_matrix, 2);

    string fileName1, fileName2;
    fileName1 = to_string(img1) + "_" + to_string(img2) + "epiline1.jpg";
    fileName2 = to_string(img1) + "_" + to_string(img2) + "epiline2.jpg";
    //cout << "fileName1 : " << fileName1 << endl;
    //const char* fN1 = fileName1.c_str();
    //const char* fN2 = fileName2.c_str();
    imwrite(fileName1, epiline1);
    cout << fileName1 << " save complete. " << endl;
    imwrite(fileName2, epiline2);
    cout << fileName2 << " save complete. " << endl;
}


void SFM::getCamPosition(){
    for (int i=0; i<this->Num; i++){
        /*
        Mat K_inv = this->instrinsic.inv();
        Matx34d P_ = this->nCameraPose[i];
        Mat P = Mat(P_);

        Mat R_T = multiplyMatrix(K_inv, P);
        Matx34d R_Tx = Matx34d(R_T);
        */
        
        Matx34d R_Tx = this->nCameraPose[i];
        Mat R(R_Tx.get_minor<3,3>(0,0));
        Mat T(R_Tx.get_minor<3,1>(0,3));

        Mat R_trans = R.t();
        Mat pose = -multiplyMatrix(R_trans, T); // 
        //cout << "pose : " << pose << endl;//
        this->camPositions.push_back(pose);
    }
    cout << "get " << this->camPositions.size() << " camposition complete." << endl;


    for (int i=0; i<this->Num;i++){
        Point3d oneCamPose;
        oneCamPose.x = this->camPositions[i].at<double>(0);
        oneCamPose.y = this->camPositions[i].at<double>(1);
        oneCamPose.z = this->camPositions[i].at<double>(2);
        this->cam3Dpts.push_back(oneCamPose);
        Point3d oneRGB;
        if (i==this->world){
            oneRGB.x = 0.;
            oneRGB.y = 0.;
            oneRGB.z = 255.;
            this->cam3DRGB.push_back(oneRGB);
        }else{
            oneRGB.x = 0.;
            oneRGB.y = 255.;
            oneRGB.z = 0.;
            this->cam3DRGB.push_back(oneRGB);
        }
    }
}


/*
void SFM::adjustCurrentBundle(){
    cout << "Bundle adjuster..." << endl;
    BundleAdjustment::adjustBundle(this->recon3Dpts, this->nCameraPose, this->instrinsic, this->lensDistortion ,this->imagesPts2D);
}

*/
