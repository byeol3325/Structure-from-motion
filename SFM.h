#ifndef SFM_H_
#define SFM_H_

//#include "DendrometryE.h"
#include "my_utils.h"
#include "bundle.h"
//using namespace pcl;

struct Point3D{
    Point3d pt;
    map<int, int> idxImage;
    map<int, Point2d> pt2D;
};

class SFM{
private:
    vector<Mat> MatColorImages;
    vector<Mat> MatGrayImages;
    vector<Mat> nImages;
    vector<Matx34d> nCameraPose;
    vector<Mat> camPositions;

    string dirPATH;
    int Num;
    int world;
    vector<string> pathImages;

    map<int, pair<int, int>, greater<int>> orderPair;
    set<int> DoneViews;
    set<int> GoodViews;
    set<int> SkipViews;

    Mat instrinsic;
    vector<float> lensDistortion;
    vector<vector<KeyPoint>> imagesKeypoints;
    vector<Mat> imagesDescriptors;
    vector<vector<Point2d>> imagesPts2D;


public:
    vector<Point3D> recon3Dpts;
    vector<Point3d> RGB;

    vector<Point3d> cam3Dpts;
    vector<Point3d> cam3DRGB;

    SFM();
    void setdirPATH(int N, string dir);
    string getdirPATH(){return dirPATH;};
    
    void loadAndGetfeatures(); // load data (images, features...)
    void getMatching(int i1, int i2, vector<DMatch>& gopodMatches); // matching and remove outlier

    void findBestPair(); // find pair that has max matching pts
    bool baseRecon(); //  baseRecon (in best pair)
    void removeOutlier(vector<DMatch>& good_matches, vector<Point2d>& f1, vector<Point2d>& f2, Mat& fundamental); // remove outlier
    bool getCameraPose(const int& i1, const int& i2, vector<DMatch>& matches, vector<Point2d>& pts1, vector<Point2d>& pts2, Matx34d& P1, Matx34d& P2); // get camera pose
    void prundMatchingHomography(const int& i1, const int& i2, vector<DMatch>& matches, vector<DMatch>& prunedMatch); // 
    void alignedPointsFromMatch(vector<Point2d>& f1, vector<Point2d>& f2, vector<DMatch> matches, vector<Point2d>& aligned1, vector<Point2d>& aligned2); // get f1, f2 from matches
    void alignedPoints(vector<Point2d>& f1, vector<Point2d>& f2, vector<DMatch> matches, vector<Point2d>& aligned1, vector<Point2d>& aligned2, vector<int>& Id1, vector<int>& Id2); // get f1, f2 from matches
    void matchingImagePair(const int& query_idx, const int& train_dix); // i1, i2 image matching
    void drawEpilineIn2views(int img1, int img2); // draw epiline within two images
    bool triangulateViews(vector<Point2d>& f1, vector<Point2d>& f2, Matx34d& P1, Matx34d& P2, vector<DMatch> matches, pair<int, int> imagePair, vector<Point3D>& pointcloud, vector<Point2d>& aligned1forcolor);
    bool addMoreViews(); // add more views in triangulating
    void mergeNewPoints(vector<Point3D>& newPointCloud, int imageNum, vector<Point2d>& aligned1forColor); // add more 3d pts in triangulating
    void find2D3DMatches(int& NEWVIEW, vector<Point3d>& points3D, vector<Point2d>& points2D, vector<DMatch>& bestMatches, int& doneView); // find 2d, 3d points for solvePNP (P3P)
    bool findCameraPosePNP(vector<Point3d>& pts3D, vector<Point2d>& pts2D, Matx34d& P, int NEWVIEW, int DONEVIEW); // find camera pose from pnp
    bool CheckCoherentR(Mat& R); // check R 
    void getColor(int imageNum, Point2d pts2d); // get RGB values 
    void getCamPosition(); // get camera position and save (world is red)
    void adjustCurrentBundle(); // for bundle
    bool map3D(); // 3D mapping process for sfm 
};

#endif