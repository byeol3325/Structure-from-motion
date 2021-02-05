// opencv version 4.5.0
#include "SFM.h"
//#include "imagePair.h"


int main(int argc, char** argv) {
	cout << "start" << endl;   

	cout << "============================================================================================" << endl;

    // argv : output_name, data_path, data_name00000, data_number
	cout << "have " << argc << " arguments : " << endl;

	for (int i=0; i<argc; i++){
		cout << "argv[" << i << "] : " << argv[i] << endl;
	}

	SFM sfm;
	int imageNum = 3;
	string path = "./testData/"; // option : "./data/" "./testData/"


	sfm.setdirPATH(imageNum,path);

	bool success_map = sfm.map3D();
	if (not success_map){
		cout << "Could not obtain 3D mapping" << endl;
		return -1;
	}


	//sfm.drawEpilineIn2views(7,8); // draw epiline for test
	//sfm.drawEpilineIn2views(6,8);

	vector<Point3d> pts3f;
	for (int i=0; i<sfm.recon3Dpts.size(); i++){
		pts3f.push_back(sfm.recon3Dpts[i].pt);
	}

	sfm.getCamPosition();

	string plyName = "Moon_pst3D" + to_string(imageNum) + "view.ply";
	SavePointCloudPLY(plyName, sfm.getdirPATH(), pts3f, sfm.RGB, sfm.cam3Dpts, sfm.cam3DRGB);


/*
	Segmentation sg;
	sg.color_based_growing_segmentation();

	
	Dendrometry tree;
	tree.estimate(sf.cloudPCL);
	cout << "finish all" << endl;
	*/

    return 0;
    
}

