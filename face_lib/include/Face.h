#ifndef FACE_H
#define FACE_H

#include "LoadModel.h"
#include "FaceDNN.h"

class Face
{
	public:
		//methods
		Face(std::string _dataPath, bool _detectLandmarks=false, bool _recognizeFaces=true);
		~Face();

		void runFaceRecognition(cv::Mat& frame);

	private:
		//methods
		void initializeValues();
		void scanDB(cv::String& imgsPath);
		void performMatching(cv::Mat& frame, bool drawFaceAndFaceID=true);
		std::string getFaceId(cv::Mat& embeddingMat);

	private:
		//members
		std::string mDataPath;
		bool mDetectFaces;
		bool mDetectLandmarks;
		bool mRecognizeFaces;
		std::unique_ptr<LoadFaceModel> mLoadFaceModels;
		std::unique_ptr<FaceDNN> mFaceDNN;
		std::vector<FaceDetails> mFaceDetails;
		std::vector<FaceDetails> mDBFaceDetails;
		double mMatchingThreshold;

};


#endif
