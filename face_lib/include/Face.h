#ifndef FACE_H
#define FACE_H

#include "FaceDetails.h"
#include "FaceDetector.h"
#include "FaceLandmark.h"
#include "FaceEmbedding.h"
#include "CentroidTracker.h"
#include "LoadModel.h"

class Face
{
	public:
		//methods
		Face(std::string _dataPath, bool _detectFaces=true, bool _detectLandmarks=false, bool _recognizeFaces=true);
		~Face();

		void runFaceRecognition(cv::Mat& frame, unsigned long frame_number);

	private:
		//methods
		void initializeValues();
		void scanDB(cv::String& imgsPath);
		void performMatching();
		std::string getFaceId(cv::Mat& embeddingMat);
		void drawFaces(cv::Mat& frame, cv::Scalar bboxColor = cv::Scalar(180, 255, 50));

	private:
		//members
		std::string mDataPath;
		bool mDetectFaces;
		bool mDetectLandmarks;
		bool mRecognizeFaces;
		std::unique_ptr<LoadFaceModel> mLoadFaceModels;
		std::unique_ptr<CentroidTracker> mTracker;
		std::unique_ptr<FaceDetector> mFaceDetector;
		std::unique_ptr<FaceLandmark> mFaceLandmarksDetector;
		std::unique_ptr<FaceEmbedding> mFaceEmbedder;
		std::vector<FaceDetails> mFaceDetails;
		std::vector<FaceDetails> mDBFaceDetails;
		double mMatchingThreshold;

};


#endif
