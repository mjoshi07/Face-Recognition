#ifndef LOAD_MODEL_H
#define LOAD_MODEL_H

#include <opencv2/opencv.hpp>

class LoadFaceModel
{
	public:
		LoadFaceModel(){}
		LoadFaceModel(std::string data_path, bool loadDetectionModel=true, bool loadLandmarksModel=false, bool loadEmbeddingsModel=false);
		~LoadFaceModel();
		
		cv::dnn::Net* getDetectionModel();
		cv::dnn::Net* getLandmarksModel();
		cv::dnn::Net* getEmbeddingsModel();

	private:
		void loadDetectionModel();
		void loadLandmarksModel();
		void loadEmbeddingsModel();

	private:
		std::string mDataPath;
		cv::dnn::Net mDetectionModel;
		cv::dnn::Net mLandmarksModel;
		cv::dnn::Net mEmbeddingsModel;
		bool mLoadDetectionModel;
		bool mLoadLandmarksModel;
		bool mLoadEmbeddingsModel;
};

#endif
