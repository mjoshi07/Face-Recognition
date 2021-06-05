#include "LoadModel.h"


LoadFaceModel::LoadFaceModel(std::string data_path, bool _loadDetectionModel, bool _loadLandmarksModel, bool _loadEmbeddingsModel)
{
	mDataPath = data_path;
	mLoadDetectionModel = _loadDetectionModel;
	mLoadLandmarksModel = _loadLandmarksModel;
	mLoadEmbeddingsModel = _loadEmbeddingsModel;

	if (mLoadDetectionModel) { loadDetectionModel(); }
	if (mLoadLandmarksModel) { loadLandmarksModel(); }
	if (mLoadEmbeddingsModel) { loadEmbeddingsModel(); }

}

LoadFaceModel::~LoadFaceModel()
{
}

void LoadFaceModel::loadDetectionModel()
{
	std::string model_weights = mDataPath + "\\models\\face-detection-retail-0005.bin";
	std::string model_config = mDataPath + "\\models\\face-detection-retail-0005.xml";

	// read the weights and config file 
	mDetectionModel = cv::dnn::readNet(model_weights, model_config);

	// set backend as INFERENCE_ENGINE
	mDetectionModel.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_INFERENCE_ENGINE);
}

void LoadFaceModel::loadLandmarksModel()
{
	std::string model_weights = mDataPath + "\\models\\landmarks-regression-retail-0009.bin";
	std::string model_config = mDataPath + "\\models\\landmarks-regression-retail-0009.xml";

	// read the weights and config file 
	mLandmarksModel = cv::dnn::readNet(model_weights, model_config);

	// set backend as INFERENCE_ENGINE
	mLandmarksModel.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_INFERENCE_ENGINE);
}

void LoadFaceModel::loadEmbeddingsModel()
{
	std::string model_weights = mDataPath + "\\models\\Sphereface.bin";
	std::string model_config = mDataPath + "\\models\\Sphereface.xml";

	// read the weights and config file 
	mEmbeddingsModel = cv::dnn::readNet(model_weights, model_config);

	// set backend as INFERENCE_ENGINE
	mEmbeddingsModel.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_INFERENCE_ENGINE);
}

cv::dnn::Net LoadFaceModel::getDetectionModel()
{
	return mDetectionModel;
}

cv::dnn::Net LoadFaceModel::getLandmarksModel()
{
	return mLandmarksModel;
}

cv::dnn::Net LoadFaceModel::getEmbeddingsModel()
{
	return mEmbeddingsModel;
}


