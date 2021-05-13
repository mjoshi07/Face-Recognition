#include "Face.h"

Face::Face(std::string _dataPath, bool _detectLandmarks, bool _recognizeFaces)
{
	mDataPath = _dataPath;
	mDetectLandmarks = _detectLandmarks;
	mRecognizeFaces = _recognizeFaces;
	
	initializeValues();
}

Face::~Face()
{
}

void Face::initializeValues()
{
	
	mLoadFaceModels = std::make_unique<LoadFaceModel>(mDataPath,mDetectFaces, mDetectLandmarks, mRecognizeFaces);

	cv::dnn::Net detectionModel = mLoadFaceModels->getDetectionModel();
	cv::dnn::Net landmarksModel = mLoadFaceModels->getLandmarksModel();
	cv::dnn::Net embeddingsModel = mLoadFaceModels->getEmbeddingsModel();
	
	mFaceDNN = std::make_unique<FaceDNN>(detectionModel, embeddingsModel, landmarksModel);

	cv::String imgsPath = mDataPath + "\\faceImages";
	scanDB(imgsPath);

	double mMatchingThreshold = 0.6;
}

void Face::scanDB(cv::String & imgsPath)
{
	//Scans the database, detect faces in images, extracts the feature vectors and then stores them into memory for recognition
	std::vector<std::string> individualFilePaths;
	cv::glob( imgsPath, individualFilePaths, false);

	if (individualFilePaths.size())
	{
		int i = 0;
		for (auto& filePath : individualFilePaths)
		{
			cv::Mat img = cv::imread(filePath);

			std::string tempPath = filePath;
			const size_t last_slash_idx = tempPath.find_last_of("\\/");
			if (std::string::npos != last_slash_idx)
			{
				tempPath.erase(0, last_slash_idx + 1);
			}

			const size_t period_idx = tempPath.rfind('.');
			if (std::string::npos != period_idx)
			{
				tempPath.erase(period_idx);
			}

			mFaceDNN->getFeatures(img, mDBFaceDetails);

			if (mDBFaceDetails.size()) { mDBFaceDetails[i].faceID = tempPath; }
			i++;
		}
	}
}

void Face::performMatching(cv::Mat &frame, bool drawFaceAndFaceID)
{
	if (mFaceDetails.size())
	{
		for (auto& face : mFaceDetails)
		{
			face.faceID = getFaceId(face.embeddingMat);

			cv::rectangle(frame, face.faceRect, cv::Scalar(180, 255, 50), 2, 16);
			cv::putText(frame, face.faceID, cv::Point(face.faceRect.x, face.faceRect.y - 5), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(180, 255, 50), 1, 16);

		}
	}
}

std::string Face::getFaceId(cv::Mat& embeddingMat)
{
	std::string faceId = "unknown";

	int maxMatchIdx(-1);
	double maxMatchConf(-1);
	for (int i = 0; i < mDBFaceDetails.size(); i++)
	{
		cv::Mat dbFaceEmbeddingMat = mDBFaceDetails[i].embeddingMat;
		double dotProduct = embeddingMat.dot(dbFaceEmbeddingMat);
		if (dotProduct > maxMatchConf && dotProduct > mMatchingThreshold)
		{
			maxMatchConf = dotProduct;
			maxMatchIdx = i;
		}
	}
	if(maxMatchIdx != -1)
	{
		faceId = mDBFaceDetails[maxMatchIdx].faceID;
	}

	return faceId;
}

void Face::runFaceRecognition(cv::Mat & frame)
{
	mFaceDetails.clear();
	mFaceDNN->getFeatures(frame, mFaceDetails);

	performMatching(frame, true);
}