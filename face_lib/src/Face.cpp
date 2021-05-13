#include "Face.h"

Face::Face(std::string _dataPath, bool _detectFaces, bool _detectLandmarks, bool _recognizeFaces)
{
	mDataPath = _dataPath;
	mDetectFaces = _detectFaces;
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
	cv::dnn::Net embeddingsModel =mLoadFaceModels->getEmbeddingsModel();
	
	const double conf = 0.4;
	const int frame_to_skip = 5;
	std::vector<std::string> only_class_to_detect = { "face" };
	mFaceDetector = std::make_unique<FaceDetector>(detectionModel, conf, only_class_to_detect, frame_to_skip);

	mFaceLandmarksDetector = std::make_unique<FaceLandmark>(landmarksModel);

	mFaceEmbedder = std::make_unique<FaceEmbedding>(embeddingsModel);
	
	const int distance_threshold = 50;
	const int max_skipped_frames = 25;
	const int history_size = 60;
	mTracker = std::make_unique<CentroidTracker>(distance_threshold, max_skipped_frames, history_size);

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

			mFaceDetector->getDetectedRects(img, mDBFaceDetails);
			mFaceEmbedder->getEmbeddedFeatures(mDBFaceDetails);

			if (mDBFaceDetails.size()) { mDBFaceDetails[i].faceID = tempPath; }

			i++;
		}
	}
}

void Face::performMatching()
{
	if (mFaceDetails.size())
	{
		for (auto& face : mFaceDetails)
		{
			face.faceID = getFaceId(face.embeddingMat);
		}
	}
}

std::string Face::getFaceId(cv::Mat& embeddingMat)
{
	std::string faceId = "unknown";

	int minMatchIdx(-1);
	double minMatch(1000000000.0);
	for (int i = 0; i < mDBFaceDetails.size(); i++)
	{
		cv::Mat dbFaceEmbeddingMat = mDBFaceDetails[i].embeddingMat;
		double dotProduct = embeddingMat.dot(dbFaceEmbeddingMat);

		if (dotProduct < minMatch)
		{
			minMatch = dotProduct;
			minMatchIdx = i;
		}
	}
	if(minMatchIdx != -1)
	{
		faceId = mDBFaceDetails[minMatchIdx].faceID;
	}

	return faceId;
}

void Face::runFaceRecognition(cv::Mat & frame, unsigned long frame_number)
{
	mFaceDetails.clear();
	mFaceDetector->getDetectedRects(frame, mFaceDetails, frame_number);
	mFaceEmbedder->getEmbeddedFeatures(mFaceDetails);

	performMatching();
	
	if (mFaceDetails.size())
	{
		for (auto& face : mFaceDetails)
		{
			cv::rectangle(frame, face.faceRect, cv::Scalar(255, 0, 120), 2, 16);
			cv::putText(frame, face.faceID, face.faceRect.tl(), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 0, 120), 1, 16);
		}
	}

}