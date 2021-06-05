#include "Face.h"

Face::Face(std::string _dataPath, bool _detectFaces, bool _detectLandmarks, bool _recognizeFaces)
{
	mDataPath = _dataPath;
	mDetectFaces = _detectFaces;
	mDetectLandmarks = _detectLandmarks;
	mRecognizeFaces = _recognizeFaces;
	
	// initialize the data members values and DNN models
	initializeValues();
}

Face::~Face()
{
}

void Face::initializeValues()
{
	// create loadModel class object and initialize it
	mLoadFaceModels = std::make_unique<LoadFaceModel>(mDataPath,mDetectFaces, mDetectLandmarks, mRecognizeFaces);

	// get face detection model
	cv::dnn::Net detectionModel = mLoadFaceModels->getDetectionModel();

	// get face landmarks detection model
	cv::dnn::Net landmarksModel = mLoadFaceModels->getLandmarksModel();

	// get face embedding vector model
	cv::dnn::Net embeddingsModel =mLoadFaceModels->getEmbeddingsModel();
	
	const double conf = 0.4;
	const int frame_to_skip = 5;
	std::vector<std::string> only_class_to_detect = { "face" };

	// initialize Face Detector object
	mFaceDetector = std::make_unique<FaceDetector>(detectionModel, conf, only_class_to_detect, frame_to_skip);

	// initialize Face Landmarks detector object
	mFaceLandmarksDetector = std::make_unique<FaceLandmark>(landmarksModel);

	// initialize Face Embedding object
	mFaceEmbedder = std::make_unique<FaceEmbedding>(embeddingsModel);
	
	const int distance_threshold = 50;
	const int max_skipped_frames = 25;
	const int history_size = 60;

	// initialize Centroid Tracker Object
	mTracker = std::make_unique<CentroidTracker>(distance_threshold, max_skipped_frames, history_size);

	// images path from the data folder
	cv::String imgsPath = mDataPath + "\\faceImages";

	// scan the imgs directory for faces and store them in memory
	scanDB(imgsPath);

	// define the matching threshold for a face match
	mMatchingThreshold = 0.6;
	
}

void Face::scanDB(cv::String & imgsPath)
{
	/*
	1. read all images
	2. detect face in each image
	3. extract 512 feature vector from each face
	4. assign the image name to the 512 feature vector as its FACE ID
	*/

	std::vector<std::string> individualFilePaths;

	// get individual image paths
	cv::glob( imgsPath, individualFilePaths, false);

	if (individualFilePaths.size())
	{
		int i = 0;
		for (auto& filePath : individualFilePaths)
		{
			// read the image
			cv::Mat img = cv::imread(filePath);

			std::string tempPath = filePath;

			// get the file name from the full image file path
			const size_t last_slash_idx = tempPath.find_last_of("\\/");
			if (std::string::npos != last_slash_idx)
			{
				tempPath.erase(0, last_slash_idx + 1);
			}

			// remove the extension and dot(.) from the image name
			const size_t period_idx = tempPath.rfind('.');
			if (std::string::npos != period_idx)
			{
				tempPath.erase(period_idx);
			}

			// detect faces and store them in database face details vector
			mFaceDetector->getDetectedRects(img, mDBFaceDetails);

			// extract feature vectors and store them in database face details vector
			mFaceEmbedder->getEmbeddedFeatures(mDBFaceDetails);


			if (mDBFaceDetails.size()) 
			{ 
				// assign the image file name to the detected face in the image
				mDBFaceDetails[i].faceID = tempPath; 
			}
			i++;
		}
	}
}

void Face::performMatching()
{
	// start matching faces IFF faces detected in current frame
	if (mFaceDetails.size())
	{
		for (auto& face : mFaceDetails)
		{
			// get Face ID for each detected face in the current frame
			face.faceID = getFaceId(face.embeddingMat, face.selfDotProduct);
		}
	}
}

std::string Face::getFaceId(cv::Mat& embeddingMat, double embeddingMatSelfDotProduct)
{
	// FACE ID for an unmatched/new face
	std::string faceId = "unknown";

	int maxMatchIdx(-1);
	double maxMatchConf(-1);

	// loop through all the faces in database 
	for (int i = 0; i < mDBFaceDetails.size(); i++)
	{
		cv::Mat dbFaceEmbeddingMat = mDBFaceDetails[i].embeddingMat;

		// dot product of current face and db face
		double dotProduct = embeddingMat.dot(dbFaceEmbeddingMat);

		// self dot product of db face
		double dbFaceSelfDotProduct = mDBFaceDetails[i].selfDotProduct;

		// get cosine distance [matching value] between current face and db face
		double cosineDistance = getCosineDistance(embeddingMatSelfDotProduct, dbFaceSelfDotProduct, dotProduct);

		if (cosineDistance > maxMatchConf && cosineDistance > mMatchingThreshold)
		{
			// get the db face with highest cosine distance, i.e best match face
			maxMatchConf = cosineDistance;
			maxMatchIdx = i;
		}
	}
	if(maxMatchIdx != -1)
	{
		// assign the FACE ID of db face with best match to the current face
		faceId = mDBFaceDetails[maxMatchIdx].faceID;
	}

	return faceId;
}

double Face::getCosineDistance(double aa, double bb, double ab)
{
	// formula to calculate the cosine distance between 2 vectors
	return 	ab / (std::sqrtl(aa)*std::sqrtl(bb));
}

void Face::drawFaces(cv::Mat& frame, cv::Scalar color)
{
	// draw faces IFF face is detected in the current frame
	if (mFaceDetails.size())
	{
		for (auto& face : mFaceDetails)
		{
			// draw the bounding box around the face
			cv::rectangle(frame, face.faceRect, color, 2, 16);

			// write the FACE ID on TOP-LEFT side of bounding box
			cv::putText(frame, face.faceID, cv::Point(face.faceRect.x, face.faceRect.y - 5), cv::FONT_HERSHEY_COMPLEX, 0.6, color, 1, 16);
		}
	}
}

void Face::runFaceRecognition(cv::Mat & frame, unsigned long frame_number)
{
	// clear the face details vector for the current frame
	mFaceDetails.clear();

	// detect faces in current frame and store them in face details vector
	mFaceDetector->getDetectedRects(frame, mFaceDetails, frame_number);

	// extract feature vector for the detected faces in current frame and store them in face details vector
	mFaceEmbedder->getEmbeddedFeatures(mFaceDetails);

	// start matching the database faces with the detected faces in current frame
	performMatching();

	// draw bbox and write FACE ID on the detected faces in current frame
	drawFaces(frame);
}
