#ifndef FACE_DETAILS_H
#define FACE_DETAILS_H

#include <opencv2\opencv.hpp>

struct FaceDetails
{
	cv::Rect faceRect;
	cv::Mat faceImg;
	std::vector<cv::Point> faceLandmarks;
	std::vector<double> faceEmbeddings;
	cv::Mat embeddingMat;
	double selfDotProduct;
	
	std::string faceID;
	double matchingConfidence;

	bool readEmbeddings{ false };

	
	/*
		 TODO  - align the face, classify as male/female, detect age and emotion

	bool matched;
	std::string matchedId;
	double matchedConfidence;
	bool isFaceAligned;
	bool isMale;
	int age;
	std::string emotion;
	
	*/
};

#endif
