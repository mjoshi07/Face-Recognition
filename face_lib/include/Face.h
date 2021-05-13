#ifndef FACE_H
#define FACE_H

#include <opencv2\opencv.hpp>

struct Face
{
	cv::Rect faceRect;
	cv::Mat faceImg;
	std::vector<cv::Point> faceLandmarks;
	std::vector<double> faceEmbeddings;


	
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
