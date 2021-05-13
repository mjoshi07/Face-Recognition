#ifndef FACE_LANDMARK_H
#define FACE_LANDMARK_H

#include "FaceDetails.h"


class FaceLandmark {
	public:
		FaceLandmark(cv::dnn::Net& model, bool draw_landmarks=true);
		~FaceLandmark();

		void getFaceLandmarks(cv::Mat& img, std::vector<FaceDetails>& faces);

	private:
		void warmUp();

	private:
		bool mDrawLandMarks{ false };
		cv::dnn::Net mNet;
		cv::Size mNetInputSize;
		double mScaleFactor;
		cv::Scalar mMeanToSubtract;
		bool mCrop{};
		bool mSwapRB{};
		std::vector<cv::String> mOutNames;

		int mFrameToSkip;
		std::string mOutLayerType;

};

#endif
