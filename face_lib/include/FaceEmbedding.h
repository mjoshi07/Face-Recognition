#ifndef FACE_EMBEDDING_H
#define FACE_EMBEDDING_H

#include "FaceDetails.h"

class FaceEmbedding {
	public:
		FaceEmbedding(cv::dnn::Net& model);
		~FaceEmbedding();

		void getEmbeddedFeatures(std::vector<FaceDetails>& faces);

	private:
		void warmUp();

	private:
		cv::dnn::Net mNet;
		cv::Size mNetInputSize;
		double mScaleFactor;
		cv::Scalar mMeanToSubtract;
		bool mCrop{};
		bool mSwapRB{};
};

#endif
