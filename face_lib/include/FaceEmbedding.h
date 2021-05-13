#ifndef FACE_EMBEDDING_H
#define FACE_EMBEDDING_H

#include "Face.h"


class FaceEmbedding {
	public:
		FaceEmbedding(cv::dnn::Net& model);
		~FaceEmbedding();

		void getEmbeddedFeatures(std::vector<Face>& faces);

	private:
		void warmUp();

	private:
		cv::dnn::Net mNet;
		cv::Size mNetInputSize;
		double mNmsThreshold;
		double mScaleFactor;
		cv::Scalar mMeanToSubtract;
		bool mCrop{};
		bool mSwapRB{};
		std::vector<cv::String> mOutNames;

		int mFrameToSkip;
		std::string mOutLayerType;

};

#endif
