#include "FaceEmbedding.h"

FaceEmbedding::FaceEmbedding(cv::dnn::Net& model)
{
	mNet = model;
	mNetInputSize = cv::Size(96, 112);
	mScaleFactor = 1.0;
	mMeanToSubtract = cv::Scalar(127.5,127.5,127.5);
	mCrop = false;
	mSwapRB = false;
}

FaceEmbedding::~FaceEmbedding()
{
}

void FaceEmbedding::getEmbeddedFeatures(std::vector<FaceDetails>& faces)
{
	if (faces.size())
	{
		for (auto& faceObject : faces)
		{
			cv::Mat faceImg = faceObject.faceImg;
			
			int frameHeight = faceImg.rows;
			int frameWidth = faceImg.cols;

			cv::Mat inputBlob = cv::dnn::blobFromImage(faceImg, mScaleFactor, mNetInputSize, mMeanToSubtract, mSwapRB, mCrop);

			mNet.setInput(inputBlob);
			cv::Mat out = mNet.forward();

			cv::Mat scores = out.row(0);
			long out_size = scores.total();
			double* data = (double*)scores.data;
			std::vector<double> embedding(data, data + out_size);
			faceObject.faceEmbeddings = embedding;
			cv::Mat tempMat = cv::Mat(1, embedding.size(), CV_64F);
			std::memcpy(tempMat.data, embedding.data(), embedding.size() * sizeof(double));
			faceObject.embeddingMat = tempMat;
			faceObject.selfDotProduct = tempMat.dot(tempMat);
		}
	}
}


/*
warm up the network with a test image
*/
void FaceEmbedding::warmUp()
{
	cv::Mat test_img = cv::Mat(mNetInputSize, CV_8UC3, cv::Scalar(0));
	FaceDetails faceObject;
	faceObject.faceRect = cv::Rect(1, 1, mNetInputSize.width - 1, mNetInputSize.height - 1);
	faceObject.faceImg = test_img(faceObject.faceRect);
	std::vector<FaceDetails> faces = { faceObject };
	getEmbeddedFeatures(faces);
}
