#include "FaceEmbedding.h"

FaceEmbedding::FaceEmbedding(cv::dnn::Net& model)
{
	mNet = model;
	mNetInputSize = cv::Size(96, 112);
	mScaleFactor = 1.0;
	mMeanToSubtract = cv::Scalar(127.5,127.5,127.5);
	mCrop = false;
	mSwapRB = false;

	mOutNames = mNet.getUnconnectedOutLayersNames();
	std::vector<int> outLayers = mNet.getUnconnectedOutLayers();
	mOutLayerType = mNet.getLayer(outLayers[0])->type;
}

FaceEmbedding::~FaceEmbedding()
{
}

void FaceEmbedding::getEmbeddedFeatures(std::vector<Face>& faces)
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
		}
	}
}


/*
warm up the network with a test image
*/
void FaceEmbedding::warmUp()
{
	cv::Mat test_img = cv::Mat(mNetInputSize, CV_8UC3, cv::Scalar(0));
	Face faceObject;
	faceObject.faceRect = cv::Rect(1, 1, mNetInputSize.width - 1, mNetInputSize.height - 1);
	faceObject.faceImg = test_img(faceObject.faceRect);
	std::vector<Face> faces = { faceObject };
	getEmbeddedFeatures(faces);
}
