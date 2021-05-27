#include "FaceDNN.h"

FaceDNN::FaceDNN(cv::dnn::Net& _detection_model, cv::dnn::Net& _embeddings_model, cv::dnn::Net& _landmarks_model)
{
	// Detection Model details;
	mDetectionNet = std::make_unique<cv::dnn::Net>(_detection_model);
	mConfidenceThreshold = 0.4;

	// Embeddings Model details;
	mEmbeddingNet = std::make_unique<cv::dnn::Net>(_embeddings_model);
}

FaceDNN::~FaceDNN()
{
}

void FaceDNN::getFeatures(cv::Mat & img, std::vector<FaceDetails>& faces)
{
	std::vector<detected_object> detected_objects = {};
	
	if (!img.empty())
	{
		//run detector and get rects
		detected_objects = detectFaces(img);

		for (auto& obj : detected_objects)
		{
			std::string obj_class = obj.getClass();
			FaceDetails faceObject;
			if (obj_class != "")
			{
				cv::Rect obj_rect = obj.getRect();

				faceObject.faceRect = obj_rect;
				faceObject.faceImg = img(obj_rect);
				faceObject.embeddingMat = detectEmbeddings(faceObject.faceImg);
				faceObject.dbSelfDotProduct = faceObject.embeddingMat.dot(faceObject.embeddingMat);
				faces.push_back(faceObject);
			}
		}
	}
}

std::vector<detected_object> FaceDNN::detectFaces(cv::Mat & fullImg)
{
	std::vector<detected_object> detections = {};
	if (!fullImg.empty())
	{
		int frameHeight = fullImg.rows;
		int frameWidth = fullImg.cols;

		cv::Mat inputBlob = cv::dnn::blobFromImage(fullImg, 1.0, cv::Size(300,300), cv::Scalar(104.0, 177.0, 123.0), false, false);

		mDetectionNet->setInput(inputBlob);
		cv::Mat out = mDetectionNet->forward();

		cv::Mat detectionMat(out.size[2], out.size[3], CV_32F, out.ptr<float>());

		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);

			if (confidence > mConfidenceThreshold)
			{
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

				cv::Rect bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
				detected_object face(bbox, confidence);
				detections.emplace_back(face);
			}
		}
	}

	return detections;
}

cv::Mat FaceDNN::detectEmbeddings(cv::Mat & faceImg)
{
	int frameHeight = faceImg.rows;
	int frameWidth = faceImg.cols;

	cv::Mat inputBlob = cv::dnn::blobFromImage(faceImg, 1.0, cv::Size(96, 112), cv::Scalar(127.5, 127.5, 127.5), false, false);

	mEmbeddingNet->setInput(inputBlob);
	cv::Mat out = mEmbeddingNet->forward();

	cv::Mat scores = out.row(0);
	long out_size = scores.total();
	double* data = (double*)scores.data;
	std::vector<double> embedding(data, data + out_size);
	cv::Mat embeddingMat = cv::Mat(embedding).reshape(0, embedding.size());
	embeddingMat.convertTo(embeddingMat, CV_64F);
	return embeddingMat;
}
