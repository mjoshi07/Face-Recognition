#ifndef FACE_DNN_H
#define FACE_DNN_H

#include "FaceDetails.h"
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>


struct detected_object {
	detected_object(cv::Rect obj_bbox, double obj_conf, std::string obj_class="face")
	{
		_bbox = obj_bbox;
		_conf = obj_conf;
		_class = obj_class;
	}
private:
	cv::Rect _bbox;
	double _conf;
	std::string _class;

public:
	cv::Rect getRect() const{ return _bbox; }
	double getConfidence() const { return _conf; }
	std::string getClass() const { return _class; }
};

class FaceDNN{
	public:
		FaceDNN(cv::dnn::Net& _detection_model, cv::dnn::Net& _embeddings_model, cv::dnn::Net& _landmarks_model);
		~FaceDNN();

		void getFeatures(cv::Mat& img, std::vector<FaceDetails>& faces);

	private:
		std::vector<detected_object> detectFaces(cv::Mat& full_img);
		cv::Mat detectEmbeddings(cv::Mat& face_img);

	private:
		cv::dnn::Net mDetectionNet;
		cv::dnn::Net mEmbeddingNet;
		cv::dnn::Net mLandmarkNet;
		double mConfidenceThreshold;
};

#endif
