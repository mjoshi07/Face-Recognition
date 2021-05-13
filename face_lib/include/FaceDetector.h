#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

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

class FaceDetector{
	public:
		FaceDetector(cv::dnn::Net& model,double confidence=0.3, std::vector<std::string> only_classes_to_detect = {}, const int frame_to_skip = 2);
		~FaceDetector();

		void getDetectedRects(cv::Mat& img, std::vector<FaceDetails>& faces, unsigned long  frame_number=1);

	private:
		void warmUp();
		std::vector<detected_object> detect(cv::Mat& img);

	private:
		cv::dnn::Net mNet;
		cv::Size mNetInputSize;
		double mConfidenceThreshold;
		double mScaleFactor;
		cv::Scalar mMeanToSubtract;
		bool mCrop{false};
		bool mSwapRB{false};
		std::vector<std::string> mClassNames;
		std::vector<std::string> mOnlyClassesToDetect;
		int mFrameToSkip;
		std::vector<dlib::correlation_tracker> mDlibTrackerList;




};

#endif
