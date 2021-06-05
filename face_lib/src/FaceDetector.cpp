#include "FaceDetector.h"

FaceDetector::FaceDetector(cv::dnn::Net& model,double confidence, std::vector<std::string> only_classes_to_detect, const int frame_to_skip)
{
	mNet = model;
	mOnlyClassesToDetect = only_classes_to_detect;
	mFrameToSkip = frame_to_skip;
	mConfidenceThreshold = confidence;
	mNetInputSize = cv::Size(300, 300);
	mDlibTrackerList = {};
	mScaleFactor = 1.0;
	mMeanToSubtract = cv::Scalar(104.0, 177.0, 123.0);
	mCrop = false;
	mSwapRB = false;
	mClassNames = {"no-face", "face"};

	warmUp();
}

FaceDetector::~FaceDetector()
{
}

void FaceDetector::getDetectedRects(cv::Mat & img, std::vector<FaceDetails>& faces, unsigned long  frame_number)
{
	std::vector<detected_object> detected_objects = {};
	
	if (!img.empty())
	{
		dlib::matrix<dlib::rgb_pixel> dlib_img;
		assign_image(dlib_img, dlib::cv_image<dlib::bgr_pixel>(img));

		if (frame_number % mFrameToSkip == 0)
		{
			//run detector and get rects
			mDlibTrackerList.clear();
			detected_objects = detect(img);

			for (auto& obj : detected_objects)
			{
				std::string obj_class = obj.getClass();
				FaceDetails faceObject;
				if (obj_class != "")
				{
					cv::Rect obj_rect = obj.getRect();

					// create dlib correlation tracker object
					dlib::correlation_tracker corr_tracker;

					// initialize correlation tracker 
					corr_tracker.start_track(dlib_img, dlib::centered_rect(dlib::point(obj_rect.x + obj_rect.width*0.5, obj_rect.y + obj_rect.height*0.5), obj_rect.width, obj_rect.height));
					
					// store the correlation tracker in dlib tracker list
					mDlibTrackerList.push_back(corr_tracker);
					faceObject.faceRect = obj_rect;
					faceObject.faceImg = img(obj_rect);
					faces.push_back(faceObject);
				}
			}
		}
		else
		{
			//get rects from dlib tracker
			for (dlib::correlation_tracker &corr_track : mDlibTrackerList)
			{
				FaceDetails faceObject;

				// update the object rect by correlation tracker 
				corr_track.update(dlib_img);

				// get object rect in dlib rectangle format
				dlib::rectangle rect = corr_track.get_position();

				int width = corr_track.get_position().width();
				int height = corr_track.get_position().height();
				int x = corr_track.get_position().right() - width;
				int y = corr_track.get_position().bottom() - height;

				// get object rect in opencv rectangle format
				cv::Rect rec(x, y, width, height);
				faceObject.faceRect = rec;
				faceObject.faceImg = img(rec);
				faces.push_back(faceObject);
			}
		}
	}
}

/*
warm up the network with a test image
*/
void FaceDetector::warmUp()
{
	cv::Mat test_img = cv::Mat(mNetInputSize, CV_8UC3, cv::Scalar(0));
	detect(test_img);
}

std::vector<detected_object> FaceDetector::detect(cv::Mat & img)
{
	std::vector<detected_object> detections = {};
	if (!img.empty())
	{
		int frameHeight = img.rows;
		int frameWidth = img.cols;

		// get input blob from image
		cv::Mat inputBlob = cv::dnn::blobFromImage(img, mScaleFactor, mNetInputSize, mMeanToSubtract, mSwapRB, mCrop);

		mNet.setInput(inputBlob);

		// run a forward pass
		cv::Mat out = mNet.forward();

		cv::Mat detectionMat(out.size[2], out.size[3], CV_32F, out.ptr<float>());

		for (int i = 0; i < detectionMat.rows; i++)
		{
			/*
				output is of the shape [1,1,N, 7]
				[image_id, label, conf, x_min, y_min, x_max, _y_max]
			*/
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
