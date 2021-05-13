#include "FaceLandmark.h"

FaceLandmark::FaceLandmark(cv::dnn::Net& model,bool draw_landmarks)
{
	mNet = model;
	mNetInputSize = cv::Size(48, 48);
	mScaleFactor = 1.0;
	mMeanToSubtract = cv::Scalar();
	mCrop = false;
	mSwapRB = false;

	mDrawLandMarks = draw_landmarks;
}

FaceLandmark::~FaceLandmark()
{
}

void FaceLandmark::getFaceLandmarks(cv::Mat & img, std::vector<FaceDetails>& faces)
{
	if (faces.size())
	{
		for (auto& faceObject : faces)
		{
			cv::Rect face = faceObject.faceRect;
			cv::Mat faceImg = faceObject.faceImg;
			if (!faceImg.empty())
			{
				int frameHeight = faceImg.rows;
				int frameWidth = faceImg.cols;

				cv::Mat inputBlob = cv::dnn::blobFromImage(faceImg, mScaleFactor, mNetInputSize, mMeanToSubtract, mSwapRB, mCrop);

				mNet.setInput(inputBlob);
				cv::Mat out = mNet.forward();

				cv::Mat row1 = out.row(0);
				long out_size = row1.total();
				float* data = (float*)row1.data;

				for (int j = 0; j < 5; j++)
				{
					float x = data[2 * j] * frameWidth;
					float y = data[2 * j + 1] * frameHeight;
					x += face.x;
					y += face.y;
					cv::Point p = cv::Point(x, y);

					if (mDrawLandMarks)
					{
						cv::circle(img, p, 5, cv::Scalar(120, 0, 255), -1, 16);
					}
					faceObject.faceLandmarks.push_back(p);
				}
			}
		}
	}
}


/*
warm up the network with a test image
*/
void FaceLandmark::warmUp()
{
	cv::Mat test_img = cv::Mat(mNetInputSize, CV_8UC3, cv::Scalar(0));
	FaceDetails faceObject;
	faceObject.faceRect = cv::Rect(1, 1, mNetInputSize.width - 1, mNetInputSize.height - 1);
	faceObject.faceImg = test_img(faceObject.faceRect);
	std::vector<FaceDetails> faces = { faceObject };
	getFaceLandmarks(test_img, faces);
}
