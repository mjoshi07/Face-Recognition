#ifndef CENTROID_TRACKER_H
#define CENTROID_TRACKER_H

#include "Track.h"
#include "HungarianAlg.h"


class CentroidTracker 
{
	public:
		CentroidTracker(const unsigned int _dist_thresh = 50, const unsigned int _maxskipped = 25, const int historySize=20);
		~CentroidTracker();

		void Update(std::vector<cv::Rect> rects);
		void Update(std::vector<cv::Point2d> &detectedCenterPoints, std::vector<cv::Rect> rects);
		void Draw(cv::Mat &frame, bool drawTrack=true, cv::Scalar& color=cv::Scalar(0,255,0));

		std::vector<TrackPtr>& getTracks() { return mTracks; }
		void clear();
		TrackPtr getTrack(int id) {
			for (auto &t : mTracks)
			{
				if (t->getId() == id)
				{
					return t;
				}
			}
			return NULL;
		}

	

	protected:

		unsigned int mDistanceThreshold;      
		unsigned int mMaxSkippedFrames;  
		int mHistorySize;
		unsigned long mNextTrackId;
		std::vector<TrackPtr> mTracks;
};


#endif
