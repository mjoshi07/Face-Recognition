#ifndef TRACK_H
#define TRACK_H

#include <opencv2/opencv.hpp>

class Track {
	const int min = 0, max = 255;

	public:
		Track(size_t trackID, const cv::Point2d& initialPosition, const cv::Rect& intialBoundingRect, int _maxSkippedFrames,unsigned int historySize = 50):
		mId(trackID),
		mConsecutiveInvisibleCount(0),
		mCurrentCenter(initialPosition),
		lastRect(intialBoundingRect),
		mTotalVisibleCount(1),
		age(1),
		mStartPosition(initialPosition),
		mHistorySize(historySize),
		mMaxSkippedFrames(_maxSkippedFrames)
		{
			trackDead = false;
		}

		float CalcDist(const cv::Point2d& p) const
		{
			cv::Point2d diff = mCurrentCenter - p;
			return sqrtf(diff.x * diff.x + diff.y * diff.y);
		}

		virtual void Update(const cv::Point2d &p, const cv::Rect &rect, bool dataCorrect)
		{
			if (dataCorrect) // update point only when datacorrect is true
			{
				mCurrentCenter = p;
				lastRect = rect;
				mConsecutiveInvisibleCount = 0;
				mTotalVisibleCount++;
			}
			else 
			{
				/// update invisible, visible count based on whether measurement was provided
				mConsecutiveInvisibleCount++;
			}

			if (trace.size() > mHistorySize) {
				trace.erase(trace.begin(), trace.end() - mHistorySize);
			}

			trace.push_back(mCurrentCenter);

			if ( mConsecutiveInvisibleCount > mMaxSkippedFrames)
			{
				trackDead = true;
			}				
		}

		virtual void Draw(cv::Mat &image, bool drawTrack = true, cv::Scalar& color = cv::Scalar(0,255,0)) const {

				cv::rectangle(image, GetLastRect(), color, 2, 16);
				if (drawTrack)
				{
					for (int i = 1; i < trace.size(); i++)
					{
						cv::line(image, cv::Point(trace[i - 1]), cv::Point(trace[i]), color);
					}

					int baseline(0);
					std::string label = "ID:" + std::to_string(mId);
					
					cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.8, 1, &baseline);
					cv::Point bottomLeftPointRect = cv::Point(GetLastRect().x + 4, GetLastRect().y + GetLastRect().height - 4);
					cv::rectangle(image, bottomLeftPointRect, cv::Point(bottomLeftPointRect.x + textSize.width, bottomLeftPointRect.y - textSize.height), cv::Scalar::all(0), -1, 16);
					cv::putText(image, label, bottomLeftPointRect, cv::FONT_HERSHEY_DUPLEX, 0.8, color, 1, cv::LINE_AA, false);
				}
		}

		cv::Rect GetLastRect() const
		{
			auto r =  cv::Rect(
				static_cast<int>(mCurrentCenter.x - lastRect.width / 2),
				static_cast<int>(mCurrentCenter.y - lastRect.height / 2),
				lastRect.width -1,
				lastRect.height -1);
			if (r.x < 0 || r.y < 0) // boundary cases where x and y can be negative
			{
				return lastRect;
			}
			return r;
		}


		size_t getId() const { return mId; }

		bool operator< (const Track &another) const {
			return lastRect.area() < another.lastRect.area();
		}

		std::vector<cv::Point2f> getTraces()
		{
			return trace;
		}

		unsigned int getInvisibleCount() const { return mConsecutiveInvisibleCount; }
		unsigned long getAge() const { return age; }
		void IncrementAge() { age++; }
		void setTraceHistory(int traceLength) { mHistorySize = traceLength; }
		bool isTrackDead() { return trackDead; }

	protected:
		cv::Point2d mCurrentCenter;                ///< current prediction or center in case of centroid tracking
		cv::Rect lastRect;                  ///< last bounding box of the object

		std::vector<cv::Point2f> trace{};       ///< trajectory
		size_t mId;                         ///< id of track
		unsigned int mConsecutiveInvisibleCount;   ///< consecutive frames for which the track has not been detected
		unsigned int mTotalVisibleCount;           ///< the total number of frames in which the track was detected(visible)
		unsigned long  age;                 ///< the number of frames since the track was first detected

		cv::Point mStartPosition;           ///< initial position where the track started
		unsigned int mBirthThreshold;       ///< number of frames after which to consider track normal
		unsigned int mHistorySize;          ///< number of previous measurements to keep
		unsigned int mMaxSkippedFrames;         ///< if object is invisible for these number of frames then it is considered dead
		bool trackDead;

	};

	typedef std::shared_ptr<Track> TrackPtr;


#endif
