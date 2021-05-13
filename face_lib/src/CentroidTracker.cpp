#include "../include/CentroidTracker.h"


CentroidTracker::CentroidTracker(unsigned int _dist_thresh, unsigned int _maxskipped, int _historySize) 
{
	mDistanceThreshold = _dist_thresh;
	mMaxSkippedFrames = _maxskipped;
	mHistorySize = _historySize;
	mNextTrackId = 0;
	mTracks.clear();
}


CentroidTracker::~CentroidTracker()
{
	mTracks.clear();
}


void CentroidTracker::Update(std::vector<cv::Rect> rects)
{
	std::vector<cv::Point2d> centroids;

	for (size_t i = 0; i < rects.size(); i++)
	{
		cv::Rect r = rects.at(i);
		double cX = (r.x + (r.x + r.width)) / 2.0;
		double cY = (r.y + (r.y + r.height)) / 2.0;

		centroids.push_back(cv::Point2d(cX, cY));
	}

	Update(centroids, rects);
}


/*
* Update the tracker with new blob information Determine whether tracks are new or belong to objects already on the scene
* @param centerpoints centre of gravity of new blobs
* @param rects bounding rectangles of new blobs
*/
void CentroidTracker::Update(std::vector<cv::Point2d> & centerpoints, std::vector<cv::Rect> rects)
{
	/// If there are no tracks yet, every blob creates it's own new track
	if (mTracks.empty() && !centerpoints.empty())
	{
		for (size_t i = 0; i < centerpoints.size(); ++i)
		{
			TrackPtr tr = std::make_shared<Track>(mNextTrackId++, centerpoints[i], rects[i],mMaxSkippedFrames, mHistorySize);
			mTracks.push_back(tr);
		}
	}

	/// Increase the age of all the tracks that were previously detected
	for (auto &mTrack : mTracks) {
		mTrack->IncrementAge();
	}

	/// Set up Hungarian algorithm to assign tracks with vehicles
	/// set up cost matrix:
	auto N = static_cast<int>(mTracks.size());  ///< number of existing tracks
	int M;
	if (!centerpoints.empty())
	{
		M = static_cast<int>(centerpoints.size());
	}
	else
	{
		M = static_cast<int>(mTracks.size());
	}

	distMatrix_t cost(N * M);
	std::vector<int> assignment;

	/// Assignment problem only needs to be run if blob was detected:
	if (!centerpoints.empty())
	{
		int position = 0;
		for (int i = 0; i < N; ++i) /// Fill the cost matrix with distances between predictions and newly detected centers
		{
			for (int j = 0; j < M; ++j)
			{
				position = i + j * N;
				cost[position] = mTracks[i]->CalcDist(centerpoints[j]);
			}
		}

		/// Apply Hungarian algorithm
		AssignmentProblemSolver APS;
		APS.Solve(cost, N, M, assignment, AssignmentProblemSolver::optimal);


		/// Mark assignments with too large a distance between them
		for (int i = 0; i< assignment.size(); ++i)
		{
			if (assignment[i] != -1)
			{
				/// if distance is too large, mark that assignment (set to -1)
				auto pos = i + assignment[i] * N;
				if (cost[pos] > mDistanceThreshold)
				{
					assignment[i] = -1;
				}
			}
		}

		/// Find blobs that are Unassigned
		std::vector<int> detections_not_assigned;
		std::vector<int>::iterator it;

		for (int i = 0; i < centerpoints.size(); ++i)
		{
			it = find(assignment.begin(), assignment.end(), i);
			if (it == assignment.end())
			{
				detections_not_assigned.push_back(i);
			}
		}

		/// Start a new track if there are unassigned blobs
		if (!detections_not_assigned.empty()) {
			for (int i = 0; i < detections_not_assigned.size(); ++i)
			{
				TrackPtr tr = std::make_shared<Track>(mNextTrackId++, centerpoints[detections_not_assigned[i]], rects[i], mMaxSkippedFrames, mHistorySize);
				mTracks.push_back(tr);
			}
		}
	}
	else
	{
		/// if no new blobs were detected, cost is set to zero for the cost matrix
		/// Assignment matrix is set to -1 for each track. This will make the Kalman update with the mPrediction instead of a measurement
		for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j < M; ++j)
			{
				cost[i + j * N] = 0;
			}
		}

		for (int i = 0; i< mTracks.size(); ++i) {
			assignment.push_back(-1);
		}
	}

	/// Update the Kalman filter for each assignment
	for (int i = 0; i < assignment.size(); ++i)
	{
		/// If there is an assignment, correct KF with coordinates
		/// If we have assigned detect, then update using its coordinates
		if (assignment[i] != -1)
		{	
			mTracks[i]->Update(centerpoints[assignment[i]], rects[assignment[i]], true);
		}
		else
		{
			/// if there is no assignment, use predictions
			mTracks[i]->Update(cv::Point2d(0, 0), cv::Rect(), false);

		}
	}

	/// Analyze tracks that haven't been detected for longer than max skipped frames, then remove them
	for (int i = 0; i<mTracks.size(); ++i)
	{
		if (mTracks[i]->isTrackDead())
		{
			mTracks.erase(mTracks.begin() + i);
			assignment.erase(assignment.begin() + i);
			i--;
		}
	}

}

void CentroidTracker::Draw(cv::Mat & frame, bool drawTrack, cv::Scalar& color)
{
	for (auto &mTrack : mTracks) {
		mTrack->Draw(frame, drawTrack, color);
	}
}


void CentroidTracker::clear()
{
	mTracks.clear();
}


