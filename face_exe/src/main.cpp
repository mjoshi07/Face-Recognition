//#include "../../face_lib/include/FaceDetector.h"
//#include "../../face_lib/include/FaceLandmark.h"
//#include "../../face_lib/include/FaceEmbedding.h"
//#include "../../face_lib/include/CentroidTracker.h"
//#include "../../face_lib/include/LoadModel.h"

#include "FaceDetector.h"
#include "FaceLandmark.h"
#include "FaceEmbedding.h"
#include "CentroidTracker.h"
#include "LoadModel.h"
#include "Face.h"


std::string data_path = "..//..//models";
std::string imgs_path = data_path + "//faceImgs";


const int distance_threshold = 50;
const int max_skipped_frames = 25;
const int history_size = 60;

const int frame_to_skip = 5;
unsigned long frame_number = 0;	

std::vector<std::string> only_class_to_detect = {"face"};
const std::string video_src = data_path +"//project_video.mp4";
const std::string windowName = "Output frame";

void scanDBLoadMem(FaceDetector& _faceDetector, FaceEmbedding& _faceEmbedder, std::string& _imgPath);
void recognize_faces(CentroidTracker& _tracker);
void rectsForTracker(std::vector<Face>& faces, std::vector<cv::Rect>& rects);

int main()
{    
	//load face models
	LoadFaceModel loadModels(data_path, true, true, true);

	cv::dnn::Net detectionModel = loadModels.getDetectionModel();
	cv::dnn::Net embeddingsModel = loadModels.getEmbeddingsModel();
	cv::dnn::Net landmarksModel = loadModels.getLandmarksModel();

    //Create a detection model object 
    FaceDetector faceDetector(detectionModel,0.4, only_class_to_detect, frame_to_skip);

	//Create a landmark detection model object
	FaceLandmark faceLandmarksDetector(landmarksModel);

	//Create a embeddings[reidentification] model object
	FaceEmbedding faceEmbedder(embeddingsModel);

	// Scan the database, detect faces, generate face embeddings and load them into memory for Recognition
	//scanDBLoadMem(faceDetector, faceEmbedder, imgs_path);

    //Create a centroid tracker object
    CentroidTracker tracker(distance_threshold, max_skipped_frames, history_size);

    //Create video capture object and open video source
    cv::VideoCapture cap(0);
    cv::Mat frame;

	//Create a window to display the output frame
	cv::namedWindow(windowName, cv::WINDOW_FREERATIO);

    if (cap.isOpened())
    {
        while(1)
        {
            if(cap.read(frame))
            {
				std::vector<Face> faces;
				std::vector<cv::Rect> rects;

				faceDetector.getDetectedRects(frame, faces, frame_number);
				faceLandmarksDetector.getFaceLandmarks(frame,faces);
				faceEmbedder.getEmbeddedFeatures(faces);

				rectsForTracker(faces, rects);
				tracker.Update(rects);
				tracker.Draw(frame, false);

				//recognize_faces(tracker);
         
				cv::imshow(windowName, frame);
				char k = cv::waitKey(1);
				if (k == 27 || k == 'q')
				{
					break;
				}
				else if (k == 'p')
				{
					cv::waitKey(0);
				}
				frame_number++;
			}
			else
			{
				break;
			}
        }
    }


    return 0;
}

void rectsForTracker(std::vector<Face>& faces, std::vector<cv::Rect>& rects)
{
	if (faces.size())
	{
		for (auto& face : faces)
		{
			rects.push_back(face.faceRect);
		}
	}
}
