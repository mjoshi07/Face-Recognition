#include "Face.h"

int main()
{    
	// path to data folder where DNN models and database for face recognition is stored
	const std::string data_path = "..\\..\\data";

	unsigned long frame_number = 0;

	const std::string video_src = data_path + "//video//test_video.mp4";
	const std::string windowName = "Output frame";

	//Create a Face class object
	Face faceObject(data_path, true, false, true);

    //Create video capture object and open video source[webcam in this case]
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

				faceObject.runFaceRecognition(frame, frame_number);
				         
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

