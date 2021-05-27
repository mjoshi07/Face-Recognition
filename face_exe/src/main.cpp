#include "Face.h"

int main()
{    

	//Path to data folder where DNN models and database for face recognition is stored
	const std::string data_path = "..\\..\\data";

	const std::string video_src = data_path + "//video//test_video.mp4";
	const std::string windowName = "Output frame";

	//Create a Face class object
	Face faceObject(data_path, false, true);

    //Create video capture object and open video source
    cv::VideoCapture cap(video_src);
    cv::Mat frame;

	//Create a window to display the output frame
	cv::namedWindow(windowName, cv::WINDOW_FREERATIO);

    if (cap.isOpened())
    {
        while(1)
        {
            if(cap.read(frame))
            {

				faceObject.runFaceRecognition(frame);
				         
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
			}
			else
			{
				break;
			}
        }
    }
    return 0;
}

