#include "Face.h"

int main()
{    
	// path to data folder where DNN models and face database for face recognition is stored
	const std::string data_path = "..\\..\\data";

	// starting frame number
	unsigned long frame_number = 0;

	// video file path location
	const std::string video_src = data_path + "//video//test_video.mp4";
	
	// display window name
	const std::string windowName = "Output frame";

	// create a Face class object
	Face faceObject(data_path, true, false, true);

    	// create video capture object and open video source
   	cv::VideoCapture cap(video_src);
	
	// create opencv Mat object
   	cv::Mat frame;

	// create a window to display the output frame 
	cv::namedWindow(windowName, cv::WINDOW_FREERATIO);

    if (cap.isOpened()) // check if video capture object is open
    {
        while(1)
        {
		if(cap.read(frame))
		{
			// call the runFaceRecognition and start face recognition
			faceObject.runFaceRecognition(frame, frame_number);
			
			// display the video stream
			cv::imshow(windowName, frame);
			
			// display the frame for 1 milisecond 
			char k = cv::waitKey(1);
			
			// if user presses "Esc" or "q" stop displaying the frame and exit the while loop
			if (k == 27 || k == 'q')
			{
				break;
			}
			else if (k == 'p') // if user presses "p" pause the video streaming
			{
				cv::waitKey(0);
			}
			
			// increment the frame number
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

