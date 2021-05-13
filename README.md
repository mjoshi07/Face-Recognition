# Face-Recognition
* To run the application, clone the repository and build it using CMake and any generator[VS 2015 used]
* Should satisfy all dependencies
* Download the weight files for all the models and paste them where .xml files are present, i.e here [models](/data/models) 
* Put `.jpg` images **only** inside the faceImages folder, here [faceImages](/data/faceImages)
* 1 image should contain only 1 face and names of all files should be unique
* Name of the image file is assigned as the faceId to the face detected in the image
* No Need to crop the `.jpg` images, application will first detect faces in it and then store the face embeddings from face images
* Rename you video as **"test_video.mp4"** and move it here [video](/data/video)

## How it works
* Scans the [faceImages](/data/faceImages) for faces in images, extracts 512 feature vector for each image and stores them in memory
* Reads the specified input video stream frame-by-frame, be it a camera device or a video file 
* The application deploys 2 models[face detection and face embeddings] and runs them in synchronous manner
* An input frame is processed by the face detection model to predict face bounding boxes
* Face images are created using the face bounding boxes and are sent to face embedding model to provide a 512 feaure vector of each face
* Face matching is performed using the feature vectors generated at run time and the ones generate from database
* Dot product/cosine distance is calculated to assign FaceIDs, if no match is found **"unknown"** ID is assigned to that face

## Models Used
* `face-detection-retail-0005` to detect faces in a frame
* `landmarks-regression-retail-0009` to predict landmarks in a face Image
* `SphereFace` [converted to IR] so as to use with INFERENCE_ENGINE

## Dependencies
* Opencv[4.5.1 and above] built with InferenceEngine backend
* OpenVino[2021 and above]
* CMake[3.10 and above]
* vcpkg [dlib Threads]
* C++11

## Test Results
![output1](https://user-images.githubusercontent.com/31381335/118119835-9d80c000-b40c-11eb-8d7f-84274952e845.png)
![output2](https://user-images.githubusercontent.com/31381335/118119871-a7a2be80-b40c-11eb-9d7b-ed03723b75cc.png)
![output3](https://user-images.githubusercontent.com/31381335/118119879-aa051880-b40c-11eb-982f-510ea0406aba.png)



## Things to Do
- [x] Face Detection
- [x] Face Landmarks Detection [Added but not using currently]
- [x] Face Recognition
- [ ] Face Alignment before recognition using landmarks
- [ ] Add Tracker and maintain state of each recognized face
- [ ] Perform face detection, landmarks detection and recognition in asynchronous manner
- [ ] Age Detection
- [ ] Gender Detection
- [ ] Emotion Detection
