#ifndef MODEL_PARAMETERS_H
#define MODEL_PARAMETERS_H

#include <opencv2\opencv.hpp>

struct model_parameters
{
	model_parameters(std::string _weights, std::string _config, double _conf, int _width, int _height, int _channels, cv::Scalar _mean, bool _swapRB, bool _crop, double _scaleFactor, double _nmsThreshold, std::vector < std::string>& _allClasses)
	{
		weights_path = _weights;
		config_path = _config;
		confidence_threshold = _conf;
		net_input_width = _width;
		net_input_height = _height;
		net_input_channels = _channels;
		net_mean = _mean;
		crop = _crop;
		swapRB = _swapRB;
		scale_factor = _scaleFactor;
		nms_threshold = _nmsThreshold;
		class_names = _allClasses;
	}
	std::string weights_path;
	std::string config_path;
	double confidence_threshold;
	std::vector<std::string> class_names;
	int net_input_width;
	int net_input_height;
	int net_input_channels;
	cv::Scalar net_mean;
	bool crop;
	bool swapRB;
	double scale_factor;
	double nms_threshold;
};

#endif