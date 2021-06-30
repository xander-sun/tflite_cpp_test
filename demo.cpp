#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ctime>
#include <vector>

using namespace std;
using namespace cv;


float expit(float x) {
    return 1.f / (1.f + expf(-x));
}


//nms
float iou(Rect& rectA, Rect& rectB)
{
    int x1 = std::max(rectA.x, rectB.x);
    int y1 = std::max(rectA.y, rectB.y);
    int x2 = std::min(rectA.x + rectA.width, rectB.x + rectB.width);
    int y2 = std::min(rectA.y + rectA.height, rectB.y + rectB.height);
    int w = std::max(0, (x2 - x1 + 1));
    int h = std::max(0, (y2 - y1 + 1));
    float inter = w * h;
    float areaA = rectA.width * rectA.height;
    float areaB = rectB.width * rectB.height;
    float o = inter / (areaA + areaB - inter);
    return (o >= 0) ? o : 0;
}

void test() {
		clock_t time_start, time_end;
 
		// Load model
		std::unique_ptr<tflite::FlatBufferModel> model =
		tflite::FlatBufferModel::BuildFromFile("../models/model.tflite");
		
		// Build the interpreter
		tflite::ops::builtin::BuiltinOpResolver resolver;
		std::unique_ptr<tflite::Interpreter> interpreter;
		tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);


		TfLiteTensor* output_locations = nullptr;
		TfLiteTensor* output_classes = nullptr;
		TfLiteTensor* output_num_detections = nullptr;
		TfLiteTensor* output_scores = nullptr;
		
		// Resize input tensors, if desired
		interpreter->AllocateTensors();
		int input = interpreter->inputs()[0];
			
			//float* input = interpreter->typed_input_tensor<float>(0);
			// Find input tensors.
			
		std::cout<<"input"<<input<<std::endl;
		std::cout<<"type"<<interpreter->tensor(input)->type<<std::endl;
		
		TfLiteIntArray* dims = interpreter->tensor(input)->dims;
		int wanted_height = dims->data[1];
		int wanted_width = dims->data[2];
		int wanted_channels = dims->data[3];
		std::cout<< wanted_height<<wanted_width<<wanted_channels <<std::endl;

		// load labelmap
		std::vector<std::string> labels;
		auto file_name="../models/labelmap.txt";
		std::ifstream labelmap(file_name);
		for( std::string line; getline( labelmap, line ); )
		{
			labels.push_back( line);
		}
		
		auto cam = cv::VideoCapture(0);
		// auto cam = cv::VideoCapture("../demo.mp4");
	    cam.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		cam.set(CV_CAP_PROP_FRAME_HEIGHT, 960);	
		cam.set(CV_CAP_PROP_FPS, 100);
		//std::cout<<"camera FPS:" << cam.get(CV_CAP_PROP_FPS) << std::endl;

		// get camera state
		auto cam_width =cam.get(CV_CAP_PROP_FRAME_WIDTH);
		auto cam_height = cam.get(CV_CAP_PROP_FRAME_HEIGHT);
		std::cout<<"camera width and height:" << cam_width << cam_height << std::endl;
		while (true) {
			// get, resize, convert image to (1, 320, 320, 3) float32 
			cv::Mat fimage;
			auto success = cam.read(fimage);
			if (!success) {
				std::cout << "cam fail" << std::endl;
				break;
			}
			cv::Mat image;
			// resize image
		
			
			resize(fimage, image, Size(wanted_height, wanted_width), 0, 0, INTER_AREA);
			const float IMAGE_MEAN = 128.0;
			const float IMAGE_STD = 128.0;
			// convert image to float32
			image.convertTo(image, CV_32FC3, 1 / IMAGE_STD, -IMAGE_MEAN / IMAGE_STD);
			// feed image to tensor
			std::cout<<"interpreter->tensor(input)->data.f..."<<interpreter->tensor(input)->data.f<<std::endl;
			time_start = clock();
			
			memcpy(interpreter->tensor(input)->data.f, image.data, image.total() * image.elemSize());
		
			//interpreter->SetAllowFp16PrecisionForFp32(true);
			interpreter->SetNumThreads(1);
			interpreter->Invoke();

			output_locations = interpreter->tensor(interpreter->outputs()[0]);
			output_classes   = interpreter->tensor(interpreter->outputs()[1]);
			output_scores    = interpreter->tensor(interpreter->outputs()[2]);
			output_num_detections   = interpreter->tensor(interpreter->outputs()[3]);

			const float *detection_locations = output_locations->data.f;
			const float *detection_classes = output_classes->data.f;
			const float *detection_scores = output_scores->data.f;
			const int num_detections = (int) *output_num_detections->data.f;

			vector< vector<float> > boxes;
			vector<int> classes;
			vector<float> scores;
			for (int i = 0; i < num_detections && i < 20; ++i) {
				scores.push_back(detection_scores[i]);
				classes.push_back((int) detection_classes[i]);

		// Get the bbox, make sure its not out of the image bounds, and scale up to src image size
				auto ymin = std::fmax(0.0f, detection_locations[4 * i] * fimage.rows);
				auto xmin = std::fmax(0.0f, detection_locations[4 * i + 1] * fimage.cols);
				auto ymax = std::fmin(float(fimage.rows - 1), detection_locations[4 * i + 2] * fimage.rows);
				auto xmax = std::fmin(float(fimage.cols - 1), detection_locations[4 * i + 3] * fimage.cols);
				boxes.push_back({ymin,xmin,ymax,xmax});
	}
			time_end = clock();
			RNG rng(12345);
			int index = 0;
			for(int i = 0; i < num_detections && i < 20; i++)
     		 {		
					auto score=scores[i];
					std::cout << i <<":"<<score << std::endl;
					if (score < 0.50f) continue;

					Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
					auto cls = classes[i];
					
					cv::rectangle(fimage, cv::Point(boxes[i][1], boxes[i][0]), cv::Point(boxes[i][3], boxes[i][2]),color, 6);
					cv::putText(fimage, labels[cls], cv::Point(boxes[i][1], boxes[i][0] - 5),
					cv::FONT_HERSHEY_COMPLEX, 1.2, color);
					std::cout<< cls<< std::endl;
					index++;
			}
			
			std::cout << "size: "<<index << std::endl;
			std::cout << "time costs: "<< (double)(time_end - time_start) * 1000 / CLOCKS_PER_SEC << " ms" << std::endl;
		
			cv::namedWindow("object detect by c++", CV_WINDOW_NORMAL);
			cv::imshow("object detect by c++", fimage);
			auto k = cv::waitKey(30);
			if (k == 'q') {
					break;
			}
		}


}
int main(int argc, char** argv) {
    test();
    return 0;
}
