#ifndef _TRT_ENGINE_H_
#define _TRT_ENGINE_H_

#include <iostream>
#include <NvInfer.h>
#include <cassert>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <NvOnnxParser.h>

using namespace std;
using namespace nvinfer1;
using namespace cv;
using namespace nvonnxparser;

#define OUT

// C*W*H Input size of the model.
static const int INPUT_SIZE[] = {3, 640, 640};
// Output size of the model.
static const int OUTPUT_SIZE[] = {1, 84, 8400};

/************************************************************************
 *
 * Encapsulation of data processing and model reasoning.
 *
 ************************************************************************/
struct TrtEngine
{
    float scale = 0.0f;
    Mat img_input;
    Mat raw_image;
    vector<Rect> boxes;
    vector<float> scores;
    vector<int> classes;
    void build_engine();
    void preprocess();
    void inference();
    void draw_image();
};

#endif