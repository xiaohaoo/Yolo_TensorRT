/************************************************************************
* Project Name: Yolov8-TensorRT
* Create Date: 2023/07/01
* Author: xiaohao
**
Description : Using Tensorrt and Yolov8 Reasoning.
*
************************************************************************/

#include "main.h"

void detection_img(string img_path, TrtEngine trt_engine)
{
    Mat img = imread(img_path);
    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    exec_duration();
    trt_engine.img_input = img;
    trt_engine.preprocess();
    trt_engine.inference();
    trt_engine.draw_image();
    exec_duration();
    imshow("Image", trt_engine.raw_image);
    string img_name = img_path.substr(0, img_path.find_last_of("."));
    cv::imwrite(img_name + "_detected.png", trt_engine.raw_image);
    if (cv::waitKey(-1) == 27 || getWindowProperty("Image", WND_PROP_VISIBLE) < 1)
    {
        destroyAllWindows();
    }
}

void detection_video(string video_path, TrtEngine trt_engine)
{
    VideoCapture cap(video_path);
    Mat img;
    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    while (cap.read(img))
    {
        exec_duration();
        trt_engine.img_input = img;
        trt_engine.preprocess();
        trt_engine.inference();
        trt_engine.draw_image();
        putText(trt_engine.raw_image, "FPS: " + to_string((1000 / exec_duration())), Point(10, 35), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        imshow("Image", trt_engine.raw_image);
        if (cv::waitKey(1) == 27 || getWindowProperty("Image", WND_PROP_VISIBLE) < 1)
        {
            cap.release();
            destroyAllWindows();
            break;
        }
    }
}

int main()
{
    printf("[I] %s\n", "--------------- Program Start -----------------");
    TrtEngine trt_engine;
    trt_engine.build_engine();
    string path = "assets/img_3.jpg";
    detection_img(path, trt_engine);
    return 0;
}
