/************************************************************************
* Project Name: Yolov8-TensorRT
* Create Date: 2023/07/01
* Author: xiaohao
**
Description : Using Tensorrt and Yolov8 Reasoning.
*
************************************************************************/

#include "main.h"

int main()
{
    printf("[I] %s\n", "--------------- Program Start -----------------");
    TrtEngine trt_engine;
    trt_engine.build_engine();
    VideoCapture cap("assets/video_1.mp4");
    Mat img;
    while (cap.read(img))
    {
        exec_duration();
        trt_engine.img_input = img;
        trt_engine.preprocess();
        trt_engine.inference();
        trt_engine.draw_image();
        putText(trt_engine.raw_image, "FPS: " + to_string((1000 / exec_duration())), Point(10, 35), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        imshow("Yolo", trt_engine.raw_image);
        if (cv::waitKey(1) == 27 || getWindowProperty("Yolo", WND_PROP_VISIBLE) < 1)
        {
            destroyAllWindows();
            cap.release();
            break;
        }
    }
    return 0;
}
