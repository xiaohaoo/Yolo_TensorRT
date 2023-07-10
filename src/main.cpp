/************************************************************************
* Function Name : nucFindThread
* Create Date : 2014/06/07
* Author/Corporation : xiaohao
**
Description : Find a proper thread inthread array.
* If it’s a new then search an empty.
*
* Param : ThreadNo： someParamdescription
* ThreadStatus： someParamdescription
**
Return Code : Return Code description,eg:
ERROR_Fail: not find a thread
ERROR_SUCCEED: found
*
* Global Variable : DISP_wuiSegmentAppID
* File Static Variable : naucThreadNo
* Function Static Variable : None
*
*------------------------------------------------------------------------
* Revision History
* No. Date Revised by Item Description
* V0.5 2014/06/21 your name … …
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
        putText(trt_engine.raw_image, "FPS: " + (1000 / exec_duration()), Point(10, 35), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        imshow("Yolo", trt_engine.raw_image);
        if (cv::waitKey(1) == 27)
        {
            destroyAllWindows();
            cap.release();
            break;
        }
    }
    return 0;
}
