# Yolov8_TensorRT

## Installation

```bash
git clone git@github.com:xiaohaoo/Yolo_TensorRT.git

cd Yolo_TensorRT
cmake -DCMAKE_BUILD_TYPE:STRING=Release -B build
cmake --build build --config Release --target install

# run app
cd build\output
.\yolo.exe
```
