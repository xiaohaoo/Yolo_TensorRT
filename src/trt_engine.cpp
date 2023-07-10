#pragma once

#include "trt_engine.h"

/************************************************************************
 *
 * Global environment variable information.
 *
 ************************************************************************/

static shared_ptr<IRuntime> engine_runtime{nullptr};
static shared_ptr<ICudaEngine> trt_engine{nullptr};
static shared_ptr<IExecutionContext> engine_context{nullptr};

class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kERROR)
            std::cout << msg << std::endl;
    }
} trt_logger;

/************************************************************************
 *
 * To build a Tensorrt engine, first parse the ONNX model,
 * serialize the ONNX into a Tensorrt model, and save it.
 * Then, load it directly the second time.
 *
 ************************************************************************/
void TrtEngine::build_engine()
{
    _putenv("CUDA_MODULE_LOADING=LAZY");
    int device = 0;
    cudaGetDevice(&device);
    printf("[I] Current GPU device id is %d.\n", device);

    const string tensorrt_model = "model/yolov8n.engine";
    const string onnx_model = "model/yolov8n.onnx";
    ifstream tensorrt_file(tensorrt_model, ios::binary);

    /************************************************************************
     * If the Tensorrt model does not exist, parse the ONNX model.
     ************************************************************************/
    if (!tensorrt_file.good())
    {
        printf("[I] The first load requires initializing the model. Please be patient and wait.\n");
        shared_ptr<IBuilder> trt_builder = shared_ptr<IBuilder>(createInferBuilder(trt_logger));
        const uint32_t network_flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        shared_ptr<INetworkDefinition> trt_network = shared_ptr<INetworkDefinition>(trt_builder->createNetworkV2(network_flags));
        shared_ptr<IBuilderConfig> build_config = shared_ptr<IBuilderConfig>(trt_builder->createBuilderConfig());
        shared_ptr<IParser> trt_parser = shared_ptr<IParser>(createParser(*trt_network, trt_logger));
        trt_parser->parseFromFile(onnx_model.c_str(), static_cast<int>(ILogger::Severity::kINFO));
        trt_builder->buildSerializedNetwork(*trt_network, *build_config);
        trt_engine.reset(trt_builder->buildEngineWithConfig(*trt_network, *build_config));
        shared_ptr<IHostMemory> serialized_engine(trt_engine->serialize());
        std::ofstream outFile(tensorrt_model, ios::binary);
        outFile.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());
        outFile.flush();
        outFile.close();
    }

    /************************************************************************
     * If the Tensorrt model exists, load it directly.
     ************************************************************************/
    else
    {
        tensorrt_file.seekg(0, ios::end);
        auto file_size = tensorrt_file.tellg();
        tensorrt_file.seekg(0, ios::beg);
        vector<char> buffer(file_size);
        tensorrt_file.read(buffer.data(), file_size);
        tensorrt_file.close();
        engine_runtime.reset(createInferRuntime(trt_logger));
        trt_engine.reset(engine_runtime->deserializeCudaEngine(buffer.data(), file_size));
    }
    engine_context.reset(trt_engine->createExecutionContext());

    printf("[I] %s\n", "---------Yolov8s context load success ---------");
};

/************************************************************************
 *
 * Preprocess the image.
 *
 ************************************************************************/
void TrtEngine::preprocess()
{

    /************************************************************************
     * Adjust the maximum edge of the image to not exceed 640 and fill in the
     * minimum edge.
     ************************************************************************/
    raw_image = img_input.clone();
    int img_w = img_input.size().width;
    int img_h = img_input.size().height;
    int max_wh = max(img_w, img_h);
    scale = (float)INPUT_SIZE[1] / max_wh;
    int w_pad = max_wh - img_w;
    int h_pad = max_wh - img_h;
    copyMakeBorder(img_input, img_input, 0, h_pad, 0, w_pad, BORDER_CONSTANT, Scalar(114, 114, 114));
    cout << "img_input: " << img_input.size << endl;
    dnn::blobFromImage(img_input, img_input, 1.0f / 255.0f, Size(INPUT_SIZE[1], INPUT_SIZE[2]), Scalar(), true, false, CV_32F);
}

/************************************************************************
 *
 * Input image data into the model for inference
 *
 ************************************************************************/
void TrtEngine::inference()
{
    /************************************************************************
     * Allocate CUDA memory
     ************************************************************************/
    void *input_cuda = nullptr;
    void *output_cuda = nullptr;
    cudaStream_t stream;
    cudaMalloc(&input_cuda, img_input.total() * img_input.elemSize());
    cudaMalloc(&output_cuda, OUTPUT_SIZE[0] * OUTPUT_SIZE[1] * OUTPUT_SIZE[2] * sizeof(float));
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(input_cuda, img_input.data, img_input.total() * img_input.elemSize(), cudaMemcpyHostToDevice, stream);

    /************************************************************************
     * Run inference
     ************************************************************************/
    void *buffers[2] = {input_cuda, output_cuda};
    engine_context->enqueueV2(buffers, stream, nullptr);

    /************************************************************************
     * Copy output data to CPU memory
     ************************************************************************/
    Mat output = Mat(OUTPUT_SIZE[1], OUTPUT_SIZE[2], CV_32F);
    cudaMemcpyAsync(output.data, output_cuda, OUTPUT_SIZE[0] * OUTPUT_SIZE[1] * OUTPUT_SIZE[2] * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(input_cuda);
    cudaFree(output_cuda);

    /************************************************************************
     * Convert the output to a 8400*84 float array.
     ************************************************************************/
    transpose(output, output);

    /************************************************************************
     * Decode the output.
     ************************************************************************/
    vector<Rect> detect_boxes;
    vector<float> detect_scores;
    vector<int> detect_classes;
    for (size_t i = 0; i < output.rows; i++)
    {
        float *row = output.row(i).ptr<float>();
        float *max_class_socre = max_element(row + 4, row + 4 + 80);
        if (*max_class_socre > 0.25)
        {
            int x = row[0];
            int y = row[1];
            int w = row[2];
            int h = row[3];
            int x0 = static_cast<int>((x - 0.5f * w));
            int y0 = static_cast<int>((y - 0.5f * h));
            Rect box(x0 / scale, y0 / scale, w / scale, h / scale);
            detect_boxes.push_back(box);
            detect_scores.push_back(*max_class_socre);
            detect_classes.push_back(static_cast<int>(max_class_socre - row - 4));
        }
    }
    vector<int> selected_indices;
    dnn::NMSBoxes(detect_boxes, detect_scores, 0.45f, 0.50f, selected_indices);

    vector<Rect> selected_boxes;
    vector<float> selected_scores;
    vector<int> selected_classes;

    selected_boxes.reserve(selected_indices.size());
    selected_scores.reserve(selected_indices.size());
    selected_classes.reserve(selected_indices.size());

    for (int &index : selected_indices)
    {
        Rect box = detect_boxes[index];
        float score = detect_scores[index];
        int class_idx = detect_classes[index];
        selected_boxes.push_back(box);
        selected_scores.push_back(score);
        selected_classes.push_back(class_idx);
    }
    boxes = selected_boxes;
    scores = selected_scores;
    classes = selected_classes;
}

/************************************************************************
 *
 * Draw image
 *
 ************************************************************************/
void TrtEngine::draw_image()
{
    for (size_t i = 0; i < boxes.size(); i++)
    {
        Rect box = boxes[i];
        float score = scores[i];
        int class_idx = classes[i];
        rectangle(raw_image, box, Scalar(0, 255, 0), 2);
        putText(raw_image, to_string(class_idx), Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    }
}