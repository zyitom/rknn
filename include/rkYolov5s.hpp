#ifndef RKYOLOV5S_H
#define RKYOLOV5S_H

#include "rknn_api.h"

#include "opencv2/core/core.hpp"
struct DetectedObject {
    std::string name;
    float conf;
};


static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

class rkYolov5s
{

private:
    int ret;
    std::mutex mtx;
    std::string model_path;
    unsigned char *model_data;

    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];

    int channel, width, height;
    int img_width, img_height;

    float nms_threshold, box_conf_threshold;

public:
    std::vector<DetectedObject> processDetectionResults() ;
    rkYolov5s(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
cv::Mat infer(cv::Mat &orig_img, const std::string& camera_side);
    ~rkYolov5s();
};

#endif