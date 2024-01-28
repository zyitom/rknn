#ifndef RKYOLOV5S_H
#define RKYOLOV5S_H

#include "rknn_api.h"
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/string.hpp"
#include "opencv2/core/core.hpp"
class FilterTypeManager {
public:
    static FilterTypeManager& getInstance() {
        static FilterTypeManager instance; // 单例实例
        return instance;
    }

    void setFilterType(const std::string& type) {
        std::lock_guard<std::mutex> lock(mutex_);
        filterType_ = type;
    }

    std::string getFilterType() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return filterType_;
    }

private:
    mutable std::mutex mutex_;
    std::string filterType_;

    FilterTypeManager() {} // 构造函数私有化
};

struct DetectedObject {
    std::string name;
    float conf;
};


static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);
class DetectorNode : public rclcpp::Node {
public:
    DetectorNode() : Node("detector_node") {
        publisher_ = this->create_publisher<std_msgs::msg::String>("detection_topic", 10);
    }

    void publish_detection(const std::string& message) {
        auto msg = std_msgs::msg::String();
        msg.data = message;
        publisher_->publish(msg);
    }

private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};
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
bool filter_type_set = false; 
public:
//     filter_type = "B"; 
    std::vector<DetectedObject> processDetectionResults() ;
    rkYolov5s(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    std::string filter_type; 
    void setFilterType(const std::string& type);
cv::Mat infer(cv::Mat &orig_img, const std::string& camera_side);
    ~rkYolov5s();
    std::shared_ptr<DetectorNode> detector_node_;
    const std::string& getFilterType() const {
        return filter_type;
    }

};

#endif