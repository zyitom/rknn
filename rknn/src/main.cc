#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/string.hpp"
#include <optional>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolov5s.hpp"
#include "rknnPool.hpp"
class FilterSubscriber : public rclcpp::Node {
public:
    FilterSubscriber() : Node("filter_subscriber"), new_filter_received(false) {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "filter_topic", 10,
            std::bind(&FilterSubscriber::filter_callback, this, std::placeholders::_1));
    }

    std::string getFilterType() const {
        return filter_type;
    }

    bool isNewFilterReceived() {
        bool temp = new_filter_received;
        new_filter_received = false; // 重置标志
        return temp;
    }

private:
void filter_callback(const std_msgs::msg::String::SharedPtr msg) {
    if (msg->data == "R" || msg->data == "B") {
        FilterTypeManager::getInstance().setFilterType(msg->data);
        new_filter_received = true;
        RCLCPP_INFO(this->get_logger(), "New filter type received: '%s'", msg->data.c_str());
    }
}


    std::string filter_type;
    bool new_filter_received;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto filter_subscriber = std::make_shared<FilterSubscriber>();
    rkYolov5s detector("../../yolov5_L.rknn");

    int threadNum = 12;
    bool poolInitialized = false;
 std::unique_ptr<rknnPool<rkYolov5s, cv::Mat, cv::Mat>> testPool;

    cv::namedWindow("Camera Left");
    cv::namedWindow("Camera Right");

    cv::VideoCapture captureLeft, captureRight;
    captureLeft.open(0);
    captureRight.open(2);

    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    int frames = 0;
    auto beforeTime = startTime;

    bool startInference = false;

    while (rclcpp::ok() && !startInference) {
        rclcpp::spin_some(filter_subscriber);

        if (filter_subscriber->isNewFilterReceived()) {
            std::string filterType = filter_subscriber->getFilterType();
            RCLCPP_INFO(filter_subscriber->get_logger(), "Setting filter type to: '%s'", filterType.c_str());
            detector.setFilterType(filterType);
            startInference = true;

 testPool = std::make_unique<rknnPool<rkYolov5s, cv::Mat, cv::Mat>>("../../yolov5_L.rknn", threadNum);
            if (testPool->init() != 0) {
                printf("rknnPool init fail!\n");
                return -1;
            }

            startInference = true;
        }
    }
    
    // 当跳出循环时，开始推理过程
    while (startInference) {
        cv::Mat imgLeft, imgRight;
        bool hasFrameLeft = captureLeft.read(imgLeft);
        bool hasFrameRight = captureRight.read(imgRight);

        if (!hasFrameLeft || !hasFrameRight) {
            std::cerr << "Failed to capture frame from camera." << std::endl;
            break;
        }

        if (imgLeft.empty() || imgRight.empty()) {
            std::cerr << "Empty frame received from camera." << std::endl;
            continue;
        }

if (hasFrameLeft && testPool->put(std::make_pair(imgLeft, "left")) != 0)
            break;
        if (hasFrameRight && testPool->put(std::make_pair(imgRight, "right")) != 0)
            break;

        std::pair<cv::Mat, std::string> resultLeft;
        std::pair<cv::Mat, std::string> resultRight;
        if (frames >= threadNum) {
            if (testPool->get(resultLeft) != 0 || testPool->get(resultRight) != 0)
                break;

            cv::imshow("Camera Left", resultLeft.first);
            cv::imshow("Camera Right", resultRight.first);
        }

        if (cv::waitKey(1) == 'q')
            break;
        frames++;

        if (frames % 120 == 0) {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("120帧内平均帧率:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
        }
    }

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    rclcpp::shutdown();
    return 0;
}