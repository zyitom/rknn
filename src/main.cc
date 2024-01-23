#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolov5s.hpp"
#include "rknnPool.hpp"


extern std::vector<DetectionResult> globalDetectionResults;

int main(int argc, char **argv) {
    char *model_name = NULL;
    if (argc != 3) {
        printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
        return -1;
    }
    char ros2Input = 'R';
    model_name = (char *)argv[1];
    char *vedio_name = argv[2];

    int threadNum = 12;
    rknnPool<rkYolov5s, cv::Mat, cv::Mat> testPool(model_name, threadNum);
    if (testPool.init() != 0) {
        printf("rknnPool init fail!\n");
        return -1;
    }

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

    std::vector<DetectionResult> leftCameraResults;
    std::vector<DetectionResult> rightCameraResults;

 while (captureLeft.isOpened() && captureRight.isOpened()) {
    cv::Mat imgLeft, imgRight;
    bool hasFrameLeft = captureLeft.read(imgLeft);
    bool hasFrameRight = captureRight.read(imgRight);

    if (!hasFrameLeft || !hasFrameRight)
        break;

            if (hasFrameLeft) {
            globalDetectionResults.clear();
            testPool.put(imgLeft);
            testPool.get(imgLeft);
            leftCameraResults = globalDetectionResults;
            if (!leftCameraResults.empty()) {
                for (const auto& result : leftCameraResults) {
                    // 检查结果是否匹配ROS2输入
                    if (result.name[0] == ros2Input) {
                        printf("左侧摄像头: %s @ (%d %d %d %d) %f\n", result.name.c_str(), result.left, result.top, 
                               result.right, result.bottom, result.prop);
                    }
                }
            }
            cv::imshow("Camera Left", imgLeft);
        }

        // 处理右侧摄像头的结果
        if (hasFrameRight) {
            globalDetectionResults.clear();
            testPool.put(imgRight);
            testPool.get(imgRight);
            rightCameraResults = globalDetectionResults;
            if (!rightCameraResults.empty()) {
                for (const auto& result : rightCameraResults) {
                    // 检查结果是否匹配ROS2输入
                    if (result.name[0] == ros2Input) {
                        printf("右侧摄像头: %s @ (%d %d %d %d) %f\n", result.name.c_str(), result.left, result.top, 
                               result.right, result.bottom, result.prop);
                    }
                }
            }
            cv::imshow("Camera Right", imgRight);
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

// for (const auto& result : leftCameraResults) {
//     printf("左侧摄像头: %s @ (%d %d %d %d) %f\n", result.name.c_str(), result.left, result.top, 
//            result.right, result.bottom, result.prop);
// }

// for (const auto& result : rightCameraResults) {
//     printf("右侧摄像头: %s @ (%d %d %d %d) %f\n", result.name.c_str(), result.left, result.top, 
//            result.right, result.bottom, result.prop);
// }


    while (true) {
        cv::Mat img;
        if (testPool.get(img) != 0)
            break;
        cv::imshow("Camera FPS", img);
        if (cv::waitKey(1) == 'q')
            break;
        frames++;
    }

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;
    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    return 0;
}
