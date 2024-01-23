#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolov5s.hpp"
#include "rknnPool.hpp"
int main(int argc, char **argv)
{
    char *model_name = NULL;
    if (argc != 3)
    {
        printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
        return -1;
    }
    // 参数二，模型所在路径/The path where the model is located
    model_name = (char *)argv[1];
    // 参数三, 视频/摄像头
    char *vedio_name = argv[2];

    // 初始化rknn线程池/Initialize the rknn thread pool
    int threadNum = 12;
    rknnPool<rkYolov5s, cv::Mat, cv::Mat> testPool(model_name, threadNum);
    if (testPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    cv::namedWindow("Camera Left");
    cv::namedWindow("Camera Right");

    cv::VideoCapture captureLeft, captureRight;
    // 假设摄像头ID分别为0和1
    captureLeft.open(0); 
    captureRight.open(2);
    cv::VideoCapture capture;
    if (strlen(vedio_name) == 1)
        capture.open((int)(vedio_name[0] - '0'));
    else
        capture.open(vedio_name);

    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    int frames = 0;
    auto beforeTime = startTime;
 while (captureLeft.isOpened() && captureRight.isOpened()) {
        cv::Mat imgLeft, imgRight;
        bool hasFrameLeft = captureLeft.read(imgLeft);
        bool hasFrameRight = captureRight.read(imgRight);

        if (!hasFrameLeft || !hasFrameRight)
            break;

        // 提交到线程池
        if (hasFrameLeft && testPool.put(imgLeft) != 0)
            break;
        if (hasFrameRight && testPool.put(imgRight) != 0)
            break;

        // 获取处理后的图像
        if (frames >= threadNum) {
            if (testPool.get(imgLeft) != 0)
                break;
            if (testPool.get(imgRight) != 0)
                break;

            cv::imshow("Camera Left", imgLeft);
            cv::imshow("Camera Right", imgRight);
        }
        if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
            break;
        frames++;

        if (frames % 120 == 0)
        {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("120帧内平均帧率:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
        }
    }

    // 清空rknn线程池/Clear the thread pool
    while (true)
    {
        cv::Mat img;
        if (testPool.get(img) != 0)
            break;
        cv::imshow("Camera FPS", img);
        if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
            break;
        frames++;
    }

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    return 0;
}