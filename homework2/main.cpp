#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::putText(img, "OpenCV is working!", cv::Point(50, 240), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::imshow("Test", img);
    cv::waitKey(0);
    return 0;
}
