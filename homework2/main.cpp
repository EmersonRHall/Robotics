#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// === Load Intrinsics ===
cv::Mat getK() {
    return (cv::Mat_<double>(3, 3) << 
        718.856, 0.0, 607.1928,
        0.0, 718.856, 185.2157,
        0.0, 0.0, 1.0);
}

// === Load image paths ===
std::vector<std::string> loadImagePaths(const std::string& folder) {
    std::vector<std::string> paths;
    for (const auto& entry : fs::directory_iterator(folder)) {
        paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// === Detect and Track Features ===
void detectAndTrack(const cv::Mat& prev, const cv::Mat& next, 
                    std::vector<cv::Point2f>& prevPts, 
                    std::vector<cv::Point2f>& nextPts) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::ORB> detector = cv::ORB::create(2000);
    detector->detect(prev, keypoints);
    cv::KeyPoint::convert(keypoints, prevPts);

    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev, next, prevPts, nextPts, status, err);

    std::vector<cv::Point2f> filteredPrev, filteredNext;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            filteredPrev.push_back(prevPts[i]);
            filteredNext.push_back(nextPts[i]);
        }
    }
    prevPts = filteredPrev;
    nextPts = filteredNext;
}

int main() {
    std::string img_folder = "kitti_seq/00/image_0";
    std::vector<std::string> images = loadImagePaths(img_folder);
    cv::Mat K = getK();

    // Camera path accumulation
    std::vector<cv::Point2f> trajectory;
    cv::Mat R_f = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_f = cv::Mat::zeros(3, 1, CV_64F);

    for (size_t i = 0; i < images.size() - 1; ++i) {
        cv::Mat img1 = cv::imread(images[i], cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(images[i + 1], cv::IMREAD_GRAYSCALE);

        if (img1.empty() || img2.empty()) break;

        std::vector<cv::Point2f> pts1, pts2;
        detectAndTrack(img1, img2, pts1, pts2);

        // Compute Essential matrix
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC);
        cv::Mat R, t;
        cv::recoverPose(E, pts1, pts2, K, R, t);

        // Update pose
        t_f = t_f + (R_f * t);
        R_f = R * R_f;

        trajectory.emplace_back(t_f.at<double>(0), t_f.at<double>(2));

        // Plot trajectory
        cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);
        for (const auto& pt : trajectory) {
            int x = int(pt.x) + 300;
            int y = int(pt.y) + 100;
            cv::circle(traj, cv::Point(x, y), 1, cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("Trajectory", traj);
        cv::waitKey(1);
    }

    std::cout << "Visual Odometry Finished." << std::endl;
    cv::waitKey(0);
    return 0;
}
