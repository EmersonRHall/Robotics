#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

using namespace std;
using namespace cv;

// === Intrinsic Camera Matrix ===
Mat K = (Mat_<double>(3, 3) << 707.0493, 0, 604.0814,
                                0, 707.0493, 180.5066,
                                0, 0, 1);

int main() {
    string folder_path = "first_200_right/";
    vector<string> image_files;

    // Load image filenames
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        image_files.push_back(entry.path().string());
    }
    sort(image_files.begin(), image_files.end());

    // Trajectory plot
    int traj_size = 600;
    Mat traj = Mat::zeros(traj_size, traj_size, CV_8UC3);

    // Initialize camera pose
    Mat R_f = Mat::eye(3, 3, CV_64F);
    Mat t_f = Mat::zeros(3, 1, CV_64F);

    // Video writers
    VideoWriter traj_video("output_trajectory.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(traj_size, traj_size));
    VideoWriter pointcloud_video("output_pointcloud.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(800, 800));

    Ptr<ORB> orb = ORB::create();

    for (size_t i = 0; i < image_files.size() - 1; i++) {
        Mat img1 = imread(image_files[i], IMREAD_GRAYSCALE);
        Mat img2 = imread(image_files[i+1], IMREAD_GRAYSCALE);

        if (img1.empty() || img2.empty()) {
            cout << "Error loading images!" << endl;
            continue;
        }

        vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;

        // Detect ORB features
        orb->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
        orb->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

        // Match features
        BFMatcher matcher(NORM_HAMMING);
        vector<DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);

        // Filter good matches
        double max_dist = 0; double min_dist = 100;
        for (int k = 0; k < descriptors1.rows; k++) {
            double dist = matches[k].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }
        vector<DMatch> good_matches;
        for (int k = 0; k < descriptors1.rows; k++) {
            if (matches[k].distance <= max(2*min_dist, 30.0)) {
                good_matches.push_back(matches[k]);
            }
        }

        // Extract matched points
        vector<Point2f> pts1, pts2;
        for (size_t j = 0; j < good_matches.size(); j++) {
            pts1.push_back(keypoints1[good_matches[j].queryIdx].pt);
            pts2.push_back(keypoints2[good_matches[j].trainIdx].pt);
        }

        // Fundamental matrix
        Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC);

        // Essential matrix
        Mat E = K.t() * F * K;

        // Recover Pose
        Mat R, t, mask;
        recoverPose(E, pts1, pts2, K, R, t, mask);

        // Update current pose
        t_f = t_f + (R_f * t);
        R_f = R * R_f;

        int x = int(t_f.at<double>(0)) + traj_size/2;
        int y = int(t_f.at<double>(2)) + traj_size/2;

        circle(traj, Point(x, y), 1, Scalar(0, 0, 255), 2);

        // Show and save frame
        traj_video.write(traj);
        imshow("Trajectory", traj);
        waitKey(1);

        // Triangulate points
        Mat P1 = K * Mat::eye(3, 4, CV_64F);
        Mat Rt;
        hconcat(R, t, Rt);
        Mat P2 = K * Rt;

        Mat pts_4d;
        triangulatePoints(P1, P2, pts1, pts2, pts_4d);

        // Draw 3D points
        Mat cloud = Mat::zeros(800, 800, CV_8UC3);
        for (int k = 0; k < pts_4d.cols; k++) {
            Point3f pt;
            pt.x = pts_4d.at<float>(0,k) / pts_4d.at<float>(3,k);
            pt.y = pts_4d.at<float>(1,k) / pts_4d.at<float>(3,k);
            pt.z = pts_4d.at<float>(2,k) / pts_4d.at<float>(3,k);

            int u = int(pt.x * 10) + 400;
            int v = int(pt.y * 10) + 400;
            if (u >= 0 && u < 800 && v >= 0 && v < 800)
                circle(cloud, Point(u,v), 1, Scalar(255, 255, 255), 1);
        }
        pointcloud_video.write(cloud);
    }

    traj_video.release();
    pointcloud_video.release();
    cout << "Done! Videos saved." << endl;

    return 0;
}
