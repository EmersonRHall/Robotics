#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Camera Intrinsics (given)
Mat K = (Mat_<double>(3,3) << 707.0493, 0, 604.0814,
                               0, 707.0493, 180.5066,
                               0, 0, 1);

// Helper: Draw trajectory
void drawTrajectory(Mat &traj, Point2d current_pos, Point2d last_pos)
{
    line(traj, last_pos, current_pos, Scalar(255, 0, 0), 1); // thin dark blue current line
    circle(traj, current_pos, 2, Scalar(200, 200, 255), -1); // light blue static points
}

int main()
{
    string folder = "first_200_right/";
    int num_images = 200;

    Ptr<ORB> orb = ORB::create(2000);
    BFMatcher matcher(NORM_HAMMING);

    int width = 1200;
    int height = 370; // resized color image height
    int traj_height = 300;
    int full_height = height + traj_height;

    Mat traj = Mat::zeros(traj_height, width, CV_8UC3);
    VideoWriter output_video("output_combined.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(width, full_height));

    Mat pose = Mat::eye(4, 4, CV_64F);
    Point2d last_point(600, 150); // center starting point in traj canvas

    for (int i = 0; i < num_images-1; i++)
    {
        char filename1[100], filename2[100];
        sprintf(filename1, "%s%06d.png", folder.c_str(), i);
        sprintf(filename2, "%s%06d.png", folder.c_str(), i+1);

        Mat img1_color = imread(filename1, IMREAD_COLOR);
        Mat img2_color = imread(filename2, IMREAD_COLOR);

        Mat img1_gray, img2_gray;
        cvtColor(img1_color, img1_gray, COLOR_BGR2GRAY);
        cvtColor(img2_color, img2_gray, COLOR_BGR2GRAY);

        if (img1_gray.empty() || img2_gray.empty())
        {
            cout << "Cannot load images " << filename1 << " or " << filename2 << endl;
            continue;
        }

        // Detect ORB keypoints and descriptors
        vector<KeyPoint> kp1, kp2;
        Mat desc1, desc2;
        orb->detectAndCompute(img1_gray, noArray(), kp1, desc1);
        orb->detectAndCompute(img2_gray, noArray(), kp2, desc2);

        // Match descriptors
        vector<DMatch> matches;
        matcher.match(desc1, desc2, matches);

        // Select good matches
        sort(matches.begin(), matches.end());
        matches.resize(200); // Top 200 matches

        // Extract matched points
        vector<Point2f> pts1, pts2;
        for (auto &m : matches)
        {
            pts1.push_back(kp1[m.queryIdx].pt);
            pts2.push_back(kp2[m.trainIdx].pt);
        }

        // Draw green points on img1
        for (const auto& p : pts1)
        {
            circle(img1_color, p, 2, Scalar(0, 255, 0), -1); // green small circles
        }

        // Estimate Essential matrix
        Mat E = findEssentialMat(pts1, pts2, K);
        Mat R, t;
        recoverPose(E, pts1, pts2, K, R, t);

        // Update pose
        Mat Rt = Mat::eye(4,4,CV_64F);
        R.copyTo(Rt(Range(0,3), Range(0,3)));
        t.copyTo(Rt(Range(0,3), Range(3,4)));
        pose = pose * Rt.inv();

        // Update trajectory
        Point2d current_point(pose.at<double>(0,3)*5 + 600, pose.at<double>(2,3)*5 + 150);
        drawTrajectory(traj, current_point, last_point);
        last_point = current_point;

        // Resize img1_color to fit top half
        Mat img1_resized;
        resize(img1_color, img1_resized, Size(width, height));

        // Combine top (image) and bottom (traj)
        Mat full_frame;
        vconcat(img1_resized, traj, full_frame);

        // Write to video
        output_video.write(full_frame);
    }

    output_video.release();

    cout << "Done! Combined video saved: output_combined.avi" << endl;
    return 0;
}
