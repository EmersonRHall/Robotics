#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Camera Intrinsics
Mat K = (Mat_<double>(3,3) << 707.0493, 0, 604.0814,
                               0, 707.0493, 180.5066,
                               0, 0, 1);

// Cumulative Robot Positions
vector<Point> all_positions;

// Helper: Draw full light blue path
void drawFullPath(Mat &traj)
{
    if (all_positions.size() < 2) return;
    for (size_t i = 1; i < all_positions.size(); i++)
    {
        line(traj, all_positions[i-1], all_positions[i], Scalar(255, 255, 200), 6); // Thick light blue
    }
}

// Helper: Draw current red trajectory
void drawTrajectory(Mat &traj, Point current_pos, Point last_pos)
{
    line(traj, last_pos, current_pos, Scalar(0, 0, 255), 2); // Thin red line
    circle(traj, current_pos, 3, Scalar(0, 0, 255), -1);      // Red point
}

int main()
{
    string folder = "first_200_right/";
    int num_images = 200;

    Ptr<ORB> orb = ORB::create(5000);
    BFMatcher matcher(NORM_HAMMING);

    int width = 1200;
    int height = 370;
    int traj_height = 300;
    int full_height = height + traj_height;

    Mat traj_frame = Mat::zeros(traj_height, width, CV_8UC3);
    VideoWriter output_video("output_combined.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(width, full_height));

    Mat pose = Mat::eye(4, 4, CV_64F);
    Point last_point(600, 280);

    all_positions.push_back(last_point); // Start path

    for (int i = 0; i < num_images-1; i++)
    {
        char filename1[100], filename2[100];
        snprintf(filename1, sizeof(filename1), "%s%06d.png", folder.c_str(), i);
        snprintf(filename2, sizeof(filename2), "%s%06d.png", folder.c_str(), i+1);

        Mat img1_color = imread(filename1, IMREAD_COLOR);
        Mat img2_color = imread(filename2, IMREAD_COLOR);

        if (img1_color.empty() || img2_color.empty())
        {
            cout << "Cannot load images " << filename1 << " or " << filename2 << endl;
            continue;
        }

        Mat img1_gray, img2_gray;
        cvtColor(img1_color, img1_gray, COLOR_BGR2GRAY);
        cvtColor(img2_color, img2_gray, COLOR_BGR2GRAY);

        // ORB detection
        vector<KeyPoint> kp1, kp2;
        Mat desc1, desc2;
        orb->detectAndCompute(img1_gray, noArray(), kp1, desc1);
        orb->detectAndCompute(img2_gray, noArray(), kp2, desc2);

        // Match descriptors
        vector<DMatch> matches;
        matcher.match(desc1, desc2, matches);

        sort(matches.begin(), matches.end());
        matches.resize(500);

        vector<Point2f> pts1, pts2;
        for (auto &m : matches)
        {
            pts1.push_back(kp1[m.queryIdx].pt);
            pts2.push_back(kp2[m.trainIdx].pt);
        }

        // Draw lots of green points
        for (const auto& p : pts1)
        {
            circle(img1_color, p, 2, Scalar(0, 255, 0), -1);
        }

        // Essential matrix and pose
        Mat E = findEssentialMat(pts1, pts2, K);
        Mat R, t;
        recoverPose(E, pts1, pts2, K, R, t);

        Mat Rt = Mat::eye(4,4,CV_64F);
        R.copyTo(Rt(Range(0,3), Range(0,3)));
        t.copyTo(Rt(Range(0,3), Range(3,4)));
        pose = pose * Rt.inv();

        // Current robot position (X scaled, Z flipped and scaled)
        Point current_point(
            int(pose.at<double>(0,3) * 1.5 + 600),
            int(-pose.at<double>(2,3) * 1.5 + 280)
        );

        all_positions.push_back(current_point); // Save for full path

        // Draw
        drawFullPath(traj_frame);               // Big static light blue
        drawTrajectory(traj_frame, current_point, last_point); // Red line on top

        last_point = current_point;

        // Resize top
        Mat img1_resized;
        resize(img1_color, img1_resized, Size(width, height));

        // Combine
        Mat full_frame;
        vconcat(img1_resized, traj_frame, full_frame);

        output_video.write(full_frame);
    }

    output_video.release();

    cout << "✅ FINAL corrected video saved: output_combined.avi" << endl;
    return 0;
}
