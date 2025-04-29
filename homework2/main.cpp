#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Camera Intrinsics (K)
Mat K = (Mat_<double>(3,3) << 707.0493, 0, 604.0814,
                               0, 707.0493, 180.5066,
                               0, 0, 1);

vector<Point> trajectory_points;
vector<Point> cloud_points;

// Draw red trajectory
void drawTrajectory(Mat &traj, Point current_pos, Point last_pos)
{
    line(traj, last_pos, current_pos, Scalar(0, 0, 255), 2); // Red thin line
    circle(traj, current_pos, 3, Scalar(0, 0, 255), -1);      // Red dot
}

// Draw light blue point cloud
void drawPointCloud(Mat &traj)
{
    for (auto &p : cloud_points)
    {
        circle(traj, p, 1, Scalar(255, 255, 200), -1); // Light blue small dots
    }
}

int main()
{
    string folder = "first_200_right/";
    int num_images = 200;

    Ptr<ORB> orb = ORB::create(5000);  // More features
    BFMatcher matcher(NORM_HAMMING);

    int width = 1200;
    int height = 370;
    int traj_height = 300;
    int full_height = height + traj_height;

    Mat traj_frame = Mat::zeros(traj_height, width, CV_8UC3);
    VideoWriter output_video("output_combined.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(width, full_height));

    Mat pose = Mat::eye(4, 4, CV_64F);
    Point last_point(600, 280);
    trajectory_points.push_back(last_point);

    for (int i = 0; i < num_images-1; i++)
    {
        // Load images
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

        // Feature detection and matching
        vector<KeyPoint> kp1, kp2;
        Mat desc1, desc2;
        orb->detectAndCompute(img1_gray, noArray(), kp1, desc1);
        orb->detectAndCompute(img2_gray, noArray(), kp2, desc2);

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

        // Draw green points
        for (const auto& p : pts1)
        {
            circle(img1_color, p, 2, Scalar(0, 255, 0), -1);
        }

        // Compute Fundamental matrix (optional, not used)
        Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC);

        // Compute Essential matrix
        Mat E = findEssentialMat(pts1, pts2, K);

        // Decompose Essential matrix into R, t
        Mat R, t;
        recoverPose(E, pts1, pts2, K, R, t);

        // Update global pose
        Mat Rt = Mat::eye(4,4,CV_64F);
        R.copyTo(Rt(Rect(0,0,3,3)));
        t.copyTo(Rt(Rect(3,0,1,3)));
        pose = pose * Rt.inv();

        // Update robot position
        Point current_point(
            int(pose.at<double>(0,3) * 1.5 + 600),
            int(-pose.at<double>(2,3) * 1.5 + 280)
        );
        trajectory_points.push_back(current_point);

        // Triangulate points for cloud
        Mat P1 = K * Mat::eye(3,4,CV_64F);
        Mat P2 = K * Rt(Rect(0,0,4,3));

        Mat pts4D;
        triangulatePoints(P1, P2, pts1, pts2, pts4D);

        // Accumulate point cloud
        for (int c = 0; c < pts4D.cols; c++)
        {
            Mat x = pts4D.col(c);
            x /= x.at<float>(3); // normalize

            float X = x.at<float>(0);
            float Z = x.at<float>(2);

            int u = int(X * 1.5 + 600);
            int v = int(Z * 1.5 + 280);

            if (u > 0 && u < width && v > 0 && v < traj_height)
            {
                cloud_points.push_back(Point(u, v));
            }
        }

        // === Draw bottom panel ===
        traj_frame = Mat::zeros(traj_height, width, CV_8UC3);
        drawPointCloud(traj_frame);  // Light blue cloud
        for (size_t j = 1; j < trajectory_points.size(); j++)
        {
            drawTrajectory(traj_frame, trajectory_points[j], trajectory_points[j-1]); // Red trajectory
        }

        // Resize top image
        Mat img1_resized;
        resize(img1_color, img1_resized, Size(width, height));

        // Combine top and bottom
        Mat full_frame;
        vconcat(img1_resized, traj_frame, full_frame);

        output_video.write(full_frame);

        // Update last point
        last_point = current_point;
    }

    output_video.release();

    cout << "✅ FINAL corrected video saved: output_combined.avi" << endl;
    return 0;
}
