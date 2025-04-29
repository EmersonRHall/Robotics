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

// Cumulative Point Cloud
vector<Point2f> all_cloud_points;

// Helper: Draw trajectory
void drawTrajectory(Mat &traj, Point2d current_pos, Point2d last_pos)
{
    line(traj, last_pos, current_pos, Scalar(0, 0, 255), 2); // RED thick line
    circle(traj, current_pos, 3, Scalar(0, 0, 255), -1);      // RED circle at point
}

// Helper: Accumulate point cloud
void accumulatePointCloud(const Mat& pts4D)
{
    for (int c = 0; c < pts4D.cols; c++)
    {
        Mat x = pts4D.col(c);
        x /= x.at<float>(3);

        float x_world = x.at<float>(0);
        float z_world = x.at<float>(2);

        int x_proj = int(x_world * 3.0) + 600;
        int y_proj = int(z_world * 3.0) + 250;

        if (x_proj > 0 && x_proj < 1200 && y_proj > 0 && y_proj < 300)
        {
            all_cloud_points.push_back(Point2f(x_proj, y_proj));
        }
    }
}

// Helper: Draw point cloud
void drawAllCloudPoints(Mat &traj)
{
    for (const auto& p : all_cloud_points)
    {
        circle(traj, p, 2, Scalar(255, 255, 200), -1); // light blue points
    }
}

int main()
{
    string folder = "first_200_right/";
    int num_images = 200;

    Ptr<ORB> orb = ORB::create(5000); // More ORB features
    BFMatcher matcher(NORM_HAMMING);

    int width = 1200;
    int height = 370;
    int traj_height = 300;
    int full_height = height + traj_height;

    Mat traj = Mat::zeros(traj_height, width, CV_8UC3);
    VideoWriter output_video("output_combined.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(width, full_height));

    Mat pose = Mat::eye(4, 4, CV_64F);
    Point2d last_point(600, 250);

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

        // Matching descriptors
        vector<DMatch> matches;
        matcher.match(desc1, desc2, matches);

        sort(matches.begin(), matches.end());
        matches.resize(500); // More matches = more point cloud

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

        // Estimate Essential matrix and recover pose
        Mat E = findEssentialMat(pts1, pts2, K);
        Mat R, t;
        recoverPose(E, pts1, pts2, K, R, t);

        Mat Rt = Mat::eye(4,4,CV_64F);
        R.copyTo(Rt(Range(0,3), Range(0,3)));
        t.copyTo(Rt(Range(0,3), Range(3,4)));
        pose = pose * Rt.inv();

        // Compute current point (X regular, Z flipped)
        Point2d current_point(
            pose.at<double>(0,3) * 3.0 + 600,
           -pose.at<double>(2,3) * 3.0 + 250
        );

        // Triangulate points
        Mat proj1 = K * Mat::eye(3,4,CV_64F);
        Mat proj2 = K * Rt(Range(0,3), Range::all());
        Mat pts4D;
        triangulatePoints(proj1, proj2, pts1, pts2, pts4D);

        accumulatePointCloud(pts4D);

        // Draw bottom panel
        Mat traj_frame = Mat::zeros(traj_height, width, CV_8UC3);
        drawAllCloudPoints(traj_frame); // Draw point cloud first
        drawTrajectory(traj_frame, current_point, last_point); // Then red line on top

        traj = traj_frame.clone();
        last_point = current_point;

        // Resize top image
        Mat img1_resized;
        resize(img1_color, img1_resized, Size(width, height));

        // Combine
        Mat full_frame;
        vconcat(img1_resized, traj, full_frame);

        output_video.write(full_frame);
    }

    output_video.release();

    cout << "✅ FINAL video saved: output_combined.avi" << endl;
    return 0;
}
