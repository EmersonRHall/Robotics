#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Camera Intrinsics
Mat K = (Mat_<double>(3,3) << 707.0493, 0, 604.0814,
                               0, 707.0493, 180.5066,
                               0, 0, 1);

// Trajectory and Point Cloud
vector<Point> trajectory_points;
vector<Point> cloud_points;

// Draw robot trajectory (dark blue line)
void drawTrajectory(Mat &traj, Point current_pos, Point last_pos)
{
    line(traj, last_pos, current_pos, Scalar(255, 0, 0), 2);  // Dark blue line
    circle(traj, current_pos, 3, Scalar(255, 0, 0), -1);       // Dark blue dot
}

// Draw dense light blue point cloud
void drawPointCloud(Mat &traj)
{
    for (auto &p : cloud_points)
    {
        circle(traj, p, 2, Scalar(255, 255, 200), -1);  // Medium light blue points
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
    Point last_point(600, 250); // Start higher (shifted up)

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

        // ORB detection
        vector<KeyPoint> kp1, kp2;
        Mat desc1, desc2;
        orb->detectAndCompute(img1_gray, noArray(), kp1, desc1);
        orb->detectAndCompute(img2_gray, noArray(), kp2, desc2);

        // Match features
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

        // Essential matrix and pose
        Mat E = findEssentialMat(pts1, pts2, K);
        Mat R, t;
        recoverPose(E, pts1, pts2, K, R, t);

        Mat Rt = Mat::eye(4,4,CV_64F);
        R.copyTo(Rt(Rect(0,0,3,3)));
        t.copyTo(Rt(Rect(3,0,1,3)));
        pose = pose * Rt.inv();

        // Get current robot position
        double Tx = pose.at<double>(0,3);
        double Tz = pose.at<double>(2,3);

        Point current_point(
            int(Tx * 2.5 + 600),
            int(-Tz * 1.2 + 200)   // SHIFT UP (was 280)
        );
        trajectory_points.push_back(current_point);

        // Triangulate points
        Mat P1 = K * Mat::eye(3,4,CV_64F);
        Mat P2 = K * Rt(Rect(0,0,4,3));

        Mat pts4D;
        triangulatePoints(P1, P2, pts1, pts2, pts4D);

        // Add moving point cloud
        for (int c = 0; c < pts4D.cols; c++)
        {
            Mat x = pts4D.col(c);
            x /= x.at<float>(3);

            float X = x.at<float>(0);
            float Z = x.at<float>(2);

            if (fabs(X) < 50 && Z > 0 && Z < 50)
            {
                float X_moving = X + Tx; // Shift by robot position
                float Z_moving = Z + Tz; // Shift by robot position

                int u = int(X_moving * 2.5 + 600);
                int v = int(-Z_moving * 1.2 + 250);  // SHIFT UP (was 280)

                if (u > 0 && u < width && v > 0 && v < traj_height)
                    cloud_points.push_back(Point(u,v));
            }
        }

        // === Draw bottom panel ===
        traj_frame = Mat::zeros(traj_height, width, CV_8UC3);
        drawPointCloud(traj_frame);  // Moving smaller light blue cloud
        for (size_t j = 1; j < trajectory_points.size(); j++)
        {
            drawTrajectory(traj_frame, trajectory_points[j], trajectory_points[j-1]); // Dark blue trajectory
        }

        // Resize top image
        Mat img1_resized;
        resize(img1_color, img1_resized, Size(width, height));

        // Combine top + bottom
        Mat full_frame;
        vconcat(img1_resized, traj_frame, full_frame);

        output_video.write(full_frame);

        // Update last point
        last_point = current_point;
    }

    output_video.release();

    cout << "✅ FINAL FINAL FINAL output_combined.avi saved!" << endl;
    return 0;
}
