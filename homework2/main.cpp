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
void drawTrajectory(Mat &canvas, Point2d position, Scalar color)
{
    circle(canvas, Point(int(position.x)+300, int(position.y)+100), 1, color, 2);
}

int main()
{
    string folder = "first_200_right/";
    int num_images = 200;

    Ptr<ORB> orb = ORB::create(2000);
    BFMatcher matcher(NORM_HAMMING);

    Mat combined = Mat::zeros(600, 1200, CV_8UC3);
    VideoWriter combined_video("output_combined.avi", VideoWriter::fourcc('M','J','P','G'), 10, combined.size());

    Mat pose = Mat::eye(4, 4, CV_64F);

    for (int i = 0; i < num_images-1; i++)
    {
        char filename1[100], filename2[100];
        sprintf(filename1, "%s%06d.png", folder.c_str(), i);
        sprintf(filename2, "%s%06d.png", folder.c_str(), i+1);

        Mat img1 = imread(filename1, IMREAD_GRAYSCALE);
        Mat img2 = imread(filename2, IMREAD_GRAYSCALE);

        if (img1.empty() || img2.empty())
        {
            cout << "Cannot load images " << filename1 << " or " << filename2 << endl;
            continue;
        }

        // Detect ORB keypoints and descriptors
        vector<KeyPoint> kp1, kp2;
        Mat desc1, desc2;
        orb->detectAndCompute(img1, noArray(), kp1, desc1);
        orb->detectAndCompute(img2, noArray(), kp2, desc2);

        // Match descriptors
        vector<DMatch> matches;
        matcher.match(desc1, desc2, matches);

        // Select good matches
        sort(matches.begin(), matches.end());
        matches.resize(100); // Take top 100 matches

        // Extract matched points
        vector<Point2f> pts1, pts2;
        for (auto &m : matches)
        {
            pts1.push_back(kp1[m.queryIdx].pt);
            pts2.push_back(kp2[m.trainIdx].pt);
        }

        // Estimate Fundamental matrix
        Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC);

        // Estimate Essential matrix
        Mat E = findEssentialMat(pts1, pts2, K);

        // Decompose Essential matrix to get R and t
        Mat R, t;
        recoverPose(E, pts1, pts2, K, R, t);

        // Update pose
        Mat Rt = Mat::eye(4,4,CV_64F);
        R.copyTo(Rt(Range(0,3), Range(0,3)));
        t.copyTo(Rt(Range(0,3), Range(3,4)));

        pose = pose * Rt.inv();

        // Draw trajectory
        Point2d center(pose.at<double>(0,3)*5, pose.at<double>(2,3)*5);
        drawTrajectory(combined, center, Scalar(0,255,0));

        // Triangulation
        Mat proj1 = K * Mat::eye(3,4,CV_64F);
        Mat proj2 = K * Rt(Range(0,3), Range::all());

        Mat pts4D;
        triangulatePoints(proj1, proj2, pts1, pts2, pts4D);

        for (int c = 0; c < pts4D.cols; c++)
        {
            Mat x = pts4D.col(c);
            x /= x.at<float>(3);
            int x_proj = int(x.at<float>(0)*30) + 300;
            int y_proj = int(x.at<float>(1)*30) + 300;

            if (x_proj > 0 && y_proj > 0 && x_proj < 1200 && y_proj < 600)
            {
                circle(combined, Point(x_proj, y_proj), 1, Scalar(255,0,0), 1);
            }
        }

        combined_video.write(combined);
    }

    combined_video.release();

    cout << "Done! Combined video saved: output_combined.avi" << endl;
    return 0;
}
