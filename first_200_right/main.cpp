#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

int main() {
    string folder = "images";
    vector<string> imagePaths;

    // Load image paths
    for (const auto& entry : fs::directory_iterator(folder)) {
        imagePaths.push_back(entry.path().string());
    }

    sort(imagePaths.begin(), imagePaths.end()); // Ensure order

    Ptr<ORB> orb = ORB::create(1000);

    for (size_t i = 0; i < imagePaths.size() - 1; ++i) {
        Mat img1 = imread(imagePaths[i], IMREAD_GRAYSCALE);
        Mat img2 = imread(imagePaths[i + 1], IMREAD_GRAYSCALE);

        if (img1.empty() || img2.empty()) {
            cerr << "Error reading image pair: " << i << endl;
            continue;
        }

        vector<KeyPoint> kp1, kp2;
        Mat des1, des2;

        orb->detectAndCompute(img1, noArray(), kp1, des1);
        orb->detectAndCompute(img2, noArray(), kp2, des2);

        BFMatcher matcher(NORM_HAMMING);
        vector<vector<DMatch>> knn_matches;
        matcher.knnMatch(des1, des2, knn_matches, 2);

        // Ratio test
        vector<DMatch> good_matches;
        for (auto& m : knn_matches) {
            if (m[0].distance < 0.75f * m[1].distance) {
                good_matches.push_back(m[0]);
            }
        }

        // RANSAC
        vector<Point2f> pts1, pts2;
        for (const auto& match : good_matches) {
            pts1.push_back(kp1[match.queryIdx].pt);
            pts2.push_back(kp2[match.trainIdx].pt);
        }

        vector<char> mask;
        if (pts1.size() >= 4) {
            vector<uchar> uchar_mask;
            findHomography(pts1, pts2, RANSAC, 3, uchar_mask);
            mask.assign(uchar_mask.begin(), uchar_mask.end());
        } else {
            mask.resize(good_matches.size(), 1);
        } 

        // Draw matches
        Mat outImg;
        drawMatches(
            img1, kp1,
            img2, kp2,
            good_matches,
            outImg,
            Scalar::all(-1),
            Scalar::all(-1),
            mask,
            DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
        );

        // Save output
        fs::create_directory("output");
        string outPath = "output/match_" + to_string(i) + ".png";
        imwrite(outPath, outImg);
        cout << "Saved: " << outPath << endl;
    }

    return 0;
}
