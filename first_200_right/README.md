Feature Tracking - Introduction to Robotics

Name: Emerson Hall
Professor: Guoyu Lu
Course: Introduction to Robotics
Due Date: Sunday, April 13th, 2025

Overview:

This project implements feature tracking on a set of 200 images as part of a robotics vision assignment. It demonstrates cross-frame feature matching using ORB and robust matching with RANSAC. The output is visualized as a video of tracked features between adjacent frames.

Methodology:

Feature Detection & Description: ORB (Oriented FAST and Rotated BRIEF)
Feature Matching: Brute-Force Matcher with Hamming distance and k-Nearest Neighbor (k-NN)
Outlier Rejection: RANSAC for homography-based filtering
Visualization: Matches drawn frame-by-frame and saved as PNGs, compiled into an MP4 video
How to Run:

### Build the project (from the root directory): 
mkdir build && cd build
cmake ..
make
cd ..

### Run the executable: 
./build/FeatureTracking

### Rename output images (for video creation): 
cd output
a=0; for f in match_*.png; do mv "$f" $(printf "match_%03d.png" "$a"); ((a++));
cd ..
### Generate video: 
ffmpeg -framerate 10 -i output/match_%03d.png -c:v libx264 -pix_fmt yuv420p tracking_result.mp4