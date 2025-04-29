# Homework 2 – Robot Trajectory Estimation and Point Cloud Mapping
Name: Emerson Hall 811299513 erh04959@uga.edu
Professor: Guoyo Lu
Class: Introduction to Robotics Engineering

## Overview
This project estimates the robot's trajectory and reconstructs a local 3D point cloud using monocular image sequences.  
Both the robot path and the environment structure are visualized and saved into a video.

The robot motion and environmental mapping are based on:
- Feature detection and matching
- Fundamental and Essential matrix estimation
- Camera pose recovery (Rotation and Translation)
- Triangulation of 3D points

The point cloud **moves together** with the robot trajectory, as shown in the final output video.

---

## Files
- `main.cpp` — Main code for trajectory estimation, point cloud triangulation, and video output
- `first_200_right/` — Folder containing the input image sequence (200 images)
- `output_combined.avi` — Final video showing:
  - Top: camera images with tracked green feature points
  - Bottom: dark blue trajectory line with moving light blue point cloud

---

## How It Works
1. ORB features are detected and matched between consecutive frames.
2. The Essential matrix is computed from matched points using the given camera intrinsic matrix.
3. Camera motion (Rotation and Translation) is recovered.
4. The trajectory is plotted based on cumulative translation.
5. 3D world points are triangulated from matches and are **attached relative to the robot's current pose** (moving point cloud).
6. A final combined video is saved showing both views.

---

## How to Run
### Requirements:
- OpenCV 4.0+ installed
- `g++` compiler

### Steps:
1. Be in the 'homework2' directory
2. Compile the code:
    g++ main.cpp -o Homework2 `pkg-config --cflags --libs opencv4`
3. Run the executable:
    ./Homework2
4. Open the resulting video:
    open output_combined.avi

