#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/viz.hpp> // For 3D visualization
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// Load camera calibration data
bool loadCameraCalibration(const string& filename, Mat& camera_matrix, Mat& dist_coeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Error: Could not open calibration file." << endl;
        return false;
    }
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> dist_coeffs;
    fs.release();
    return true;
}

// Detect calibration target (chessboard) and compute its pose
bool detectCalibrationTarget(const Mat& frame, const Mat& camera_matrix, const Mat& dist_coeffs, 
                             vector<Point2f>& corners, Mat& rvec, Mat& tvec) {
    Size pattern_size(9, 6); // Chessboard size
    bool found = findChessboardCorners(frame, pattern_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
    if (found) {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

        // Define 3D points of the chessboard in the world coordinate system
        vector<Point3f> object_points;
        float square_size = 1.0f; // Size of a square in your defined unit (e.g., meters)
        for (int i = 0; i < pattern_size.height; i++) {
            for (int j = 0; j < pattern_size.width; j++) {
                object_points.push_back(Point3f(j * square_size, i * square_size, 0));
            }
        }

        // Compute pose of the chessboard
        solvePnP(object_points, corners, camera_matrix, dist_coeffs, rvec, tvec);
    }
    return found;
}

// Triangulate 3D points from 2D correspondences
void triangulatePoints(const Mat& camera_matrix, const Mat& rvec, const Mat& tvec, 
                       const vector<Point2f>& points1, const vector<Point2f>& points2, 
                       vector<Point3f>& points3d) {
    Mat R;
    Rodrigues(rvec, R); // Convert rotation vector to rotation matrix

    Mat P1 = Mat::eye(3, 4, CV_64F); // Projection matrix for the first camera
    Mat P2(3, 4, CV_64F);
    hconcat(R, tvec, P2); // Projection matrix for the second camera
    P2 = camera_matrix * P2;

    Mat points4d;
    triangulatePoints(P1, P2, points1, points2, points4d);

    // Convert from homogeneous coordinates to 3D
    for (int i = 0; i < points4d.cols; i++) {
        Mat x = points4d.col(i);
        x /= x.at<float>(3); // Normalize
        points3d.push_back(Point3f(x.at<float>(0), x.at<float>(1), x.at<float>(2)));
    }
}

// Visualize 3D point cloud using OpenCV's viz module
void visualizePointCloud(const vector<Point3f>& points3d) {
    viz::Viz3d window("3D Point Cloud");
    viz::WCloud cloud_widget(points3d, viz::Color::green());
    window.showWidget("Cloud", cloud_widget);
    window.spin();
}

int main() {
    // Load camera calibration data
    Mat camera_matrix, dist_coeffs;
    if (!loadCameraCalibration("camera_calibration.yml", camera_matrix, dist_coeffs)) {
        return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open webcam" << endl;
        return -1;
    }

    namedWindow("Feature Detection", WINDOW_AUTOSIZE);

    Mat frame, prev_frame;
    vector<Point2f> prev_corners;
    vector<Point3f> points3d;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Detect calibration target and compute its pose
        vector<Point2f> corners;
        Mat rvec, tvec;
        if (detectCalibrationTarget(frame, camera_matrix, dist_coeffs, corners, rvec, tvec)) {
            if (!prev_corners.empty()) {
                // Triangulate 3D points from previous and current frames
                triangulatePoints(camera_matrix, rvec, tvec, prev_corners, corners, points3d);
            }
            prev_corners = corners;
            prev_frame = frame.clone();
        }

        // Display the frame with detected features
        //detectHarrisCorners(frame); // Task 7: Apply Harris corner detection
        //detectSURFFeatures(frame);  // Task 8: Apply SURF feature detection
        imshow("Feature Detection", frame);

        if (waitKey(30) == 27) break; // Exit on ESC key
    }

    // Visualize the 3D point cloud
    if (!points3d.empty()) {
        visualizePointCloud(points3d);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}