#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Task 3: Camera Calibration Function (keep this as you already have it)
void calibrateCameraFromSavedData(
    const vector<vector<Point2f>>& corner_list,
    const vector<vector<Vec3f>>& point_list,
    Size image_size)
{
    if (corner_list.size() < 5) {
        cout << "At least 5 calibration images are required. Currently have: " << corner_list.size() << endl;
        return;
    }

    Mat camera_matrix = Mat::eye(3, 3, CV_64F);
    Mat dist_coeffs = Mat::zeros(8, 1, CV_64F);
    vector<Mat> rvecs, tvecs;

    camera_matrix.at<double>(0, 0) = 1;
    camera_matrix.at<double>(1, 1) = 1;
    camera_matrix.at<double>(0, 2) = image_size.width / 2.0;
    camera_matrix.at<double>(1, 2) = image_size.height / 2.0;

    double reprojection_error = calibrateCamera(point_list, corner_list, image_size, camera_matrix, dist_coeffs, rvecs, tvecs, CALIB_FIX_ASPECT_RATIO);

    FileStorage fs("camera_calibration.yml", FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "distortion_coefficients" << dist_coeffs;
    fs.release();

    cout << "Calibration complete and saved." << endl;
}

// Task 5: Project 3D Axes on All Four Corners
void project3DPoints(Mat& frame, const Mat& camera_matrix, const Mat& dist_coeffs, const Mat& rvec, const Mat& tvec) {
    // Define the 3D axes points (X, Y, Z axes)
    vector<Point3f> axes_points = {
        {0, 0, 0}, {3, 0, 0}, {0, 3, 0}, {0, 0, -3} // Origin, X, Y, Z axes
    };

    // Define the four corners of the chessboard in 3D space
    vector<Point3f> corner_points = {
        {0, 0, 0}, {9, 0, 0}, {9, -6, 0}, {0, -6, 0} // Four corners of the 7x10 chessboard
    };

    // Project and draw axes for each corner
    for (const auto& corner : corner_points) {
        // Translate the axes to the current corner
        vector<Point3f> translated_axes;
        for (const auto& axis : axes_points) {
            translated_axes.push_back(corner + axis);
        }

        // Project the translated axes onto the image plane
        vector<Point2f> projected_axes;
        projectPoints(translated_axes, rvec, tvec, camera_matrix, dist_coeffs, projected_axes);

        // Draw the 3D axes (X: red, Y: green, Z: blue)
        line(frame, projected_axes[0], projected_axes[1], Scalar(0, 0, 255), 2); // X-axis (red)
        line(frame, projected_axes[0], projected_axes[2], Scalar(0, 255, 0), 2); // Y-axis (green)
        line(frame, projected_axes[0], projected_axes[3], Scalar(255, 0, 0), 2); // Z-axis (blue)
    }
}

// Task 6: Construct and project a 3D virtual object
void projectVirtualObject(Mat& frame, const Mat& camera_matrix, const Mat& dist_coeffs, const Mat& rvec, const Mat& tvec) {
    vector<Point3f> object_points = {
        {0, 0, 0}, {2, 0, 0}, {1, 2, 0}, {1, 1, 2} // Base triangle and top point of pyramid
    };

    vector<pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 0}, // Base triangle edges
        {0, 3}, {1, 3}, {2, 3}  // Sides of the pyramid
    };

    vector<Point2f> projected_points;
    projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, projected_points);

    for (const auto& edge : edges) {
        line(frame, projected_points[edge.first], projected_points[edge.second], Scalar(255, 255, 0), 2); // Yellow lines
    }
}

// Main
int main() {
    const int CHECKERBOARD[2] = {6, 9};  // 6 rows, 9 columns

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open webcam" << endl;
        return -1;
    }

    namedWindow("Checkerboard Detection", WINDOW_AUTOSIZE);

    vector<vector<Point2f>> corner_list;
    vector<vector<Vec3f>> point_list;

    // --- Load Calibration for Task 4 ---
    Mat camera_matrix, dist_coeffs;
    FileStorage fs("camera_calibration.yml", FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Error: Could not open camera_calibration.yml. Run calibration first!" << endl;
        return -1;
    }
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> dist_coeffs;
    fs.release();
    cout << "Loaded camera matrix and distortion coefficients." << endl;

    // --- Pre-compute 3D world points (same as in calibration) ---
    vector<Vec3f> object_points;
    for (int i = 0; i < CHECKERBOARD[0]; i++) {
        for (int j = 0; j < CHECKERBOARD[1]; j++) {
            object_points.push_back(Vec3f(j, -i, 0));
        }
    }

    while (true) {
        Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Point2f> corner_set;

        // Task 1: Detect corners
        bool found = findChessboardCorners(gray, Size(CHECKERBOARD[1], CHECKERBOARD[0]), corner_set,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cornerSubPix(gray, corner_set, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.01));

            drawChessboardCorners(frame, Size(CHECKERBOARD[1], CHECKERBOARD[0]), corner_set, found);
        }

        imshow("Checkerboard Detection", frame);

        if (found) {
            Mat rvec, tvec;
            solvePnP(object_points, corner_set, camera_matrix, dist_coeffs, rvec, tvec);

            project3DPoints(frame, camera_matrix, dist_coeffs, rvec, tvec); // Draw 3D axes on all four corners
            projectVirtualObject(frame, camera_matrix, dist_coeffs, rvec, tvec); // Task 6 Projection
            imshow("3D Projection", frame);
        }

        if (waitKey(30) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}