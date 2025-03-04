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

// Task 7: Feature Detection using Harris Corners
void detectHarrisCorners(Mat& frame) {
    Mat gray, dst, dst_norm;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    dst = Mat::zeros(frame.size(), CV_32FC1);

    // Detect Harris corners
    cornerHarris(gray, dst, 2, 3, 0.04);
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    for (int i = 0; i < dst_norm.rows; i++) {
        for (int j = 0; j < dst_norm.cols; j++) {
            if ((int)dst_norm.at<float>(i, j) > 150) { // Threshold
                circle(frame, Point(j, i), 3, Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }
}

// Main Function
int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open webcam" << endl;
        return -1;
    }

    namedWindow("Feature Detection", WINDOW_AUTOSIZE);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        detectHarrisCorners(frame); // Task 7: Apply Harris corner detection

        imshow("Feature Detection", frame);

        if (waitKey(30) == 27) break; // Exit on ESC key
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
