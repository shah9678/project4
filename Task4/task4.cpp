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

            // Print number of corners and first corner's coordinates
            //cout << "Corners detected: " << corner_set.size() << endl;
            // if (!corner_set.empty()) {
            //cout << "First corner at: (" << corner_set[0].x << ", " << corner_set[0].y << ")" << endl;


        // Task 2: Save calibration frames
        char key = (char)waitKey(30);
        if (key == 's' && found) {
            corner_list.push_back(corner_set);

            vector<Vec3f> point_set;
            for (int i = 0; i < CHECKERBOARD[0]; i++) {
                for (int j = 0; j < CHECKERBOARD[1]; j++) {
                    point_set.push_back(Vec3f(j, -i, 0));
                }
            }
            point_list.push_back(point_set);

            drawChessboardCorners(frame, Size(CHECKERBOARD[1], CHECKERBOARD[0]), corner_set, found);

            static int image_count = 0;
            string filename = "calibration_frame_" + to_string(image_count++) + ".jpg";
            imwrite(filename, frame);
            cout << "Saved calibration frame: " << filename << endl;
        }

        if (key == 'c') {
            calibrateCameraFromSavedData(corner_list, point_list, frame.size());
        }

        // --- Task 4: Pose Estimation (solvePnP) ---
        if (found) {
            Mat rvec, tvec;
            solvePnP(object_points, corner_set, camera_matrix, dist_coeffs, rvec, tvec);

            cout << "Pose Estimation:" << endl;
            cout << "Rotation Vector (rvec): " << rvec.t() << endl;
            cout << "Translation Vector (tvec): " << tvec.t() << endl;
        }
        if (key == 27) { // ESC key
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
