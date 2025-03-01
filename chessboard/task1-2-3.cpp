#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Task 3: Camera Calibration Function
void calibrateCameraFromSavedData(
    const vector<vector<Point2f>>& corner_list,
    const vector<vector<Vec3f>>& point_list,
    Size image_size)
{
    if (corner_list.size() < 5) {
        cout << "At least 5 calibration images are required. Currently have: " << corner_list.size() << endl;
        return;
    }

    cout << "Starting camera calibration using " << corner_list.size() << " images..." << endl;

    // Define output calibration parameters
    Mat camera_matrix = Mat::eye(3, 3, CV_64F);
    Mat dist_coeffs = Mat::zeros(8, 1, CV_64F);  // Optional: Use 5 coefficients if you want simpler distortion

    // Initial guess for camera matrix
    camera_matrix.at<double>(0, 0) = 1;                // fx
    camera_matrix.at<double>(1, 1) = 1;                // fy
    camera_matrix.at<double>(0, 2) = image_size.width / 2.0;  // cx
    camera_matrix.at<double>(1, 2) = image_size.height / 2.0; // cy

    cout << "Initial camera matrix (pre-calibration):" << endl << camera_matrix << endl;

    // Calibration flags
    int flags = CALIB_FIX_ASPECT_RATIO;  // Assume square pixels (fx = fy)

    // Perform calibration
    vector<Mat> rvecs, tvecs;
    double reprojection_error = calibrateCamera(
        point_list, corner_list, image_size,
        camera_matrix, dist_coeffs,
        rvecs, tvecs,
        flags
    );

    cout << "Calibration complete." << endl;
    cout << "Final camera matrix:" << endl << camera_matrix << endl;
    cout << "Distortion coefficients:" << endl << dist_coeffs.t() << endl;
    cout << "Reprojection error: " << reprojection_error << " pixels" << endl;

    // Save calibration to file
    FileStorage fs("camera_calibration.yml", FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "distortion_coefficients" << dist_coeffs;
    fs.release();
    cout << "Saved calibration to 'camera_calibration.yml'" << endl;

    // Print rotations and translations for each image (optional for debugging or reporting)
    for (size_t i = 0; i < rvecs.size(); ++i) {
        cout << "Image " << i << " rotation vector: " << rvecs[i].t() << endl;
        cout << "Image " << i << " translation vector: " << tvecs[i].t() << endl;
    }
}

int main() {
    const int CHECKERBOARD[2] = {6, 9};  // Checkerboard size (rows, cols)

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open webcam" << endl;
        return -1;
    }

    namedWindow("Checkerboard Detection", WINDOW_AUTOSIZE);

    vector<vector<Point2f>> corner_list;
    vector<vector<Vec3f>> point_list;

    while (true) {
        Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Point2f> corner_set;

        // Detect corners
        bool found = findChessboardCorners(gray, Size(CHECKERBOARD[1], CHECKERBOARD[0]), corner_set,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cornerSubPix(gray, corner_set, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.01));

            drawChessboardCorners(frame, Size(CHECKERBOARD[1], CHECKERBOARD[0]), corner_set, found);
        }

        imshow("Checkerboard Detection", frame);  // This MUST be inside the loop

        char key = waitKey(30);
        if (key == 27) break;  // 'Esc' to exit

        if (key == 's') {
            if (found) {
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
            } else {
                cout << "No checkerboard detected - nothing saved." << endl;
            }
        }

        if (key == 'c') {
            calibrateCameraFromSavedData(corner_list, point_list, frame.size());
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
