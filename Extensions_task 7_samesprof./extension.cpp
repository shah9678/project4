#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

const int CHESSBOARD_WIDTH = 6;  // Number of inner corners per row
const int CHESSBOARD_HEIGHT = 4; // Number of inner corners per column
const float SQUARE_SIZE = 2.5f;  // Size of a square in real-world units (e.g., cm)

// Load camera parameters
bool loadCameraCalibration(Mat &cameraMatrix, Mat &distCoeffs) {
    FileStorage fs("camera_calibration.yml", FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Error: Unable to load camera calibration file!" << endl;
        return false;
    }
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();
    return true;
}

// Define 3D points of the chessboard corners
vector<Point3f> generateChessboard3DPoints() {
    vector<Point3f> objectPoints;
    for (int i = 0; i < CHESSBOARD_HEIGHT; i++) {
        for (int j = 0; j < CHESSBOARD_WIDTH; j++) {
            objectPoints.push_back(Point3f(j * SQUARE_SIZE, i * SQUARE_SIZE, 0));
        }
    }
    return objectPoints;
}

// Define 3D points for overlaying a cube
vector<Point3f> generate3DCube() {
    float size = 5.0f;
    return {
        {0, 0, 0}, {size, 0, 0}, {size, size, 0}, {0, size, 0}, // Base square
        {0, 0, -size}, {size, 0, -size}, {size, size, -size}, {0, size, -size} // Top square
    };
}

// Define 3D points for overlaying cylinders
vector<vector<Point3f>> generate3DCylinders() {
    vector<vector<Point3f>> cylinders;
    int numCylinders = 4;
    float radius = 1.5f, height = 4.0f;
    int numSegments = 10; // Approximate a circle

    for (int c = 0; c < numCylinders; c++) {
        vector<Point3f> cylinder;
        float cx = (c % 2) * 6.0f; // Offset for positioning
        float cy = (c / 2) * 6.0f;
        for (int i = 0; i < numSegments; i++) {
            float angle = 2 * CV_PI * i / numSegments;
            float x = cx + radius * cos(angle);
            float y = cy + radius * sin(angle);
            cylinder.push_back(Point3f(x, y, 0)); // Base
            cylinder.push_back(Point3f(x, y, -height)); // Top
        }
        cylinders.push_back(cylinder);
    }
    return cylinders;
}

// Draw the projected cube
void draw3DCube(Mat &frame, vector<Point2f> &imagePoints) {
    vector<vector<int>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Base edges
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Top edges
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Vertical edges
    };

    for (auto &edge : edges) {
        line(frame, imagePoints[edge[0]], imagePoints[edge[1]], Scalar(255, 0, 0), 2);
    }
}

// Draw projected cylinders
void draw3DCylinders(Mat &frame, vector<vector<Point2f>> &cylinderPoints) {
    for (auto &cylinder : cylinderPoints) {
        for (size_t i = 0; i < cylinder.size() / 2; i++) {
            line(frame, cylinder[i * 2], cylinder[i * 2 + 1], Scalar(0, 255, 255), 2);
        }
    }
}

// Main function
int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to access webcam" << endl;
        return -1;
    }

    Mat cameraMatrix, distCoeffs;
    if (!loadCameraCalibration(cameraMatrix, distCoeffs)) {
        return -1;
    }

    namedWindow("Augmented Reality", WINDOW_AUTOSIZE);

    vector<Point3f> chessboard3DPoints = generateChessboard3DPoints();
    vector<Point3f> cube3DPoints = generate3DCube();
    vector<vector<Point3f>> cylinders3DPoints = generate3DCylinders();

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        vector<Point2f> corners;
        bool found = findChessboardCorners(frame, Size(CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), corners,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            drawChessboardCorners(frame, Size(CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), corners, found);

            Mat rvec, tvec;
            solvePnP(chessboard3DPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

            // Project 3D cube onto the image
            vector<Point2f> projectedCubePoints;
            projectPoints(cube3DPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedCubePoints);
            draw3DCube(frame, projectedCubePoints);

            // Project 3D cylinders onto the image
            vector<vector<Point2f>> projectedCylinderPoints;
            for (auto &cylinder : cylinders3DPoints) {
                vector<Point2f> projectedCylinder;
                projectPoints(cylinder, rvec, tvec, cameraMatrix, distCoeffs, projectedCylinder);
                projectedCylinderPoints.push_back(projectedCylinder);
            }
            draw3DCylinders(frame, projectedCylinderPoints);
        }

        imshow("Augmented Reality", frame);
        if (waitKey(30) == 27) break; // Press ESC to exit
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
