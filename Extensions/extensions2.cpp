#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// Structure to hold AR object data
struct ARObject {
    vector<Point3f> points;
    Scalar color;
};

// Camera calibration function
void calibrateCameraFromSavedData(
    const vector<vector<Point2f>>& corner_list,
    const vector<vector<Vec3f>>& point_list,
    Size image_size,
    Mat& camera_matrix,
    Mat& dist_coeffs)
{
    if (corner_list.size() < 5) {
        cout << "Need at least 5 images for calibration. Current: " << corner_list.size() << endl;
        return;
    }

    vector<Mat> rvecs, tvecs;
    camera_matrix = Mat::eye(3, 3, CV_64F);
    dist_coeffs = Mat::zeros(8, 1, CV_64F);
    
    camera_matrix.at<double>(0, 2) = image_size.width / 2.0;
    camera_matrix.at<double>(1, 2) = image_size.height / 2.0;

    calibrateCamera(point_list, corner_list, image_size, camera_matrix, 
                   dist_coeffs, rvecs, tvecs, CALIB_FIX_ASPECT_RATIO);

    FileStorage fs("camera_calibration.yml", FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "distortion_coefficients" << dist_coeffs;
    fs.release();
}

// Harris corner detection
void detectHarrisCorners(Mat& frame, vector<Point2f>& corners, float threshold = 150) {
    Mat gray, dst, dst_norm;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    dst = Mat::zeros(frame.size(), CV_32FC1);
    
    cornerHarris(gray, dst, 2, 3, 0.04);
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    
    corners.clear();
    for (int i = 0; i < dst_norm.rows; i++) {
        for (int j = 0; j < dst_norm.cols; j++) {
            if (dst_norm.at<float>(i, j) > threshold) {
                corners.push_back(Point2f(j, i));
                circle(frame, Point(j, i), 3, Scalar(0, 0, 255), 2);
            }
        }
    }
}

// Draw AR cube
void drawARObject(Mat& frame, const vector<Point2f>& imagePoints, 
                 const Mat& cameraMatrix, const Mat& distCoeffs) {
    ARObject cube;
    cube.points = {
        Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(1, 1, 0), Point3f(0, 1, 0),  // bottom
        Point3f(0, 0, 1), Point3f(1, 0, 1), Point3f(1, 1, 1), Point3f(0, 1, 1)   // top
    };
    cube.color = Scalar(0, 255, 0);

    vector<Point2f> projectedPoints;
    Mat rvec, tvec;
    solvePnP(cube.points, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    projectPoints(cube.points, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

    // Draw cube edges
    for (int i = 0; i < 4; i++) {
        line(frame, projectedPoints[i], projectedPoints[(i + 1) % 4], cube.color, 2);
        line(frame, projectedPoints[i + 4], projectedPoints[(i + 1) % 4 + 4], cube.color, 2);
        line(frame, projectedPoints[i], projectedPoints[i + 4], cube.color, 2);
    }
}

// Modify target appearance
void modifyTarget(Mat& frame, const vector<Point2f>& corners) {
    if (corners.size() >= 4) {
        vector<Point> contour(corners.begin(), corners.end());
        Mat mask = Mat::zeros(frame.size(), CV_8UC1);
        fillConvexPoly(mask, contour, Scalar(255));
        frame.setTo(Scalar(100, 100, 255), mask);  // Fill with pink color
    }
}

// Process frame with multiple targets
void processFrame(Mat& frame, const Mat& cameraMatrix, const Mat& distCoeffs) {
    vector<Point2f> corners;
    detectHarrisCorners(frame, corners);

    if (corners.size() >= 4) {
        // Sort corners to get consistent ordering (top-left, top-right, bottom-right, bottom-left)
        sort(corners.begin(), corners.end(), [](Point2f a, Point2f b) { return a.y < b.y; });
        vector<Point2f> topCorners(corners.begin(), corners.begin() + 2);
        vector<Point2f> bottomCorners(corners.end() - 2, corners.end());
        sort(topCorners.begin(), topCorners.end(), [](Point2f a, Point2f b) { return a.x < b.x; });
        sort(bottomCorners.begin(), bottomCorners.end(), [](Point2f a, Point2f b) { return a.x < b.x; });

        vector<Point2f> orderedCorners = {topCorners[0], topCorners[1], 
                                        bottomCorners[1], bottomCorners[0]};

        drawARObject(frame, orderedCorners, cameraMatrix, distCoeffs);
        modifyTarget(frame, orderedCorners);
    }
}

int main() {
    // Test multiple cameras
    vector<int> cameraIds = {0, 1};  // Add more camera IDs as needed
    map<int, pair<Mat, Mat>> calibrations;

    for (int id : cameraIds) {
        VideoCapture cap(id);
        if (!cap.isOpened()) continue;

        vector<vector<Point2f>> corner_list;
        vector<vector<Vec3f>> point_list;
        Mat frame, cameraMatrix, distCoeffs;
        
        // Collect calibration data (simplified - in practice, you'd want more images)
        for (int i = 0; i < 5; i++) {
            cap >> frame;
            vector<Point2f> corners;
            detectHarrisCorners(frame, corners);
            if (corners.size() >= 4) {
                corner_list.push_back(corners);
                vector<Vec3f> points(corners.size());
                for (size_t j = 0; j < corners.size(); j++) {
                    points[j] = Vec3f(j % 2, j / 2, 0);
                }
                point_list.push_back(points);
            }
            waitKey(500);
        }
        
        calibrateCameraFromSavedData(corner_list, point_list, frame.size(), 
                                   cameraMatrix, distCoeffs);
        calibrations[id] = {cameraMatrix, distCoeffs};
    }

    // Main processing loop with selected camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open webcam" << endl;
        return -1;
    }

    Mat cameraMatrix = calibrations[0].first;
    Mat distCoeffs = calibrations[0].second;

    // Process static image (optional)
    Mat staticImage = imread("target_photo.jpg");
    if (!staticImage.empty()) {
        processFrame(staticImage, cameraMatrix, distCoeffs);
        imwrite("output_static.jpg", staticImage);
    }

    // Video processing
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        processFrame(frame, cameraMatrix, distCoeffs);
        imshow("AR Scene", frame);

        if (waitKey(30) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}