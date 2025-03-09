#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// Load the target image
Mat target_image = imread("/Users/aditshah/Desktop/PRCV/Project4/Extensions/test.jpeg", IMREAD_GRAYSCALE);

void detectAndPlaceARObject(Mat& frame) {
    if (target_image.empty()) {
        cerr << "Error: Target image not found!" << endl;
        return;
    }

    Mat gray_frame;
    cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

    // Feature detection using SURF
    Ptr<SURF> detector = SURF::create(400);
    vector<KeyPoint> keypoints_target, keypoints_frame;
    Mat descriptors_target, descriptors_frame;

    detector->detectAndCompute(target_image, noArray(), keypoints_target, descriptors_target);
    detector->detectAndCompute(gray_frame, noArray(), keypoints_frame, descriptors_frame);

    // Feature matching using FLANN
    FlannBasedMatcher matcher;
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(descriptors_target, descriptors_frame, knn_matches, 2);

    vector<DMatch> good_matches;
    for (const auto& match : knn_matches) {
        if (match[0].distance < 0.7 * match[1].distance) {
            good_matches.push_back(match[0]);
        }
    }

    if (good_matches.size() > 10) {
        vector<Point2f> target_pts, frame_pts;
        for (const auto& match : good_matches) {
            target_pts.push_back(keypoints_target[match.queryIdx].pt);
            frame_pts.push_back(keypoints_frame[match.trainIdx].pt);
        }

        // Find homography
        Mat H = findHomography(target_pts, frame_pts, RANSAC);
        if (!H.empty()) {
            vector<Point2f> target_corners = {
                {0.0f, 0.0f},
                {static_cast<float>(target_image.cols), 0.0f},
                {static_cast<float>(target_image.cols), static_cast<float>(target_image.rows)},
                {0.0f, static_cast<float>(target_image.rows)}
            };
            vector<Point2f> frame_corners;
            perspectiveTransform(target_corners, frame_corners, H);

            // Draw the detected target region
            for (int i = 0; i < 4; i++) {
                line(frame, frame_corners[i], frame_corners[(i + 1) % 4], Scalar(0, 255, 0), 3);
            }

            // Draw AR object (a simple virtual cube)
            vector<Point2f> cube_pts = {
                {(frame_corners[0].x + frame_corners[1].x) / 2, frame_corners[0].y - 50},
                {(frame_corners[1].x + frame_corners[2].x) / 2, frame_corners[1].y - 50},
                {(frame_corners[2].x + frame_corners[3].x) / 2, frame_corners[2].y - 50},
                {(frame_corners[3].x + frame_corners[0].x) / 2, frame_corners[3].y - 50}
            };

            for (int i = 0; i < 4; i++) {
                line(frame, frame_corners[i], cube_pts[i], Scalar(255, 0, 0), 2);
                line(frame, cube_pts[i], cube_pts[(i + 1) % 4], Scalar(255, 0, 0), 2);
            }
        }
    }
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open webcam" << endl;
        return -1;
    }

    namedWindow("AR Feature Detection", WINDOW_AUTOSIZE);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        detectAndPlaceARObject(frame);
        imshow("AR Feature Detection", frame);

        if (waitKey(30) == 27) break; // Exit on ESC key
    }

    cap.release();
    destroyAllWindows();
    return 0;
}