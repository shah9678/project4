#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // Define checkerboard dimensions
    const int CHECKERBOARD[2] = {6, 9}; // 6 rows, 9 columns of internal corners
    
    // Open webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open webcam" << endl;
        return -1;
    }
    
    namedWindow("Checkerboard Detection", WINDOW_AUTOSIZE);
    
    // Vectors to store detected corners and 3D world points
    vector<vector<Point2f>> corner_list;
    vector<vector<Vec3f>> point_list;
    
    while (true) {
        Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;
        
        // Convert to grayscale
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // Vector to store detected corners
        vector<Point2f> corner_set;
        
        // Detect checkerboard corners
        bool found = findChessboardCorners(gray, Size(CHECKERBOARD[1], CHECKERBOARD[0]), corner_set,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        
        if (found) {
            // Refine detected corner locations
            cornerSubPix(gray, corner_set, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.01));
            
            // Draw corners on the frame
            drawChessboardCorners(frame, Size(CHECKERBOARD[1], CHECKERBOARD[0]), corner_set, found);
            
            // Print number of corners and first corner's coordinates
            //cout << "Corners detected: " << corner_set.size() << endl;
            if (!corner_set.empty()) {
                //cout << "First corner at: (" << corner_set[0].x << ", " << corner_set[0].y << ")" << endl;
            }
        }
        
        imshow("Checkerboard Detection", frame);
        
        char key = waitKey(30);
        if (key == 27) break; // Exit on 'Esc' key
        
        if (key == 's' && found) {
    // Store detected corners
            corner_list.push_back(corner_set);
            
            // Generate 3D world points
            vector<Vec3f> point_set;
            for (int i = 0; i < CHECKERBOARD[0]; i++) {
                for (int j = 0; j < CHECKERBOARD[1]; j++) {
                    point_set.push_back(Vec3f(j, -i, 0));
                }
            }
            
            point_list.push_back(point_set);

            // Save the calibration frame
            static int image_count = 0;
            string filename = "calibration_frame_" + to_string(image_count++) + ".jpg";
            imwrite(filename, frame);
            cout << "Saved calibration frame: " << filename << endl;
        }
    }
    
    cap.release();
    destroyAllWindows();
    return 0;
}
