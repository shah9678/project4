# **Project 4- README**

## **Project Information**
**Project Title:** Camera Calibration and Augmented Reality System

**Group Members:** Adit Shah & Jheel Kamdar

**Submission Date:** 13 March 2025

## **Project Overview**
This project focuses on building a camera calibration and augmented reality (AR) system using OpenCV. The primary objective is to calibrate the camera using a checkerboard pattern and then utilize this calibration to estimate the camera's position relative to the target in real-time. The final system enables real-time visualization of both coordinate axes and virtual objects over the detected target, enhancing the AR experience.


---

## **Video Demonstration**
N/A

---

## **Development Environment**
- **Operating System:** macOS
- **IDE Used:** Visual Studio Code
- **Compiler:** g++ / Clang 
- **Dependencies:**  
  - OpenCV (Version 4.x)
  - CMake (Version 3.x or above)

---

## **Instructions to Run the Code**
### **Running the System on Video or Images**
### **Step 1: Clone or Download the Repository**  
```bash
git clone git@github.com:shah9678/project4.git
cd project4
```
### **Step 2:Compile the Code Using CMake**
   ```
   mkdir build
   cd build
   cmake ..
   make
```
---

### **Controls in the Program:**  
- **'s'** - Save a calibration frame when the checkerboard is detected.  
- **'c'** - Run the camera calibration using saved frames.  
- **'ESC'** - Exit the program.

---

## **Testing Extensions**
- **3D Projection:** Ensure the camera calibration has been completed. During the runtime, if the checkerboard is detected, the 3D coordinate axes and virtual objects (like a pyramid) will be projected onto the target.  
- **Pose Validation:** Move the camera around the checkerboard to observe how the virtual object and axes remain correctly aligned.  
- **Error Handling:** The program will notify if the checkerboard is not detected or if calibration hasn't been performed yet.

---

## **Time Travel Days**

**Time Travel Days Used:** 0  
We have completed and are submitting this project within the original deadline, without the use of any additional time travel days.

---

## **Known Issues**
- Slight inaccuracies in projection can occur if the lighting is poor or if the checkerboard is partially obscured.  
- The system performs best when the checkerboard is clearly visible and flat.  
- Calibration accuracy might reduce if fewer than five calibration frames are provided.  
- The 3D axes may slightly misalign if the calibration data is outdated or incorrect.

---

---

## **Additional Notes**
- Ensure that the OpenCV library is correctly installed and linked with your development environment.  
- If errors occur with the camera feed, check if the webcam is correctly configured and accessible.  
- The `camera_calibration.yml` file is generated after running the calibration step and must be present for pose estimation tasks.  
- For optimal accuracy, ensure that the checkerboard is held flat and fully visible to the camera during calibration and testing.  
- If pose estimation seems incorrect, consider recalibrating the camera with more varied angles and distances.  
- Virtual object projections might misalign if the camera or target moves too quickly, leading to detection errors.

---
