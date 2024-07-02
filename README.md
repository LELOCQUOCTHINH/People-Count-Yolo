
# RTSP Camera People Counting System

This project utilizes the YOLOv4-tiny model for detecting and tracking people using two RTSP streams. It counts people entering and exiting a predefined area from both camera streams.

## Prerequisites

- Python 3.x
- OpenCV
- NumPy
- imutils

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/shubhamshnd/People-Count-Yolo
    cd People-Count-Yolo
    ```

2. Install the required libraries:
    ```bash
    pip install opencv-python numpy imutils
    ```

3. Download the YOLOv4-tiny weights and configuration files:
    - [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)
    - [yolov4-tiny.cfg](https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/cfg/yolov4-tiny.cfg)

    Place these files in the same directory as your script.

## Usage

1. Update the RTSP URLs in the script to match your camera streams.

2. Run the script:
    ```bash
    python single_cam.py
    ```

3. Press `ctrl + c` to quit the application.

## Description

The script performs the following steps:
- Loads the YOLOv4-tiny model.
- Initializes parameters for both camera streams.
- Opens the RTSP streams with proper codecs and increases buffer size.
- Processes frames from both cameras, detecting people using YOLOv4-tiny.
- Tracks people across frames using the MIL tracker.
- Counts the number of people entering and exiting a predefined area for both cameras.

## Demo

For a demonstration of the people counting system, check out this [YouTube video](https://www.youtube.com/watch?v=dC4sn8eqSxU).

## Contact

For any questions or issues, please contact [shubhamshindesunil@gmail.com](mailto:shubhamshindesunil@gmail.com).
[My Linkedin](https://www.linkedin.com/in/shubham-shnd/)
