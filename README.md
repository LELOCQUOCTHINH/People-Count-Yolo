# People Counting on Bus Project Using YOLO Tiny V4

![Screenshot 2025-06-03 123444](https://github.com/user-attachments/assets/f6d51e63-36d4-4987-87ba-29fd4c0c61da)

## Overview
This project implements a people counting system for buses using a Raspberry Pi Zero 2W as the edge computing device. It leverages the YOLOv4-tiny model for person detection, a 5MP Pi Camera for capturing video frames, and MQTT to send processed data to a ThingsBoard dashboard for real-time monitoring. The system tracks people entering and exiting the bus, calculates the number of people inside, and monitors system resources (CPU, memory, and temperature).

## Repository Contents
- **single_cam.py**: Main script for processing video from a single camera (Pi Camera or video file). It performs person detection, tracking, counting, and sends telemetry data to ThingsBoard.
- **cam.py**: Script for processing video from two RTSP camera streams. It performs person detection and tracking for dual-camera setups but does not include telemetry or resource monitoring.
- **postTelemetry_mqtt_tb.py**: Utility script for handling MQTT communication with the ThingsBoard server to send telemetry data.

## Features
- **Person Detection**: Uses YOLOv4-tiny for efficient person detection on resource-constrained devices like the Raspberry Pi Zero 2W.
- **Tracking**: Implements KCF (Kernelized Correlation Filter) tracking in `single_cam.py` and MIL (Multiple Instance Learning) tracking in `cam.py` to follow detected persons across frames.
- **Counting Logic**: Counts people crossing a virtual line to determine entries ("IN") and exits ("OUT") from the bus.
- **Telemetry**: Sends real-time data (entry/exit counts, people inside, CPU usage, memory usage, temperature, and FPS) to a ThingsBoard dashboard.
- **Resource Monitoring**: Tracks CPU, memory, and temperature usage on the Raspberry Pi Zero 2W for performance optimization.
- **Frame Optimization**: Processes frames at a reduced resolution (160x120 for `single_cam.py`, 600px width for `cam.py`) and skips frames to optimize performance on the Raspberry Pi Zero 2W.

## Hardware Preferences
- **Raspberry Pi Zero 2W**: Acts as the edge computing device.
- **Pi Camera Module (5MP)**: Captures video frames for person detection.
- **Internet Connection**: Required for sending telemetry data to ThingsBoard.

## Software Requirements
- **Python 3.7+**
- **OpenCV (with contrib)**: For computer vision tasks and tracking.
  ```bash
  pip install opencv-contrib-python
  ```
- **Picamera2**: For interfacing with the Pi Camera.
  ```bash
  pip install picamera2
  ```
- **Paho MQTT**: For communication with ThingsBoard.
  ```bash
  pip install paho-mqtt
  ```
- **Psutil**: For resource monitoring.
  ```bash
  pip install psutil
  ```
- **Imutils**: For image processing utilities (used in `cam.py`).
  ```bash
  pip install imutils
  ```
- **YOLOv4-tiny Model Files**:
  - Download `yolov4-tiny.weights` and `yolov4-tiny.cfg` from the official YOLO repository or a trusted source.
  - Place them in the same directory as the scripts.
  - Or you can use the available `yolov4-tiny.weights` and `yolov4-tiny.cfg` of this repo.

## Setup Instructions
1. **Install Dependencies**:
   Ensure all required Python packages are installed using the commands above.
2. **Configure ThingsBoard**:
   - Set up a ThingsBoard server (e.g., `app.coreiot.io` or a local instance).
   - Create a device in ThingsBoard and obtain the **device access token**.
   - Note the server IP and MQTT port (e.g., 1883).
3. **Prepare YOLO Model**:
   - Download your `yolov4-tiny.weights` and `yolov4-tiny.cfg` version.
   - Place them in the project directory.
   - Or you can use the available `yolov4-tiny.weights` and `yolov4-tiny.cfg` of this repo.
4. **Connect Pi Camera**:
   - Ensure the 5MP Pi Camera is properly connected to the Raspberry Pi Zero 2W.
   - Enable the camera interface in `raspi-config`.
5. **Run the Script**:
   - For single-camera operation with the Pi Camera:
     ```bash
     python3 single_cam.py --server-IP <THINGSBOARD_IP> --Port <MQTT_PORT> --token <DEVICE_TOKEN>
     ```
   - For testing with a video file:
     ```bash
     python3 single_cam.py --input <VIDEO_FILE_PATH> --server-IP <THINGSBOARD_IP> --Port <MQTT_PORT> --token <DEVICE_TOKEN>
     ```
   - For dual-camera RTSP streams (using `cam.py`):
     ```bash
     python3 cam.py
     ```
     Note: Replace `"Your RTSP URL"` in `cam.py` with actual RTSP URLs for your cameras.
6. **View Dashboard**:
   - Access the ThingsBoard dashboard to monitor:
     - Number of people entering (`entered_people`)
     - Number of people exiting (`exited_people`)
     - Current people inside (`people_inside`)
     - System metrics (CPU usage, memory usage, temperature, FPS)
   - Example:
    ![Screenshot 2025-06-03 211056](https://github.com/user-attachments/assets/cbb0fb58-f331-4a4d-9ebc-9c0204fe77c5)
    ![Screenshot 2025-06-03 211103](https://github.com/user-attachments/assets/ce96f62d-67c0-4cc1-a9c0-4bd79099c53a)

    - You can view the example dashboard via this link: [My Dashboard](https://app.coreiot.io/dashboard/5eef5c50-3ca9-11f0-aae0-0f85903b3644?publicId=00e331c0-f1ec-11ef-87b5-21bccf7d29d5).
    - if you want to manipulate with [My Dashboard](https://app.coreiot.io/dashboard/5eef5c50-3ca9-11f0-aae0-0f85903b3644?publicId=00e331c0-f1ec-11ef-87b5-21bccf7d29d5), you can run this command:
      
    ```bash
     python3 single_cam.py --input <test1.mp4> --server-IP app.coreiot.io --Port 1883 --token I1WYm7V1FMBsKgBLMJVL
    ```

    (For testing with test1.mp4).
    
## Usage Notes
- **Single Camera (`single_cam.py`)**:
  - Optimized for Raspberry Pi Zero 2W with low-resolution processing (160x120).
  - Uses threading for detection and resource monitoring to improve performance.
  - Sends telemetry data to ThingsBoard for real-time monitoring.
  - Supports both Pi Camera and video file inputs.
  - Press `q` to quit or use `Ctrl+C` to gracefully exit and view average resource usage and FPS.
- **Dual Camera (`cam.py`)**:
  - Designed for two RTSP camera streams.
  - Uses MIL tracking instead of KCF and does not include telemetry or resource monitoring.
  - Suitable for testing or environments with more powerful hardware.
- **Performance Considerations**:
  - The Raspberry Pi Zero 2W is resource-constrained; `single_cam.py` is optimized for this hardware with frame skipping and low resolution.
  - Adjust `frame_skip` and resolution in the scripts if needed for performance tuning.
- **Counting Logic**:
  - A virtual line is drawn across the frame (horizontal in `single_cam.py`, slanted in `cam.py`).
  - People crossing the line are counted as "IN" or "OUT" based on their direction relative to the line.
  - A buffer zone in `single_cam.py` reduces false counts due to position fluctuations.

## Limitations
- **Single Camera**: The system assumes a single entry/exit point for accurate counting. Multiple entry points may require additional cameras or logic.
- **YOLOv4-tiny**: While lightweight, it may miss detections in low-light or crowded scenes. Consider tuning confidence thresholds or using a more robust model if hardware allows.
- **Raspberry Pi Zero 2W**: Limited processing power may lead to lower FPS. Adjust frame skip and resolution for better performance.
- **RTSP Streams (`cam.py`)**: Requires stable network connectivity and proper RTSP URLs.

## Future Improvements
- Add support for multiple entry/exit points with additional cameras.
- Implement more robust tracking algorithms (e.g., DeepSORT) for improved accuracy.
- Enhance error handling for network disconnections in MQTT communication.
- Optimize YOLO model parameters for better detection in challenging conditions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or contributions, please contact [thinhle.hardware@gmail.com](mailto:thinhle.hardware@gmail.com) or
[My Linkedin](https://www.linkedin.com/in/lelocquocthinh/) or open an issue on the repository.
