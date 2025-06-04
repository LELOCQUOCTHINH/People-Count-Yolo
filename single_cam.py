#!/usr/bin/env python3
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import threading
from queue import Queue, Empty  # Correct import for Empty exception
import signal
import sys
import psutil
import logging
import argparse
from postTelemetry_mqtt_tb import MQTTThingsBoardClient

# Setup logger
logging.basicConfig(level=logging.DEBUG, format="[DEBUG] %(message)s")
logger = logging.getLogger(__name__)

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Shared variables for threading
out_count = 0
in_count = 0
tracked_ids = {}
trackers = []
tracker_id = 0
frame_queue = Queue(maxsize=1)  # Queue for passing frames to detection thread
result_queue = Queue(maxsize=1)  # Queue for passing detection results back
lock = threading.Lock()  # Lock for shared variables
cpu_usages = []
memory_usages = []
temperatures = []
fps_values = []
running = True
tb_client = None
source = None

# Signal handler for Ctrl+C
def signal_handler(sig, frame):
    print("Ctrl+C detected, cleaning up...")
    global running, tb_client, source
    running = False
    if source:
        if isinstance(source, Picamera2):
            source.stop()
        else:
            source.release()
    if tb_client:
        tb_client.disconnect()
    cv2.destroyAllWindows()
    # Print average resource usage and FPS
    if cpu_usages:
        print(f"Average CPU Usage: {sum(cpu_usages)/len(cpu_usages):.2f}%")
    if memory_usages:
        print(f"Average Memory Usage: {sum(memory_usages)/len(memory_usages):.2f}%")
    if temperatures:
        print(f"Average Temperature: {sum(temperatures)/len(temperatures):.2f}°C")
    if fps_values:
        print(f"Average FPS: {sum(fps_values)/len(fps_values):.2f}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def create_tracker(tracker_type="KCF"):
    if tracker_type == "KCF":
        try:
            if hasattr(cv2, 'TrackerKCF_create'):
                return cv2.TrackerKCF_create()
            else:
                raise AttributeError("TrackerKCF_create not found in cv2. Ensure opencv-contrib-python is installed.")
        except AttributeError as e:
            logger.error(f"Tracker creation failed: {e}")
            return None
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")

def is_above_line(x, y, line_start, line_end):
    vx = line_end[0] - line_start[0]
    vy = line_end[1] - line_start[1]
    px = x - line_start[0]
    py = y - line_start[1]
    return px * vy - py * vx > 0

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = inter_area / (box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
    return iou

def is_valid_bbox(bbox, width, height):
    x, y, w, h = bbox
    min_bbox_size = 5
    if w < min_bbox_size or h < min_bbox_size:
        return False
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return False
    return True

# Detection thread function
def detection_thread():
    global trackers, tracker_id, tracked_ids
    max_trackers = 10
    iou_threshold = 0.3
    frame_skip = 5  # Increased to reduce detection frequency
    frame_count = 0

    while running:
        try:
            frame = frame_queue.get(timeout=0.1)
        except Empty:
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        height, width, _ = frame.shape

        # YOLO detection with increased input size
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (96, 96), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        boxes = []
        confidences = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.05:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    bbox = (x, y, w, h)
                    if is_valid_bbox(bbox, width, height):
                        boxes.append(bbox)
                        confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.05, nms_threshold=0.5)

        with lock:
            if len(indices) > 0:
                for i in indices.flatten():
                    bbox = boxes[i]
                    if len(trackers) < max_trackers:
                        best_iou = 0
                        for _, tracked_bbox, _ in trackers:
                            iou = calculate_iou(bbox, tracked_bbox)
                            if iou > best_iou:
                                best_iou = iou
                        if best_iou < iou_threshold:
                            tracker = create_tracker("KCF")
                            if tracker is None:
                                logger.error("Skipping tracker creation due to missing TrackerKCF_create")
                                continue
                            try:
                                tracker.init(frame, bbox)
                                trackers.append((tracker, bbox, tracker_id))
                                tracked_ids[tracker_id] = None
                                logger.debug(f"New tracker created with ID {tracker_id}")
                                tracker_id += 1
                            except Exception as e:
                                logger.error(f"Failed to initialize tracker: {e}")

def monitor_resources(tb_client, server_IP, port, token):
    global cpu_usages, memory_usages, temperatures
    last_time = time.time()
    while running:
        current_time = time.time()
        if current_time - last_time >= 10:
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read()) / 1000.0
            except:
                temp = 0.0  # Fallback if temp not available
            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)
            temperatures.append(temp)
            tb_client.send_telemetry(server_IP, port, token, "CPU_usage", round(cpu_usage, 2))
            tb_client.send_telemetry(server_IP, port, token, "memory_usage", round(memory_usage, 2))
            tb_client.send_telemetry(server_IP, port, token, "Temperature", round(temp, 2))
            last_time = current_time
        time.sleep(1)

def main():
    global running, tb_client, out_count, in_count, fps_values, source, trackers
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="picam",
                        help="Input source: video file path or 'picam' for PiCamera")
    parser.add_argument("-s", "--server-IP", type=str, default="",
                        help="ThingsBoard server domain")
    parser.add_argument("-P", "--Port", type=int, default=0,
                        help="MQTT port for ThingsBoard server")
    parser.add_argument("-a", "--token", type=str, default="",
                        help="Device access token for ThingsBoard authentication")
    args = parser.parse_args()

    if not args.server_IP or not args.Port or not args.token:
        print("Error: --server-IP, --Port, and --token are required")
        sys.exit(1)

    # Initialize input source
    if args.input.lower() == "picam":
        source = Picamera2()
        config = source.create_video_configuration(main={"size": (160, 120), "format": "RGB888"})
        source.configure(config)
        source.start()
        print("Using camera input")
        video_fps = 10  # Default for camera
    else:
        source = cv2.VideoCapture(args.input)
        if not source.isOpened():
            print(f"Error: Could not open video file {args.input}")
            sys.exit(1)
        print("Using video file input")
        # Get video frame rate
        video_fps = source.get(cv2.CAP_PROP_FPS)
        print(f"Video FPS: {video_fps}")

    # Initialize ThingsBoard client
    tb_client = MQTTThingsBoardClient()

    # Start the detection thread
    threading.Thread(target=detection_thread, daemon=True).start()

    # Start resource monitoring thread
    monitor_thread = threading.Thread(target=monitor_resources, args=(tb_client, args.server_IP, args.Port, args.token))
    monitor_thread.daemon = True
    monitor_thread.start()

    try:
        last_time = time.time()
        tracker_update_counter = 0
        frame_count = 0
        start_time = time.time()
        fps_start_time = time.time()
        while running:
            # Read frame from the selected input source
            if isinstance(source, Picamera2):
                image = source.capture_array()
            else:
                ret, image = source.read()
                if not ret:
                    print("End of video or error reading frame")
                    running = False
                    # Print average resource usage and FPS
                    if cpu_usages:
                        print(f"Average CPU Usage: {sum(cpu_usages)/len(cpu_usages):.2f}%")
                    if memory_usages:
                        print(f"Average Memory Usage: {sum(memory_usages)/len(memory_usages):.2f}%")
                    if temperatures:
                        print(f"Average Temperature: {sum(temperatures)/len(temperatures):.2f}°C")
                    if fps_values:
                        print(f"Average FPS: {sum(fps_values)/len(fps_values):.2f}")
                    # Print playback speed
                    total_time = time.time() - start_time
                    total_frames = source.get(cv2.CAP_PROP_FRAME_COUNT)
                    expected_time = total_frames / video_fps
                    playback_speed = expected_time / total_time
                    print(f"Playback speed: {playback_speed:.2f}x")
                    break
                image = cv2.resize(image, (160, 120))

            frame_count += 1
            height, width, _ = image.shape
            line_start = (0, height // 2)  # Line at middle (y=60)
            line_end = (width, height // 2)

            # Pass frame to detection thread
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(image.copy())

            # Update trackers every 2 frames to reduce CPU usage
            tracker_update_counter += 1
            if tracker_update_counter % 2 == 0:
                with lock:
                    new_trackers = []
                    if trackers:  # Check if trackers is not empty
                        for tracker, prev_bbox, tid in trackers:
                            ok, new_bbox = tracker.update(image)
                            if ok and is_valid_bbox(new_bbox, width, height):
                                x, y, w, h = [int(v) for v in new_bbox]
                                current_position = (x + w // 2, y + h // 2)
                                prev_position = (prev_bbox[0] + prev_bbox[2] // 2, prev_bbox[1] + prev_bbox[3] // 2)

                                prev_above = is_above_line(prev_position[0], prev_position[1], line_start, line_end)
                                curr_above = is_above_line(current_position[0], current_position[1], line_start, line_end)

                                # Buffer to handle position fluctuations
                                buffer = 5
                                if prev_above and not curr_above and abs(prev_position[1] - line_start[1]) <= buffer and tracked_ids[tid] is None:
                                    in_count += 1
                                    tracked_ids[tid] = "in"
                                    logger.debug(f"Counted IN: Total IN = {in_count}")
                                    tb_client.send_telemetry(args.server_IP, args.Port, args.token, "entered_people", in_count)
                                    tb_client.send_telemetry(args.server_IP, args.Port, args.token, "exited_people", out_count)
                                    tb_client.send_telemetry(args.server_IP, args.Port, args.token, "people_inside", in_count - out_count)
                                elif not prev_above and curr_above and abs(prev_position[1] - line_start[1]) <= buffer and tracked_ids[tid] is None:
                                    out_count += 1
                                    tracked_ids[tid] = "out"
                                    logger.debug(f"Counted OUT: Total OUT = {out_count}")
                                    tb_client.send_telemetry(args.server_IP, args.Port, args.token, "exited_people", out_count)
                                    tb_client.send_telemetry(args.server_IP, args.Port, args.token, "entered_people", in_count)
                                    tb_client.send_telemetry(args.server_IP, args.Port, args.token, "people_inside", in_count - out_count)

                                new_trackers.append((tracker, new_bbox, tid))
                                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                                cv2.putText(image, f'ID: {tid}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                                cv2.circle(image, current_position, 2, (255, 0, 0), -1)
                            else:
                                pass  # Skip failed trackers
                    trackers = new_trackers

            # Draw counting line and counts
            cv2.line(image, line_start, line_end, (0, 0, 255), 1)
            with lock:
                cv2.putText(image, f'OUT: {out_count}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(image, f'IN: {in_count}', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Resize frame for display to lower resolution
            display_frame = cv2.resize(image, (320, 240))
            cv2.imshow('Video' if not isinstance(source, Picamera2) else 'Camera', display_frame)

            # Cap at video FPS or 10 FPS, whichever is lower
            target_fps = min(video_fps, 10) if not isinstance(source, Picamera2) else 10
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed < 1/target_fps:
                time.sleep(1/target_fps - elapsed)
            last_time = current_time

            # Calculate and send FPS every 10 seconds
            elapsed_time = current_time - fps_start_time
            if elapsed_time >= 10:
                fps = frame_count / elapsed_time
                fps_values.append(fps)
                tb_client.send_telemetry(args.server_IP, args.Port, args.token, "FPS", round(fps, 2))
                frame_count = 0
                fps_start_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                # Print average resource usage and FPS
                if cpu_usages:
                    print(f"Average CPU Usage: {sum(cpu_usages)/len(cpu_usages):.2f}%")
                if memory_usages:
                    print(f"Average Memory Usage: {sum(memory_usages)/len(memory_usages):.2f}%")
                if temperatures:
                    print(f"Average Temperature: {sum(temperatures)/len(temperatures):.2f}°C")
                if fps_values:
                    print(f"Average FPS: {sum(fps_values)/len(fps_values):.2f}")
                # Print playback speed
                if not isinstance(source, Picamera2):
                    total_time = time.time() - start_time
                    total_frames = source.get(cv2.CAP_PROP_FRAME_COUNT)
                    expected_time = total_frames / video_fps
                    playback_speed = expected_time / total_time
                    print(f"Playback speed: {playback_speed:.2f}x")
                break

    finally:
        if source:
            if isinstance(source, Picamera2):
                source.stop()
            else:
                source.release()
        if tb_client:
            tb_client.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()