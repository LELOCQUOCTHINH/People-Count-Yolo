import cv2
import numpy as np
import imutils

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize parameters for both cameras
line_start_cam1 = (180, 160)
line_end_cam1 = (550, 400)
line_start_cam2 = (210, 190)
line_end_cam2 = (600, 450)

out_count_cam1 = 0
in_count_cam1 = 0
out_count_cam2 = 0
in_count_cam2 = 0

# Open the RTSP streams with proper codecs and increase buffer size
cap1 = cv2.VideoCapture("Your RTSP URL", cv2.CAP_FFMPEG)
cap2 = cv2.VideoCapture("Your RTSP URL", cv2.CAP_FFMPEG)

# Increase the buffer size
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 3)
cap2.set(cv2.CAP_PROP_BUFFERSIZE, 3)

def create_tracker(tracker_type="MIL"):
    if tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")

def is_above_line(x, y, line_start, line_end):
    return (line_end[1] - line_start[1]) * x + (line_start[0] - line_end[0]) * y + (line_end[0] * line_start[1] - line_start[0] * line_end[1])

frame_skip = 4  # Process every 4th frame
frame_count = 0
max_trackers = 10  # Limit the number of trackers
min_bbox_size = 20  # Minimum bounding box size
iou_threshold = 0.4  # IOU threshold for suppressing redundant boxes

trackers_cam1 = []
trackers_cam2 = []

def is_valid_bbox(bbox, width, height):
    x, y, w, h = bbox
    if w < min_bbox_size or h < min_bbox_size:
        return False
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return False
    return True

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
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Failed to retrieve frame from one or both cameras. Exiting...")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Resize frames
    frame1 = imutils.resize(frame1, width=600)
    frame2 = imutils.resize(frame2, width=600)
    height1, width1, _ = frame1.shape
    height2, width2, _ = frame2.shape

    if frame_count % (frame_skip * 5) == 0:  # Update every Nth frame to detect new objects
        # Process frame from camera 1
        blob1 = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob1)
        detections1 = net.forward(output_layers)

        trackers_cam1 = []  # Reset trackers
        boxes_cam1 = []
        confidences_cam1 = []

        for detection in detections1:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2 and class_id == 0:  # 0 is the class ID for 'person' in YOLO
                    center_x = int(obj[0] * width1)
                    center_y = int(obj[1] * height1)
                    w = int(obj[2] * width1)
                    h = int(obj[3] * height1)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    bbox = (x, y, w, h)
                    if is_valid_bbox(bbox, width1, height1):
                        boxes_cam1.append(bbox)
                        confidences_cam1.append(float(confidence))

        indices_cam1 = cv2.dnn.NMSBoxes(boxes_cam1, confidences_cam1, score_threshold=0.2, nms_threshold=iou_threshold)

        if len(indices_cam1) > 0:
            for i in indices_cam1.flatten():  # Use flatten to handle nested lists
                bbox = boxes_cam1[i]
                if len(trackers_cam1) < max_trackers:
                    try:
                        tracker = create_tracker("MIL")  # Use MIL tracker
                        tracker.init(frame1, bbox)
                        trackers_cam1.append((tracker, bbox))
                    except Exception as e:
                        print(f"Failed to create tracker: {e}")

        # Process frame from camera 2
        blob2 = cv2.dnn.blobFromImage(frame2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob2)
        detections2 = net.forward(output_layers)

        trackers_cam2 = []  # Reset trackers
        boxes_cam2 = []
        confidences_cam2 = []

        for detection in detections2:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2 and class_id == 0:  # 0 is the class ID for 'person' in YOLO
                    center_x = int(obj[0] * width2)
                    center_y = int(obj[1] * height2)
                    w = int(obj[2] * width2)
                    h = int(obj[3] * height2)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    bbox = (x, y, w, h)
                    if is_valid_bbox(bbox, width2, height2):
                        boxes_cam2.append(bbox)
                        confidences_cam2.append(float(confidence))

        indices_cam2 = cv2.dnn.NMSBoxes(boxes_cam2, confidences_cam2, score_threshold=0.2, nms_threshold=iou_threshold)

        if len(indices_cam2) > 0:
            for i in indices_cam2.flatten():  # Use flatten to handle nested lists
                bbox = boxes_cam2[i]
                if len(trackers_cam2) < max_trackers:
                    try:
                        tracker = create_tracker("MIL")  # Use MIL tracker
                        tracker.init(frame2, bbox)
                        trackers_cam2.append((tracker, bbox))
                    except Exception as e:
                        print(f"Failed to create tracker: {e}")

    # Update trackers for camera 1
    new_trackers_cam1 = []
    for tracker, bbox in trackers_cam1:
        ok, new_bbox = tracker.update(frame1)
        if ok and is_valid_bbox(new_bbox, width1, height1):
            x, y, w, h = [int(v) for v in new_bbox]
            current_position = (x + w // 2, y + h // 2)
            prev_position = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

            if is_above_line(prev_position[0], prev_position[1], line_start_cam1, line_end_cam1) > 0 and is_above_line(current_position[0], current_position[1], line_start_cam1, line_end_cam1) < 0:
                in_count_cam1 += 1
            elif is_above_line(prev_position[0], prev_position[1], line_start_cam1, line_end_cam1) < 0 and is_above_line(current_position[0], current_position[1], line_start_cam1, line_end_cam1) > 0:
                out_count_cam1 += 1

            new_trackers_cam1.append((tracker, new_bbox))
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    trackers_cam1 = new_trackers_cam1

    # Update trackers for camera 2
    new_trackers_cam2 = []
    for tracker, bbox in trackers_cam2:
        ok, new_bbox = tracker.update(frame2)
        if ok and is_valid_bbox(new_bbox, width2, height2):
            x, y, w, h = [int(v) for v in new_bbox]
            current_position = (x + w // 2, y + h // 2)
            prev_position = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

            if is_above_line(prev_position[0], prev_position[1], line_start_cam2, line_end_cam2) > 0 and is_above_line(current_position[0], current_position[1], line_start_cam2, line_end_cam2) < 0:
                in_count_cam2 += 1
            elif is_above_line(prev_position[0], prev_position[1], line_start_cam2, line_end_cam2) < 0 and is_above_line(current_position[0], current_position[1], line_start_cam2, line_end_cam2) > 0:
                out_count_cam2 += 1

            new_trackers_cam2.append((tracker, new_bbox))
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    trackers_cam2 = new_trackers_cam2

    # Draw slanted counting lines
    cv2.line(frame1, line_start_cam1, line_end_cam1, (0, 0, 255), 2)
    cv2.line(frame2, line_start_cam2, line_end_cam2, (0, 0, 255), 2)

    # Display counts
    cv2.putText(frame1, f'OUT: {out_count_cam1}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame1, f'IN: {in_count_cam1}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame2, f'OUT: {out_count_cam2}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame2, f'IN: {in_count_cam2}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display frames
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
