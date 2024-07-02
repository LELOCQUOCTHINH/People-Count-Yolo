import cv2
import numpy as np
import imutils

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize parameters for the camera
line_start = (180, 160)
line_end = (550, 400)

out_count = 0
in_count = 0

# Open the RTSP stream with proper codecs and increase buffer size
cap = cv2.VideoCapture("Your RTSP URL", cv2.CAP_FFMPEG)

# Increase the buffer size
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

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

trackers = []

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
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Resize frame
    frame = imutils.resize(frame, width=600)
    height, width, _ = frame.shape

    if frame_count % (frame_skip * 5) == 0:  # Update every Nth frame to detect new objects
        # Process frame
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        trackers = []  # Reset trackers
        boxes = []
        confidences = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2 and class_id == 0:  # 0 is the class ID for 'person' in YOLO
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

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.2, nms_threshold=iou_threshold)

        if len(indices) > 0:
            for i in indices.flatten():  # Use flatten to handle nested lists
                bbox = boxes[i]
                if len(trackers) < max_trackers:
                    try:
                        tracker = create_tracker("MIL")  # Use MIL tracker
                        tracker.init(frame, bbox)
                        trackers.append((tracker, bbox))
                    except Exception as e:
                        print(f"Failed to create tracker: {e}")

    # Update trackers
    new_trackers = []
    for tracker, bbox in trackers:
        ok, new_bbox = tracker.update(frame)
        if ok and is_valid_bbox(new_bbox, width, height):
            x, y, w, h = [int(v) for v in new_bbox]
            current_position = (x + w // 2, y + h // 2)
            prev_position = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

            if is_above_line(prev_position[0], prev_position[1], line_start, line_end) > 0 and is_above_line(current_position[0], current_position[1], line_start, line_end) < 0:
                in_count += 1
            elif is_above_line(prev_position[0], prev_position[1], line_start, line_end) < 0 and is_above_line(current_position[0], current_position[1], line_start, line_end) > 0:
                out_count += 1

            new_trackers.append((tracker, new_bbox))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    trackers = new_trackers

    # Draw slanted counting line
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

    # Display counts
    cv2.putText(frame, f'OUT: {out_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'IN: {in_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display frame
    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
