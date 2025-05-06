import cv2
import numpy as np
import imutils
from picamera.array import PiRGBArray
from picamera import PiCamera

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize camera
camera = PiCamera()
camera.resolution = (320, 240)  # Reduced resolution for performance
camera.framerate = 10
raw_capture = PiRGBArray(camera, size=(320, 240))

out_count = 0
in_count = 0
tracked_ids = {}  # Dictionary to track object IDs and their crossing status

def create_tracker(tracker_type="CSRT"):
    if tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")

def is_above_line(x, y, line_start, line_end):
    vx = line_end[0] - line_start[0]
    vy = line_end[1] - line_start[1]
    px = x - line_start[0]
    py = y - line_start[1]
    cross_product = px * vy - py * vx
    return cross_product > 0

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

frame_skip = 5  # Increased to reduce processing load
frame_count = 0
max_trackers = 10  # Reduced to save resources
min_bbox_size = 15
iou_threshold = 0.3

trackers = []
tracker_id = 0  # Unique ID for each tracker

def is_valid_bbox(bbox, width, height):
    x, y, w, h = bbox
    if w < min_bbox_size or h < min_bbox_size:
        return False
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return False
    return True

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    image = frame.array
    frame_count += 1
    if frame_count % frame_skip != 0:
        raw_capture.truncate(0)
        continue

    height, width, _ = image.shape
    line_start = (0, height // 2)
    line_end = (width, height // 2)

    if frame_count % (frame_skip * 5) == 0:
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        boxes = []
        confidences = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2 and class_id == 0:  # Class 0 is 'person'
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

        new_trackers = []
        for tracker, bbox, tid in trackers:
            ok, new_bbox = tracker.update(image)
            if ok and is_valid_bbox(new_bbox, width, height):
                new_trackers.append((tracker, new_bbox, tid))

        trackers = new_trackers

        if len(indices) > 0:
            for i in indices.flatten():
                bbox = boxes[i]
                if len(trackers) < max_trackers:
                    best_iou = 0
                    best_tracker_idx = -1
                    for idx, (_, tracked_bbox, _) in enumerate(trackers):
                        iou = calculate_iou(bbox, tracked_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_tracker_idx = idx

                    if best_iou < 0.3:  # If no good match, create new tracker
                        try:
                            tracker = create_tracker("CSRT")
                            tracker.init(image, bbox)
                            trackers.append((tracker, bbox, tracker_id))
                            tracked_ids[tracker_id] = None  # None means not crossed yet
                            tracker_id += 1
                        except Exception as e:
                            print(f"Failed to create tracker: {e}")

    new_trackers = []
    for tracker, bbox, tid in trackers:
        ok, new_bbox = tracker.update(image)
        if ok and is_valid_bbox(new_bbox, width, height):
            x, y, w, h = [int(v) for v in new_bbox]
            current_position = (x + w // 2, y + h // 2)
            prev_position = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

            prev_above = is_above_line(prev_position[0], prev_position[1], line_start, line_end)
            curr_above = is_above_line(current_position[0], current_position[1], line_start, line_end)

            if prev_above and not curr_above and tracked_ids[tid] is None:
                in_count += 1
                tracked_ids[tid] = "in"
            elif not prev_above and curr_above and tracked_ids[tid] is None:
                out_count += 1
                tracked_ids[tid] = "out"

            new_trackers.append((tracker, new_bbox, tid))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f'ID: {tid}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    trackers = new_trackers

    cv2.line(image, line_start, line_end, (0, 0, 255), 2)
    cv2.putText(image, f'OUT: {out_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)