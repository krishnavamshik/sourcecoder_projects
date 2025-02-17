import cv2
import time
import math
import numpy as np
 
# Setting parameters
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
 
# Colors for object detected
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX
 


# Reading class names from text files
class_names = []
with open("C:/projects/AMI/weights/coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
 
weapon_classes = []
with open("C:/projects/AMI/weights/obj.names", "r") as f:
    weapon_classes = [line.strip() for line in f.readlines()]
 
# Setting up OpenCV net for object detection
yoloNet_obj = cv2.dnn.readNet("C:/projects/AMI/weights/yolov7-tiny.weights", "C:/projects/AMI/weights/yolov7-tiny.cfg")
yoloNet_obj.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
yoloNet_obj.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
 
model_obj = cv2.dnn_DetectionModel(yoloNet_obj)
model_obj.setInputParams(size=(480, 480), scale=1/255, swapRB=True)
 
# Setting up OpenCV net for weapon detection
yoloNet_weapon = cv2.dnn.readNet("C:/projects/AMI/weights/yolov2-tiny-custom_200000.weights", "C:/projects/AMI/weights/yolov2-tiny.cfg")
model_weapon = cv2.dnn_DetectionModel(yoloNet_weapon)
model_weapon.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
 
# Setting camera
camera = cv2.VideoCapture(0)  # 0 for the default webcam, or provide the video file path
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()
 
# Known distance from camera to the person (in meters) for calibration
CALIB_DISTANCE = 5.0  # Adjust this value based on your setup
CALIB_HEIGHT = 1.7   # Height of the person during calibration (in meters)
 
# Variables for distance and speed calculation
focal_length = 0
person_data = {}  # Dictionary to store data of detected persons
last_frame_time = time.time()
 
def calibrate_focal_length(box_height):
    global focal_length
    focal_length = (box_height * CALIB_DISTANCE) / CALIB_HEIGHT
    print(f"Camera calibrated. Focal length: {focal_length} pixels")
 
def estimate_distance(box_height):
    return (CALIB_HEIGHT * focal_length) / box_height
 
def calculate_speed(curr_pos, prev_pos, curr_time, prev_time, distance):
    dx = (curr_pos[0] - prev_pos[0]) / camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    dy = (curr_pos[1] - prev_pos[1]) / camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    dt = curr_time - prev_time
    if dt > 0:
        scale_factor = distance * 2  # Approximation: visible ground plane is twice the distance
        return math.sqrt(dx**2 + dy**2) * scale_factor / dt
    return 0
 
def match_persons(curr_boxes, prev_data):
    matched = {}
    for i, box in enumerate(curr_boxes):
        x, y, w, h = box
        center = (x + w // 2, y + h // 2)
        best_match = None
        min_dist = float('inf')
        for id, data in prev_data.items():
            prev_center = data['center']
            dist = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
            if dist < min_dist:
                min_dist = dist
                best_match = id
        if best_match is not None and min_dist < 100:  # Threshold for considering it the same person
            matched[i] = best_match
    return matched
 
def process_detections(image, boxes, matched_ids):
    current_time = time.time()
    for i, box in enumerate(boxes):
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)
 
        # Calculate distance
        distance = estimate_distance(h)
        
        # Determine ID and calculate speed
        person_id = matched_ids.get(i, len(person_data))
        if len(person_data) == 0:  # Check if it's the first frame
            person_id = 1
        center = (x + w // 2, y + h)
        speed = 0
        if person_id in person_data:
            prev_data = person_data[person_id]
            speed = calculate_speed(center, prev_data['center'], current_time, prev_data['time'], distance)
 
        # Update or create person data
        person_data[person_id] = {
            'center': center,
            'time': current_time,
            'distance': distance
        }
 
        # Display information
        cv2.putText(image, f"ID: {person_id}, {distance:.1f}m, {speed:.1f}m/s", (x, y - 10), fonts, 0.5, PINK, 2)
 
def ObjectDetector(image):
    global focal_length, last_frame_time, person_data
    current_time = time.time()
    fps = 1 / (current_time - last_frame_time)
    last_frame_time = current_time
 
    # Run object detection model
    classes_obj, scores_obj, boxes_obj = model_obj.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
 
    person_boxes = []
    for classid, score, box in zip(classes_obj, scores_obj, boxes_obj):
        if classid == class_names.index('person'):
            x, y, w, h = box
            if focal_length == 0 and h > image.shape[0] * 0.6:  # Person covers > 60% of frame height
                calibrate_focal_length(h)
            elif focal_length > 0:
                person_boxes.append(box)
 
    # Run weapon detection model
    classes_weapon, scores_weapon, boxes_weapon = model_weapon.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
 
    # Match current detections with previous ones
    matched_ids = match_persons(person_boxes, person_data)
 
    # Process all detected persons
    process_detections(image, person_boxes, matched_ids)
 
    # Draw bounding boxes and labels for weapons
    font = cv2.FONT_HERSHEY_PLAIN
    color = (0, 0, 255)  # Red color for weapon bounding boxes
    for classid, score, box in zip(classes_weapon, scores_weapon, boxes_weapon):
        if classid in range(len(weapon_classes)):
            x, y, w, h = box
            label = weapon_classes[classid]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), font, 2, color, 2)
 
    # Remove old data
    current_time = time.time()
    for pid in list(person_data.keys()):
        if current_time - person_data[pid]['time'] > 1.0:  # Not seen for more than 1 second
            del person_data[pid]
 
    # Display FPS
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), fonts, 0.6, ORANGE, 2)
 
# Main loop
try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read a frame.")
            break
 
        original = frame.copy()
      
        ObjectDetector(frame)
        cv2.imshow('Original', original)
        cv2.imshow('Detected', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):  # Press 'c' to recalibrate
            focal_length = 0
            print("Recalibrating... Stand at the calibration distance.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    camera.release()
    cv2.destroyAllWindows()