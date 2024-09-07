import cv2
import time
import imutils
import argparse
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import math
from imutils.video import FPS
from imutils.video import VideoStream
from ultralytics.utils.plotting import Annotator

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

model_points = np.array([
    (0.0, 0.0, 0.0),  # Bout du nez
    (-30.0, -125.0, -30.0),  # Coin gauche de l'œil
    (30.0, -125.0, -30.0),  # Coin droit de l'œil
    (-60.0, -70.0, -60.0),  # Coin gauche de la bouche
    (60.0, -70.0, -60.0),  # Coin droit de la bouche
    (0.0, -330.0, -65.0)    # Menton
])

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def calculate_ear(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])  
    ear = (A + B) / (2.0 * C)
    return ear

def get_color(idx):
    np.random.seed(idx)
    return tuple(np.random.randint(0, 255, 3).tolist())

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help='Path to prototxt')
ap.add_argument("-m", "--model", required=True, help='Path to model weights')
ap.add_argument("-c", "--confidence", type=float, default=0.7, help='Minimum probability to filter weak detections')
args = vars(ap.parse_args())

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
Mouth = [61, 185, 40, 39, 37, 0, 267, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "cell phone", "cow", 
          "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
FACE_68_LANDMARKS = [1, 33, 61, 291, 199, 263, 362, 385, 387, 373, 380, 33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 61, 185, 40, 39, 37, 0, 267, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

print('[Status] SafeDriveVersionV2 Loading...')
nn = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

print('[Status] Starting Video Stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

def detect_hands_on_wheel(hand_landmarks):
    if len(hand_landmarks) == 2:  
        hand1 = hand_landmarks[0]
        hand2 = hand_landmarks[1]
        distance_between_hands = dist.euclidean(
            (hand1.landmark[mp_hands.HandLandmark.WRIST].x, hand1.landmark[mp_hands.HandLandmark.WRIST].y),
            (hand2.landmark[mp_hands.HandLandmark.WRIST].x, hand2.landmark[mp_hands.HandLandmark.WRIST].y)
        )
        if distance_between_hands < 0.1: 
            return True
    return False

def get_hand_side(hand_landmarks):
    thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    if thumb_tip_x < wrist_x:
        return "Left Hand"
    else:
        return "Right Hand"

# Define the ROI for counting people
ROI_TOP_LEFT = (100, 100)
ROI_BOTTOM_RIGHT = (500, 400)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    nn.setInput(blob)
    detections = nn.forward()

    detected_objects = []

    annotator = Annotator(frame, line_width=2)

    # Initialize the person count within ROI
    face_count_in_roi = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Filter weak detections
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Check if a face is detected 



            # Create a mask for the detected object (simple rectangular mask for illustration)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[startY:endY, startX:endX] = 255

            color = get_color(idx)
            txt_color = annotator.get_txt_color(color)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, 2)

            # Optionally draw label
            cv2.putText(frame, labels[idx], (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(frame_rgb)
    result_face_mesh = face_mesh.process(frame_rgb)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        if (x > ROI_TOP_LEFT[0] and y > ROI_TOP_LEFT[1] and (x+w) < ROI_BOTTOM_RIGHT[0] and (y+h) < ROI_BOTTOM_RIGHT[1]):
            face_count_in_roi += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        detected_objects.append("Face Detected")
        
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        
        # Detect smiles
        smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=15)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(face_color, (sx, sy), (sx+sw, sy+sh), (255, 255, 255), 2)
            cv2.putText(frame, 'Smile Detected', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            detected_objects.append("Smile Detected")
    
    # Detect hands
    if result_hands.multi_hand_landmarks:
        if detect_hands_on_wheel(result_hands.multi_hand_landmarks):
            detected_objects.append("Hands on Wheel")
        else:
            detected_objects.append("Hands off Wheel")
            
        for hand_landmarks in result_hands.multi_hand_landmarks:
            hand_side = get_hand_side(hand_landmarks)
            detected_objects.append(hand_side)
            
            # Calculate bounding box for hand
            hand_landmarks_array = np.array([[(landmark.x * w, landmark.y * h) for landmark in hand_landmarks.landmark]], dtype=np.float32)
            box = cv2.boundingRect(hand_landmarks_array)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detect gestures or manipulation of objects
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = dist.euclidean(
                (thumb_tip.x * w, thumb_tip.y * h),
                (index_finger_tip.x * w, index_finger_tip.y * h)
            )
            if distance < 0.05:  # Threshold to detect pinch or holding small object
                detected_objects.append("Manipulating Object")
            else:
                detected_objects.append("No Object Manipulation")
    
    # Detect face mesh
    if result_face_mesh.multi_face_landmarks:
        for face_landmarks in result_face_mesh.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))
                
            landmarks = face_landmarks.landmark
            left_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE]
            left_eye_points = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in left_eye_landmarks]
            right_eye_points = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in right_eye_landmarks]

            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            ear = (left_ear + right_ear) / 2.0
            
            if ear < 0.25:
                detected_objects.append("Eyes Closed")
            else:
                detected_objects.append("Eyes Open")

            image_points = np.array([
                (landmarks[1].x * frame.shape[1], landmarks[1].y * frame.shape[0]),  # Bout du nez
                (landmarks[33].x * frame.shape[1], landmarks[33].y * frame.shape[0]),  # Coin gauche de l'œil
                (landmarks[263].x * frame.shape[1], landmarks[263].y * frame.shape[0]),  # Coin droit de l'œil
                (landmarks[61].x * frame.shape[1], landmarks[61].y * frame.shape[0]),  # Coin gauche de la bouche
                (landmarks[291].x * frame.shape[1], landmarks[291].y * frame.shape[0]),  # Coin droit de la bouche
                (landmarks[199].x * frame.shape[1], landmarks[199].y * frame.shape[0])  # Menton
            ], dtype="double")
            size = frame.shape[1], frame.shape[0]
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
            dist_coeffs = np.zeros((4, 1)) 

            success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles = rotationMatrixToEulerAngles(rotation_matrix)
            
            if abs(angles[1]) > 0.25:
                detected_objects.append("Not Looking Ahead")
            elif abs(angles[1]) < 0.07:
                detected_objects.append("Not Looking Ahead")
            else:
                detected_objects.append("Looking Ahead")
            
            if abs(angles[0]) < 1.4 :
                detected_objects.append("Looking Down")
            elif abs(angles[0]) > 1.6 :
                detected_objects.append("Looking Up")
            else :
                detected_objects.append("Looking Straight")
            
            detected_objects.append("Pitch: {:.2f} Yaw: {:.2f} Roll: {:.2f}".format(angles[0], angles[1], angles[2]))

    # Draw ROI rectangle
    cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)
    # Display person count in ROI
    cv2.putText(frame, f'faces in ROI: {face_count_in_roi}', (ROI_TOP_LEFT[0], ROI_TOP_LEFT[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    info_display = np.zeros((300, 600, 3), dtype=np.uint8)

    for idx, text in enumerate(detected_objects):
        cv2.putText(info_display, text, (10, (idx + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Info", info_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
    fps.update()

fps.stop()
print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[Info] Approximate FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()
vs.stop()
