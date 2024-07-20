import cv2
import time
import imutils
import argparse
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from imutils.video import FPS
from imutils.video import VideoStream

# Fonctions de calcul
def calculate_ear(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help='Path to prototxt')
ap.add_argument("-m", "--model", required=True, help='Path to model weights')
ap.add_argument("-c", "--confidence", type=float, default=0.7, help='Minimum probability to filter weak detections')
args = vars(ap.parse_args())

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# List of labels
labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
          "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# Generate colors for each label
colors = np.random.uniform(0, 255, size=(len(labels), 3))

# Load the Caffe model
print('[Status] Loading Model...')
try:
    nn = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the Haar cascade for face and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Initialize Mediapipe for hand and face mesh detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start the video stream
print('[Status] Starting Video Stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=700)
    (h, w) = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(frame_rgb)
    result_face_mesh = face_mesh.process(frame_rgb)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        
        # Detect smiles
        smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(face_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
            cv2.putText(frame, "Smile Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Detect hands
    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Detect face mesh
    if result_face_mesh.multi_face_landmarks:
        for face_landmarks in result_face_mesh.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

            left_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE]
            left_eye_points = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in left_eye_landmarks]
            right_eye_points = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in right_eye_landmarks]
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            ear = (left_ear + right_ear) / 2.0
            
            if ear < 0.25:
                cv2.putText(frame, "Eyes Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Eyes Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Perform the forward pass
    nn.setInput(blob)
    detections = nn.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break
    
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[Info] Approximate FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()
vs.stop()
