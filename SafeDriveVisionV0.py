import cv2
import dlib
import numpy as np
import torch
import math
import time
import pygame 
from scipy.spatial import distance as dist
from scipy.spatial import Delaunay
import threading

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)

pygame.mixer.init()
current_time = time.time()

# Chemins des fichiers audio et délais associés
sounds = {
    'eye': ('./eye.mp3', 10),
    'regarder': ('./regarder.mp3', 10),
    'reposer': ('./reposer.mp3', 15),
    'phone': ('./phone.mp3', 15),
    'welcome': ('./s1.mp3', 0),
    'welcome_eng': ('./welcomeengl.mp3', 0)
}

# Dernière fois que le son a été joué
last_played = {key: 0 for key in sounds}

def play_sound(sound_key):
    audio_file, delay = sounds[sound_key]
    current_time = time.time()
    if current_time - last_played[sound_key] > delay:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        last_played[sound_key] = current_time  # Mise à jour du timestamp après lecture

def sound_thread(sound_key):
    thread = threading.Thread(target=play_sound, args=(sound_key,))
    thread.daemon = True
    thread.start()





print("[INFO] project realized by: RMA assurance Marocaine d'assurance")
# Initialiser le détecteur et le prédicteur de dlib
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks (1).dat')

print("[INFO] initializing camera...")
cap = cv2.VideoCapture(0)

desired_fps = 30
cap.set(cv2.CAP_PROP_FPS, desired_fps)

def get_camera_matrix(size):
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    return np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

# Modifiez model_points selon les points de repère que vous avez choisis
model_points = np.array([
    (0.0, 0.0, 0.0),  # Point de référence - bout du nez
    (-30.0, -125.0, -30.0),  # Coin gauche de l'oeil
    (30.0, -125.0, -30.0),  # Coin droit de l'oeil
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


def getHeadTiltAndCoords(size, image_points, frame_height):
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [
        0, focal_length, center[1]], [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                  camera_matrix, dist_coeffs,
                                                                  flags = cv2.SOLVEPNP_ITERATIVE)
    (nose_end_point2D, _) = cv2.projectPoints(np.array(
        [(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    head_tilt_degree = abs(
        [-180] - np.rad2deg([rotationMatrixToEulerAngles(rotation_matrix)[0]]))
    starting_point = (int(image_points[0][0]), int(image_points[0][1]))
    ending_point = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    ending_point_alternate = (ending_point[0], frame_height // 2)

    return head_tilt_degree, starting_point, ending_point, ending_point_alternate


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def nose_aspect_ratio(nose):
    vertical_distance = dist.euclidean(nose[0], nose[2])
    depth_distance = dist.euclidean(nose[0], nose[1])
    return depth_distance / vertical_distance

def calculate_head_angle(eye_left, eye_right, nose_tip):
    eye_center = (eye_left + eye_right) / 2
    vector_nose = nose_tip - eye_center
    vector_horizontal = (eye_right - eye_left)
    vector_horizontal[1] = 0
    vector_nose_normalized = vector_nose / np.linalg.norm(vector_nose)
    vector_horizontal_normalized = vector_horizontal / np.linalg.norm(vector_horizontal)
    angle_rad = np.arccos(np.clip(np.dot(vector_nose_normalized, vector_horizontal_normalized), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg
# Charger le modèle 
weights_path = 'C:\\Users\\o\\Downloads\\yolov5\\yolov5m.pt'  # Mettez à jour avec le chemin exact du fichier de poids
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Variables pour suivre les alertes
COUNTER1 = 0
COUNTER2 = 0
COUNTER3 = 0
EYE_AR_CONSEC_FRAMES = 30
repeat_counter = 0
face_detected = False
# Démarrer le son de bienvenue
sound_thread('welcome')
sound_thread('welcome_eng')

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if len(faces)==0 : 
        cv2.putText(img, "Le conducteur ne regarde pas devant lui", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        sound_thread("regarder")

    results = model(img)
    detections = results.xyxy[0]
    for detection in detections:
        if int(detection[5]) == 67:  # 67 est l'index de 'cell phone'
            x1, y1, x2, y2, conf = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]), detection[4]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'Cell Phone {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("driver is using cell phone ", current_time)
            COUNTER2 += 1
            if COUNTER2 >= 3:
                 
                cv2.putText(img, "Rangez votre CELL PHONE!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                sound_thread("phone")
                COUNTER2 = 0
    for face in faces:
            landmarks = predictor(gray, face)
            landmarks_points = np.array([(p.x, p.y) for p in landmarks.parts()])

            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            image_points = np.array([
                        (landmarks_points[30][0], landmarks_points[30][1]),
                        (landmarks_points[8][0], landmarks_points[8][1]),
                        (landmarks_points[36][0], landmarks_points[36][1]),
                        (landmarks_points[45][0], landmarks_points[45][1]),
                        (landmarks_points[48][0], landmarks_points[48][1]),
                        (landmarks_points[54][0], landmarks_points[54][1])
                    ], dtype="double")

            if len(landmarks_points) >= len(model_points):
                # Extraire les points d'image spécifiques pour solvePnP
                image_points = np.array([
                    landmarks_points[30],  # Bout du nez
                    landmarks_points[36],  # Coin gauche de l'œil
                    landmarks_points[45],  # Coin droit de l'œil
                    landmarks_points[48],  # Coin gauche de la bouche
                    landmarks_points[54],  # Coin droit de la bouche
                    landmarks_points[8]    # Menton
                ], dtype="double")

                camera_matrix = get_camera_matrix(img.shape)
                dist_coeffs = np.zeros((4, 1))  # Aucune distorsion

                # Effectuer solvePnP
                success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
                if success:
                    # Projeter les points 3D dans l'espace 2D
                    projected_points, _ = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    '''for point in projected_points:
                        p = (int(point[0][0]), int(point[0][1]))
                        #cv2.circle(img, p, 3, (0, 255, 0), -1)  # Dessiner en vert
'''
        # Dessiner les points de repère
            for point in landmarks_points:
                cv2.circle(img, (point[0], point[1]), 2, (255, 255, 255), -1)
            left_eye = landmarks_points[36:42]
            right_eye = landmarks_points[42:48]
            left_eyeHull = cv2.convexHull(left_eye)
            right_eyeHull = cv2.convexHull(right_eye)
            cv2.drawContours(img, [left_eyeHull], -1, (255, 255, 255), 1)
            cv2.drawContours(img, [right_eyeHull], -1, (255, 255, 255), 1)
            ear = eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye) / 2.0
            mouth = landmarks_points[48:68]
            mounthHull = cv2.convexHull(mouth)
            cv2.drawContours(img, [mounthHull], -1, (0, 255, 0), 1)
            mar = mouth_aspect_ratio(landmarks_points[48:68])
            cv2.putText(img, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(img, f'MAR: {mar:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            nose_points = [landmarks_points[27], landmarks_points[30], landmarks_points[33]]
            nar = nose_aspect_ratio(nose_points)
            text = f'Nose Aspect Ratio: {nar:.2f}'
            cv2.putText(img, text,(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
            eye_left = landmarks_points[36]
            eye_right = landmarks_points[45]
            nose_tip = landmarks_points[33]
            head_angle = calculate_head_angle(np.array(eye_left), np.array(eye_right), np.array(nose_tip))
            cv2.putText(img, f'Head Angle: {head_angle:.2f}',(10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            size = img.shape
            frame_height = img.shape[0]
            head_tilt_degree, start_point, end_point, end_point_alt = getHeadTiltAndCoords(size, image_points, frame_height)

            cv2.putText(img, f'Head Tilt: {head_tilt_degree[0]:.2f} degrees', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.line(img, start_point, end_point, (0, 255, 0), 2)
            if 75 > head_angle and head_angle > 110 :
                cv2.putText(img, "Regardez devant vous!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                COUNTER3 += 1
                if COUNTER3 >= 6:  
                    sound_thread("regarder")          
                    cv2.putText(img, "Regardez devant vous!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    COUNTER3 = 0
                 
            else:
                COUNTER3 = 0



            if ear < 0.33:
                        cv2.putText(img, "Eyes Closed!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        COUNTER1 += 1
                        if COUNTER1 >= 4:
                            sound_thread("eye")
                            repeat_counter += 1
                            cv2.putText(img, "Eyes Closed!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            COUNTER1 = 0
                            if repeat_counter >= 3:
                                sound_thread("reposer")
                                repeat_counter = 0
                                cv2.putText(img, "Eyes Closed 3 times!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                        COUNTER1 = 0
                        repeat_counter = 0
            if mar > 0.7:
                        sound_thread("reposer")
                        cv2.putText(img, "Yawning!", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            head_tilt_degree, start_point, end_point, end_point_alt = getHeadTiltAndCoords(size, image_points, frame_height)

            cv2.line(img, start_point, end_point, (255, 0, 0), 2)
            cv2.line(img, start_point, end_point_alt, (0, 0, 255), 2)


    cv2.imshow("Video Stream", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
