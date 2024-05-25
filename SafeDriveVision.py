import argparse
import cv2
import dlib
import numpy as np
import torch
import math
import time
import pygame
from scipy.spatial import distance as dist
from collections import deque
import threading
import yaml
from tqdm import tqdm
import sys
import os
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.functions import cv_draw_landmark
# modell/experimental.py


#yolov5_path = 'C:\\Users\\o\\Downloads\\3DDFA_V2-master\\3DDFA_V2-master\\yolov5-master\\yolov5-master'
sys.path.append(yolov5_path)
# Importer les modules YOLOv5
'''
from modell.yolo import Detect, Model
from modell.common import DetectMultiBackend
from utill.general import non_max_suppression, scale_coords
from utill.torch_utils import select_device
'''
# Sound configuration
pygame.mixer.init()
current_time = time.time()
sounds = {
    'eye': ('./eye.mp3', 5),
    'regarder': ('./regarder.mp3', 5),
    'reposer': ('./reposer.mp3', 5),
    'phone': ('./phone.mp3', 5),
    'welcome': ('./s1.mp3', 0),
    'welcome_eng': ('./welcomeengl.mp3', 0)
}
last_played = {key: 0 for key in sounds}

def play_sound(sound_key):
    audio_file, delay = sounds[sound_key]
    current_time = time.time()
    if current_time - last_played[sound_key] > delay:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        last_played[sound_key] = current_time

def sound_thread(sound_key):
    thread = threading.Thread(target=play_sound, args=(sound_key,))
    thread.daemon = True
    thread.start()

# Function to get camera matrix
def get_camera_matrix(size):
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    return np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

# Function to check if matrix is rotation matrix
def is_rotation_matrix(R):
    Rt = np.transpose(R)
    should_be_identity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - should_be_identity)
    return n < 1e-6

# Function to get Euler angles from rotation matrix
def rotation_matrix_to_euler_angles(R):
    assert (is_rotation_matrix(R))
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

# Define model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),  # Tip of the nose
    (-30.0, -125.0, -30.0),  # Left eye corner
    (30.0, -125.0, -30.0),  # Right eye corner
    (-60.0, -70.0, -60.0),  # Left mouth corner
    (60.0, -70.0, -60.0),  # Right mouth corner
    (0.0, -330.0, -65.0)  # Chin
])

# Function to get head tilt and coordinates
def get_head_tilt_and_coords(size, image_points, frame_height):
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    head_tilt_degree = abs([-180] - np.rad2deg([rotation_matrix_to_euler_angles(rotation_matrix)[0]]))
    starting_point = (int(image_points[0][0]), int(image_points[0][1]))
    ending_point = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    ending_point_alternate = (ending_point[0], frame_height // 2)
    return head_tilt_degree, starting_point, ending_point, ending_point_alternate

# Function to compute eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to compute mouth aspect ratio
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Function to compute nose aspect ratio
def nose_aspect_ratio(nose):
    vertical_distance = dist.euclidean(nose[0], nose[2])
    depth_distance = dist.euclidean(nose[0], nose[1])
    return depth_distance / vertical_distance

# Function to calculate head angle
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

def webcam_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()

def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX
        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    reader = webcam_frames()

    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()

    dense_flag = args.opt in ('2d_dense', '3d')
    pre_ver = None

    # Load dlib model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks (1).dat')
    # Load YOLOv5 model
    #weights_path = 'C:\\Users\\o\\Downloads\\yolov5-master\\yolov5m.pt'
    #device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
   #model = DetectMultiBackend(weights_path, device=device, dnn=False)

    COUNTER1 = 0
    COUNTER2 = 0
    COUNTER3 = 0
    EYE_AR_CONSEC_FRAMES = 30
    repeat_counter = 0

    sound_thread('welcome')
    sound_thread('welcome_eng')

    for i, frame_bgr in tqdm(enumerate(reader)):
        if i == 0:
            boxes = face_boxes(frame_bgr)
            if len(boxes) > 0:
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

                param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

                for _ in range(n_pre):
                    queue_ver.append(ver.copy())
                    queue_frame.append(frame_bgr.copy())
                queue_ver.append(ver.copy())
                queue_frame.append(frame_bgr.copy())
            else:
                continue  # Skip this frame if no face is detected
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')
            roi_box = roi_box_lst[0]
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                if len(boxes) > 0:
                    boxes = [boxes[0]]
                    param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            queue_ver.append(ver.copy())
            queue_frame.append(frame_bgr.copy())

        pre_ver = ver

        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)
            if args.opt == '2d_sparse':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)
            elif args.opt == '2d_dense':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
            elif args.opt == '3d':
                img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
            else:
                raise ValueError(f'Unknown opt {args.opt}')

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)
            if len(faces) == 0:
                sound_thread("regarder")
                cv2.putText(img_draw, "Regardez devant vous!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            '''
            results = model(frame_bgr)
            detections = results.xyxy[0].cpu().numpy()
            for detection in detections:
                if int(detection[5]) == 67:  # Assuming 67 is the class id for cell phones
                    x1, y1, x2, y2, conf = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]), detection[4]
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_draw, f'Cell Phone {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    COUNTER2 += 1
                    if COUNTER2 >= 3:
                        cv2.putText(img_draw, "Rangez votre CELL PHONE!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        sound_thread("phone")
                        COUNTER2 = 0
'''
            for face in faces:
                landmarks = predictor(gray, face)
                landmarks_points = np.array([(p.x, p.y) for p in landmarks.parts()])

                image_points = np.array([
                    (landmarks_points[30][0], landmarks_points[30][1]),
                    (landmarks_points[8][0], landmarks_points[8][1]),
                    (landmarks_points[36][0], landmarks_points[36][1]),
                    (landmarks_points[45][0], landmarks_points[45][1]),
                    (landmarks_points[48][0], landmarks_points[48][1]),
                    (landmarks_points[54][0], landmarks_points[54][1])
                ], dtype="double")

                left_eye = landmarks_points[36:42]
                right_eye = landmarks_points[42:48]
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(img_draw, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(img_draw, [right_eye_hull], -1, (0, 255, 0), 1)
                ear = eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye) / 2.0
                mouth = landmarks_points[48:68]
                mouth_hull = cv2.convexHull(mouth)
                cv2.drawContours(img_draw, [mouth_hull], -1, (0, 255, 0), 1)
                mar = mouth_aspect_ratio(landmarks_points[48:68])
                cv2.putText(img_draw, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(img_draw, f'MAR: {mar:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                nose_points = [landmarks_points[27], landmarks_points[30], landmarks_points[33]]
                nar = nose_aspect_ratio(nose_points)
                cv2.putText(img_draw, f'NAR: {nar:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                eye_left = landmarks_points[36]
                eye_right = landmarks_points[45]
                nose_tip = landmarks_points[33]
                head_angle = calculate_head_angle(np.array(eye_left), np.array(eye_right), np.array(nose_tip))
                cv2.putText(img_draw, f'Head Angle: {head_angle:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                frame_height = img_draw.shape[0]
                head_tilt_degree, start_point, end_point, end_point_alt = get_head_tilt_and_coords(img_draw.shape, image_points, frame_height)

                cv2.putText(img_draw, f'Head Tilt: {head_tilt_degree[0]:.2f} degrees', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.line(img_draw, start_point, end_point, (0, 255, 0), 2)

                if 75 > head_angle < 110:
                    cv2.putText(img_draw, "Regardez devant vous!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    COUNTER3 += 1
                    if COUNTER3 >= 6:
                        sound_thread("regarder")
                        COUNTER3 = 0
                else:
                    COUNTER3 = 0

            if ear < 0.33:
                cv2.putText(img_draw, "Eyes Closed!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                COUNTER1 += 1
                if COUNTER1 >= 6:
                    sound_thread("eye")
                    COUNTER1 = 0  # Réinitialiser le compteur ici après avoir joué le son
            else:
                COUNTER1 = 0  # Réinitialiser le compteur seulement si la condition n'est pas remplie


                if mar > 0.7:
                    sound_thread("reposer")
                    cv2.putText(img_draw, "Yawning!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                head_tilt_degree, start_point, end_point, end_point_alt = get_head_tilt_and_coords(img_draw.shape, image_points, frame_height)
                cv2.line(img_draw, start_point, end_point, (255, 0, 0), 2)
                cv2.line(img_draw, start_point, end_point_alt, (0, 0, 255), 2)

            cv2.imshow('image', img_draw)
            k = cv2.waitKey(20)
            if (k & 0xff == ord('q')):
                break

            queue_ver.popleft()
            queue_frame.popleft()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
