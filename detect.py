import base64
import threading

import cv2
import mediapipe as mp
import numpy as np
import zmq
from flask import *
from flask_socketio import SocketIO

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

tracker = None
status = None

app = Flask(__name__)
socketio = SocketIO(app)


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def biceps():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:7000")
    stage = None
    counter = 0
    # Setup mediapipe instance

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            #  receiving the frame as bytes
            message = socket.recv_string()

            # converting bytes into image numpy array
            header, data = message.split(',', 1)
            image_data = base64.b64decode(data)
            np_array = np.frombuffer(image_data, np.uint8)
            # print(' array:', np_array[:2])
            image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # image = cv2.resize(image, (700, 450))

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                left_ankle_visibility = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
                right_ankle_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

                if (left_ankle_visibility < 0.70) or (right_ankle_visibility < 0.70):
                    print(left_ankle_visibility)
                    cv2.putText(image, str("please go back"),
                                (200, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

                else:

                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calculate angle
                    angle = calculate_angle(shoulder, elbow, wrist)

                    # Visualize angle
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(elbow, [450, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                                )
                    # for hip angle
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    hip_angle = calculate_angle(shoulder, hip, left_knee)
                    cv2.putText(image, str(hip_angle),
                                tuple(np.multiply(hip, [960, 640]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5, cv2.LINE_AA
                                )
                    if hip_angle < 169:
                        cv2.putText(image, str("Stand straight"),
                                    (250, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
                        # Curl counter logic
                    if angle > 160:
                        stage = "down"
                    if angle < 40 and stage == 'down':
                        stage = "up"
                        counter += 1

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                             circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            image_data = cv2.resize(image, (720, 360))
            # converting frames into bytes
            image_data = cv2.imencode('image_data.jpg', image_data)[1].tobytes()
            base_64_encoded = base64.b64encode(image_data).decode('utf-8')
            image_data = "data:image/jpeg;base64,{}".format(base_64_encoded)
            obj = {"image_data": image_data, "counter": counter}
            # sending image in bytes form to app.py
            socket.send_pyobj(obj)


def squats():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:7001")
    stage = None
    counter = 0
    # Setup mediapipe instance

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            #  receiving the frame as bytes
            message = socket.recv_string()

            # converting bytes into image numpy array
            header, data = message.split(',', 1)
            image_data = base64.b64decode(data)
            np_array = np.frombuffer(image_data, np.uint8)
            # print(' array:', np_array[:2])
            image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # image = cv2.resize(image, (700, 450))

            # Extract landmarks
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                left_ankle_visibility = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
                right_ankle_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
                if (left_ankle_visibility < 0.70) or (right_ankle_visibility < 0.70):
                    print(left_ankle_visibility)
                    cv2.putText(image, str("please go back"),
                                (200, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

                else:

                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    # print(hip,knee,ankle)
                    # Calculate angle
                    knee_angle = calculate_angle(hip, knee, ankle)

                    # print(knee_angle)
                    # Visualize angle
                    cv2.putText(image, str(knee_angle),
                                tuple(np.multiply(knee, [960, 640]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2, cv2.LINE_AA
                                )

                    # Curl counter logic
                    if knee_angle > 170:
                        stage = "up"
                    if knee_angle < 90 and stage == 'up':
                        stage = "down"
                        counter += 1
                        print(counter)

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            image_data = cv2.resize(image, (720, 360))
            # converting frames into bytes
            image_data = cv2.imencode('image_data.jpg', image_data)[1].tobytes()
            base_64_encoded = base64.b64encode(image_data).decode('utf-8')
            image_data = "data:image/jpeg;base64,{}".format(base_64_encoded)
            obj = {"image_data": image_data, "counter": counter}
            # sending image in bytes form to app.py
            socket.send_pyobj(obj)


def pushup():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:7002")
    stage = None
    counter = 0
    # Setup mediapipe instance

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            #  receiving the frame as bytes
            message = socket.recv_string()

            # converting bytes into image numpy array
            header, data = message.split(',', 1)
            image_data = base64.b64decode(data)
            np_array = np.frombuffer(image_data, np.uint8)
            # print(' array:', np_array[:2])
            image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # image = cv2.resize(image, (700, 450))


            try:
                landmarks = results.pose_landmarks.landmark
                # print(landmarks)

                # Get coordinates for elbow
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                # print(shoulder,elbow,wrist)
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [960, 640]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA
                            )

                # for knee
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                # print(hip,knee,ankle)
                # Calculate angle
                knee_angle = calculate_angle(hip, left_knee, ankle)

                # print(knee_angle)
                # Visualize angle
                # cv2.putText(image, str(knee_angle),
                #             tuple(np.multiply(knee, [960, 640]).astype(int)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA
                #                     )

                # Curl counter logic
                # print(angle)
                if (angle > 170 and knee_angle > 160):
                    print(angle)
                    #                     print(knee_angle)
                    stage = "down"
                else:
                    cv2.putText(image, 'Lift up properly', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                if angle < 90 and stage == 'down' and knee_angle > 160:
                    stage = "up"
                    counter += 1
                    print(angle)

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            #         cv2.putText(image, 'STAGE', (65,12),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            #         cv2.putText(image, stage,
            #                     (60,60),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                             circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            image_data = cv2.resize(image, (720, 360))
            # converting frames into bytes
            image_data = cv2.imencode('image_data.jpg', image_data)[1].tobytes()
            base_64_encoded = base64.b64encode(image_data).decode('utf-8')
            image_data = "data:image/jpeg;base64,{}".format(base_64_encoded)
            obj = {"image_data": image_data, "counter": counter}
            # sending image in bytes form to app.py
            socket.send_pyobj(obj)


if __name__ == "__main__":
    threading.Thread(target=biceps).start()
    threading.Thread(target=squats).start()
    threading.Thread(target=pushup).start()
