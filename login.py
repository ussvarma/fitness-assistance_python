import base64
from flask import *
from flask import render_template
from flask import request
import os.path
import cv2
import face_recognition as fr
import os
import mediapipe as mp
from datetime import datetime
from threading import Thread
from multiprocessing import Process
from flask_socketio import SocketIO, send, emit
import numpy as np

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


# Connect to your PostgresSQL database on a remote server
# def connections():
#     conn = psycopg2.connect(host="127.0.0.1", port="5432", dbname="user_details", user="postgres", password="p@ssw0rd")
#     # Open a cursor to perform database operations
#     cur = conn.cursor()
#     return cur, conn

app.secret_key = "ussvarma"


@app.route('/')
def home():
    return render_template("light_workout.html")


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/add', methods=['POST'])
def add():
    if request.method == 'POST':
        name = request.form.get('name')
        username = request.form.get('username')
        height = request.form.get('height')
        password = request.form.get('password')
        # cur, conn = connections()
        check_in_db = "SELECT * from details where username like %s"
        # cur.execute(check_in_db, [username])
        # result = cur.fetchall()
        # print(result)
        # if len(result) >= 1:
        #     msg = "user name already exists, Register with other username"
        #     return render_template('register.html', msg=msg)
        # else:
        #     cur.execute('INSERT INTO details(name,username,height,password) VALUES (%s,%s,%s,%s)',
        #                 (name, username, height, password))
        #     conn.commit()
        #     conn.close()
        #     msg = 'Registered successfully'
        #     return render_template('login.html', msg=msg)


@app.route('/login')
def login():
    return render_template("login.html")


@socketio.on('message')
def login(data):
    print(data)
    return render_template("light_workout.html")


@app.route('/face')
def face():
    tracker = None

    def face_pro():
        global tracker

        path = "static/users_images/"

        known_names = []
        known_name_encodings = []

        images = os.listdir(path)
        for _ in images:
            image = fr.load_image_file(path + _)
            image_path = path + _
            encoding = fr.face_encodings(image)[0]

            known_name_encodings.append(encoding)
            known_names.append(os.path.splitext(os.path.basename(image_path))[0].lower())

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            # Detect Faces
            cv2.imshow("hello", frame)
            # frame=cv2.resize(frame,(540,450))
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            face_locations = fr.face_locations(frame)
            face_encodings = fr.face_encodings(frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = fr.compare_faces(known_name_encodings, face_encoding)
                name = ""

                face_distances = fr.face_distance(known_name_encodings, face_encoding)
                best_match = np.argmin(face_distances)

                if matches[best_match]:
                    name = known_names[best_match]

                if name in known_names:
                    session["username"] = name
                    cap.release()
                    cv2.destroyAllWindows()
                    name = session["username"]
                    dt = datetime.now()
                    # cur, conn = connections()
                    # cur.execute('INSERT INTO  user_logindetails(username,login) VALUES(%s,%s) ', (name, dt,))
                    # conn.commit()
                    # conn.close()

                    tracker = 1
                    print(tracker)
                    return render_template("success.html")
                else:
                    status = "Couldn't recognise please login with password"
                    cap.release()
                    cv2.destroyAllWindows()
                    tracker = 2
                    print(tracker)
                    return render_template("login.html", status=status)

    p4 = Thread(target=face_pro)
    p4.start()
    p4.join()
    return render_template("success.html")


@app.route('/capture')
def capture():
    def capture_pro():
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("face capture")
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("face capture", frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                status = "Couldn't capture"
                break
            elif k % 256 == 32:
                name = session['username']
                img_name = "{}.jpg".format(name)
                save_path = 'static/users_images'
                completeName = os.path.join(save_path, img_name)
                cv2.imwrite(completeName, frame)
                status = "Hey {}..! Captured your pic. ".format(name)
                break

        cam.release()
        cv2.destroyAllWindows()
        return render_template('success.html', status=status)

    p2 = Process(target=capture_pro)
    p2.start()
    p2.join()
    return render_template("success.html")


@app.route('/food_intake', methods=['POST', 'GET'])
def food_intake():
    return render_template("exercise.html")


@socketio.on('light_biceps')
def biceps(data):
    cap = cv2.VideoCapture(0)
    # print(data)
    # Curl counter variables
    stage = None
    counter = 0
    # Setup mediapipe instance

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                image_data = cv2.imencode('image_data.jpg', image_data)[1].tobytes()
                base_64_encoded = base64.b64encode(image_data).decode('utf-8')
                image_data = "data:image/jpeg;base64,{}".format(base_64_encoded)
                emit("capture_2", {'image_data': image_data})

                if counter == 1:
                    break

        cap.release()
        emit('redirect', {'url': 'light_timer'})


@app.route("/light_timer")
def light():
    print("light_timer")
    return render_template("timer.html", timer=10, counter="light_squat")


@app.route("/thanks")
def thanks():
    print("thanks")
    return render_template("thanks.html")


@app.route('/push_up', methods=['POST', 'GET'])
def push_up():
    def push_up_pro():
        # Curl counter variables
        cap = cv2.VideoCapture("static/sample_videos/pushup.mp4")
        counter = 0
        stage = None

        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Recolor image to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False

                    # Make detection
                    results = pose.process(image)

                    # Recolor back to BGR
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image = cv2.resize(image, (960, 640))

                    # Extract landmarks
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
                            stage = "down"
                        else:
                            cv2.putText(image, 'Lift up properly', (15, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        if angle < 60 and stage == 'down' and knee_angle > 160:
                            stage = "up"
                            counter += 1
                            # print(counter)

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

                    cv2.imshow('Mediapipe Feed', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

    p5 = Process(target=push_up_pro)
    p5.start()
    p5.join()
    print("2 nd return")
    return render_template("medium_workout.html")


@socketio.on('light_squat')
def squat(data):
    cap = cv2.VideoCapture(0)
    print(data)
    # Curl counter variables
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.resize(image, (700, 450))

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    # print(landmarks)

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
                                tuple(np.multiply(knee, [700, 450]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA
                                )

                    # Curl counter logic
                    if knee_angle > 170:
                        stage = "up"
                    if knee_angle < 90 and stage == 'up':
                        stage = "down"
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

                # Stage data
                cv2.putText(image, 'STAGE', (65, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage,
                            (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                 circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                image_data = cv2.resize(image, (720, 360))
                image_data = cv2.imencode('image_data.jpg', image_data)[1].tobytes()
                base_64_encoded = base64.b64encode(image_data).decode('utf-8')
                image_data = "data:image/jpeg;base64,{}".format(base_64_encoded)
                # print("cam opened")
                # print(data)
                if counter == 1:
                    # cur.execute('INSERT INTO  user_logindetails(username,squats,timeofexercise) VALUES(%s,%s,%s) ',
                    #             (name, str(counter), dt,))
                    # conn.commit()
                    break
                emit("capture_2", {'image_data': image_data})
                # cur, conn = connections()
            else:
                # cur.execute('INSERT INTO  user_logindetails(username,squats,timeofexercise) VALUES(%s,%s,%s) ',
                #             (name, str(counter), dt,))
                # conn.commit()
                break

        cap.release()
        emit('redirect', {'url': '/thanks'})


@app.route('/ipcam_squat', methods=['POST', 'GET'])
def ipcam_squat():
    def ipcam_squat_pro():
        cap = cv2.VideoCapture("http://192.168.1.103:8080/video")

        # Curl counter variables
        counter = 0
        stage = None

        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                if ret == True:
                    # Recolor image to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False

                    # Make detection
                    results = pose.process(image)

                    # Recolor back to BGR
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image = cv2.resize(image, (700, 450))

                    # Extract landmarks
                    try:
                        landmarks = results.pose_landmarks.landmark
                        # print(landmarks)

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
                                    tuple(np.multiply(knee, [700, 450]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA
                                    )

                        # Curl counter logic
                        if knee_angle > 170:
                            stage = "up"
                        if knee_angle < 90 and stage == 'up':
                            stage = "down"
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

                    # Stage data
                    cv2.putText(image, 'STAGE', (65, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage,
                                (60, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                     circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )

                    winname = "Test"
                    cv2.namedWindow(winname)  # Create a named window
                    cv2.moveWindow(winname, 0, 500)  # Move it to (40,30)
                    cv2.imshow(winname, image)

                    name = session["username"]
                    dt = datetime.now()
                    # cur, conn = connections()
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        # cur.execute('INSERT INTO  user_logindetails(username,squats,timeofexercise) VALUES(%s,%s,%s) ',
                        #             (name, str(counter), dt,))
                        # conn.commit()
                        break

            cap.release()
            cv2.destroyAllWindows()
            return render_template("heavy_workout.html", counter=counter)

    p8 = Process(target=ipcam_squat_pro)
    p8.start()
    p8.join()
    print("2 nd return")
    return render_template("medium_workout.html")


@app.route('/light_workout')
def light_workout():
    return render_template("light_workout.html")


@app.route('/medium_workout')
def medium_workout():
    return render_template("medium_workout.html")


@app.route('/heavy_workout')
def heavy_workout():
    return render_template("heavy_workout.html")


@app.route('/logout')
def logout():
    dt = datetime.now()
    name = session['username']
    # cur, conn = connections()
    # cur.execute('INSERT INTO  user_logindetails(username,logout) VALUES(%s,%s) ', (name, dt))
    # conn.commit()
    # conn.close()
    return render_template("home.html")


if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)
