from flask import *
from flask import render_template
from flask import request
import os.path
import cv2
import numpy as np
import face_recognition as fr
import os
import psycopg2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Connect to your PostgreSQL database on a remote server
conn = psycopg2.connect(host="127.0.0.1", port="5432", dbname="postgres", user="postgres", password="p@ssw0rd")

# Open a cursor to perform database operations
cur = conn.cursor()

app = Flask(__name__)
app.secret_key = "aishwarya"


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/add', methods=['POST'])
def add():
    if request.method == 'POST':
        name = request.form.get('name')
        height = request.form.get('height')
        password = request.form.get('password')

        cur.execute('INSERT INTO details(name,height,password) VALUES (%s,%s,%s)', (name, height, password))
        conn.commit()

        return render_template('login.html')


@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/face')
def face():
    path = "static/users_images/"

    known_names = []
    known_name_encodings = []

    images = os.listdir(path)
    for _ in images:
        image = fr.load_image_file(path + _)
        image_path = path + _
        encoding = fr.face_encodings(image)[0]

        known_name_encodings.append(encoding)
        known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Detect Faces
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
                cap.release()
                cv2.destroyAllWindows()
                return render_template('success.html')
            else:
                cap.release()
                cv2.destroyAllWindows()
                return render_template('home.html')


@app.route('/capture')
def capture():
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
            break
        elif k % 256 == 32:
            # SPACE pressed
            count = 1
            name = session['name']
            img_name = "{}.jpg".format(name)
            save_path = '/home/neosoft/Desktop/Aishwarya_codes/face/static/users_images'
            completeName = os.path.join(save_path, img_name)
            cv2.imwrite(completeName, frame)
            status = "Hey {}..! Captured your pic. ".format(name)
            break

    cam.release()
    cv2.destroyAllWindows()
    return render_template('success.html', status=status)


@app.route('/check', methods=['POST', 'GET'])
def check():
    name = request.form.get('name')
    password = request.form.get('password')

    with conn:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM details WHERE name=%(name)s AND password=%(password)s",
                    {'name': name, 'password': password})

        if not cur.fetchall():
            return render_template("home.html")
        else:
            session["name"] = request.form.get("name")
            return render_template("success.html")


@app.route('/biceps', methods=['POST', 'GET'])
def biceps():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0
    stage = None

    ## Setup mediapipe instance

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 40 and stage == 'down':
                    stage = "up"
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
            #         cv2.putText(image, 'STAGE', (65,12),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            #         cv2.putText(image, stage,
            #                     (60,60),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # cv2.imshow('Mediapipe Feed', image)

            winname = "Test"
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname, 700, 700)  # Move it to (40,30)
            cv2.imshow(winname, image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return render_template("success.html", count=counter)


@app.route('/push_up', methods=['POST', 'GET'])
def push_up():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

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
                            tuple(np.multiply(elbow, [700, 450]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle > 170:
                    stage = "down"
                if angle < 60 and stage == 'down':
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
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # cv2.imshow('Mediapipe Feed', image)

            winname = "Test"
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname, 700, 700)  # Move it to (40,30)
            cv2.imshow(winname, image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return render_template("success.html", count=counter)


@app.route('/squat', methods=['POST', 'GET'])
def squat():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

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

            # cv2.imshow('Mediapipe Feed', image)

            winname = "Test"
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname, 700, 700)  # Move it to (40,30)
            cv2.imshow(winname, image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return render_template("success.html", count=counter)


if __name__ == '__main__':
    app.run(debug=True)
