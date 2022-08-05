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
from flask_socketio import SocketIO, send, emit
import numpy as np
import zmq
import psycopg2
from time import sleep

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

tracker = None
status = None

app = Flask(__name__)
socketio = SocketIO(app)

app.secret_key = "ussv"


# Connect to your PostgresSQL database on a remote server
def connections():
    conn = psycopg2.connect(host="127.0.0.1", port="5432", dbname="user_details", user="postgres", password="p@ssw0rd")

    # Open a cursor to perform database operations
    cur = conn.cursor()
    return cur, conn


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


@app.route('/check', methods=['POST', 'GET'])
def check():
    username = request.form.get('username')
    password = request.form.get('password')
    cur, conn = connections()

    with conn:
        cur.execute(f"SELECT * FROM details WHERE username=%(username)s AND password=%(password)s",
                    {'username': username, 'password': password})

        if not cur.fetchall():
            return render_template("home.html")
        else:
            session["username"] = request.form.get("username")
            name = session["username"]
            dt = datetime.now()
            cur.execute('INSERT INTO  user_logindetails(username,login) VALUES(%s,%s) ', (name, dt,))
            conn.commit()
            # conn.close()
            return render_template("success.html")


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/light_workout')
def light_workout():
    return render_template("light_workout.html")


@app.route('/medium_workout')
def medium_workout():
    return render_template("medium_workout.html")


@app.route('/heavy_workout')
def heavy_workout():
    return render_template("heavy_workout.html")


@app.route('/add', methods=['POST'])
def add():
    if request.method == 'POST':
        name = request.form.get('name')
        username = request.form.get('username')
        height = request.form.get('height')
        password = request.form.get('password')
        cur, conn = connections()
        check_in_db = "SELECT * from details where username like %s"
        cur.execute(check_in_db, [username])
        result = cur.fetchall()
        print(result)
        if len(result) >= 1:
            msg = "user name already exists, Register with other username"
            return render_template('register.html', msg=msg)
        else:
            cur.execute('INSERT INTO details(name,username,height,password) VALUES (%s,%s,%s,%s)',
                        (name, username, height, password))
            conn.commit()
            # conn.close()
            msg = 'Registered successfully'
            return render_template('login.html', msg=msg)


@socketio.on('message')
def hello(data):
    print(data)
    return render_template("light_workout.html")


@socketio.on("face")
def face(data):
    print(data)

    path = "static/users_images/"

    known_names = []
    known_name_encodings = []
    cur, conn = connections()
    images = os.listdir(path)
    for _ in images:
        image = fr.load_image_file(path + _)
        # print(image)
        image_path = path + _
        # print(image_path)
        encoding = fr.face_encodings(image)[0]

        known_name_encodings.append(encoding)
        known_names.append(os.path.splitext(os.path.basename(image_path))[0].lower())
        cap = cv2.VideoCapture(0) #"http://192.168.29.149:8080/video"
        if cap.isOpened:
            while True:
                ret, frame = cap.read()
                image_data = cv2.resize(frame, (250, 250))
                image_data = cv2.imencode('image_data.jpg', image_data)[1].tobytes()
                base_64_encoded = base64.b64encode(image_data).decode('utf-8')
                image_data = "data:image/jpeg;base64,{}".format(base_64_encoded)
                send({'image_data': image_data})
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
                        print(session["username"])
                        cap.release()
                        name = session["username"]
                        print(name)
                        dt = datetime.now()
                        cur.execute('INSERT INTO  user_logindetails(username,login) VALUES(%s,%s) ', (name, dt,))
                        conn.commit()
                        emit('redirect', {'url': url_for('success')})

                    else:
                        status = "Couldn't recognise please login with password"
                        cap.release()
                        emit('redirect', {'url': url_for('login')})


@app.route('/success')
def success():
    return render_template("success.html")


@socketio.on('capture')
def save_img(img_base64):
    header, data = img_base64.split(',', 1)
    image_data = base64.b64decode(data)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    # print(session['username'])
    name = session['username']
    # print(name)
    img_name = "{}.jpg".format(name)
    save_path = 'static/users_images'
    completeName = os.path.join(save_path, img_name)
    cv2.imwrite(completeName, image)
    status = "Hey {}..! Captured your pic. ".format(name)
    sleep(1.5)
    emit('redirect', {'url': url_for('success')})


@app.route('/capture')
def capture():
    return render_template('face.html')


@app.route('/food_intake', methods=['POST', 'GET'])
def food_intake():
    return render_template("exercise.html")


@socketio.on('light_biceps')
def biceps(data):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:7000")
    cap = cv2.VideoCapture("http://192.168.29.149:8080/video")  # "http://192.168.29.149:8080/video"
    while cap.isOpened():
        try:
            ret, im0 = cap.read()
            im0_small = cv2.resize(im0, (720, 480))
            print("shape:", im0_small.shape)
            im0_small = cv2.imencode('.jpg', im0_small)[1].tobytes()
            base_64_encoded = base64.b64encode(im0_small).decode('utf-8')
            str_img_base_64 = "data:image/jpeg;base64,{}".format(base_64_encoded)
            video_feed = {"base_64": str_img_base_64}
            # sending frames to detect.py as bytes form
            socket.send_string(str_img_base_64)
            # receiving frame after tracking from the detect.py
            message = socket.recv_pyobj()
            counter = message["counter"]
            video = {"base_64": message["image_data"]}
            if counter == 5:
                break
            # emitting frame as bytes into html page
            emit("capture_2", video)

        except:
            pass
    cap.release()
    emit('redirect', {'url': 'light_timer'})


@socketio.on('medium_lunges')
def biceps(data):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:7003")
    cap = cv2.VideoCapture("static/sample_videos/lunge_1.mp4")  # "http://192.168.29.149:8080/video"
    while cap.isOpened():
        try:
            ret, im0 = cap.read()
            im0_small = cv2.resize(im0, (720, 480))
            print("shape:", im0_small.shape)
            im0_small = cv2.imencode('.jpg', im0_small)[1].tobytes()
            base_64_encoded = base64.b64encode(im0_small).decode('utf-8')
            str_img_base_64 = "data:image/jpeg;base64,{}".format(base_64_encoded)
            video_feed = {"base_64": str_img_base_64}
            # sending frames to detect.py as bytes form
            socket.send_string(str_img_base_64)
            # receiving frame after tracking from the detect.py
            message = socket.recv_pyobj()
            counter = message["counter"]
            video = {"base_64": message["image_data"]}
            if counter == 5:
                break
            # emitting frame as bytes into html page
            emit("capture_2", video)

        except:
            pass
    cap.release()
    emit('redirect', {'url': 'medium_timer'})


@app.route("/light_timer")
def light():
    print("light_timer")
    return render_template("timer.html", timer=5, counter="light_squat")


@socketio.on('heavy_short_head_biceps')
def heavy_short_head_biceps(data):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:7004")
    cap = cv2.VideoCapture("static/sample_videos/short head biceps.mp4")
    while cap.isOpened():
        try:
            ret, im0 = cap.read()
            im0_small = cv2.resize(im0.copy(), (480, 360))
            im0_small = cv2.imencode('.jpg', im0_small)[1].tobytes()
            base_64_encoded = base64.b64encode(im0_small).decode('utf-8')
            str_img_base_64 = "data:image/jpeg;base64,{}".format(base_64_encoded)
            video_feed = {"base_64": str_img_base_64}
            # sending frames to detect.py as bytes form
            socket.send_string(str_img_base_64)
            # receiving frame after tracking from the detect.py
            message = socket.recv_pyobj()
            counter = message["counter"]
            video = {"base_64": message["image_data"]}
            if counter == 4:
                break
            # emitting frame as bytes into html page
            emit("capture_2", video)

        except:
            pass
    cap.release()
    emit('redirect', {'url': 'heavy_timer'})


@app.route("/heavy_timer")
def heavy():
    print("light_timer")
    return render_template("timer.html", timer=5, counter="heavy_pushup")


@app.route("/medium_timer")
def medium():
    print("light_timer")
    return render_template("timer.html", timer=5, counter="medium_pushup")


@app.route("/thanks")
def thanks():
    print("thanks")
    return render_template("thanks.html")


@socketio.on('light_squat')
def squat(data):
    print(data)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:7001")
    cap = cv2.VideoCapture("http://192.168.29.149:8080/video")  # "http://192.168.29.149:8080/video"
    while cap.isOpened():
        try:
            ret, im0 = cap.read()
            im0_small = cv2.resize(im0.copy(), (480, 360))
            im0_small = cv2.imencode('.jpg', im0_small)[1].tobytes()
            base_64_encoded = base64.b64encode(im0_small).decode('utf-8')
            str_img_base_64 = "data:image/jpeg;base64,{}".format(base_64_encoded)
            video_feed = {"base_64": str_img_base_64}
            # sending frames to detect.py as bytes form
            socket.send_string(str_img_base_64)
            # receiving frame after tracking from the detect.py
            message = socket.recv_pyobj()
            counter = message["counter"]
            video = {"base_64": message["image_data"]}
            if counter == 3:
                break
            # emitting frame as bytes into html page
            emit("capture_2", video)

        except:
            pass
    cap.release()
    emit('redirect', {'url': '/thanks'})


@socketio.on('medium_squat')
def medium_squat(data):
    print(data)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:7001")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        try:
            ret, im0 = cap.read()
            im0_small = cv2.resize(im0.copy(), (480, 360))
            im0_small = cv2.imencode('.jpg', im0_small)[1].tobytes()
            base_64_encoded = base64.b64encode(im0_small).decode('utf-8')
            str_img_base_64 = "data:image/jpeg;base64,{}".format(base_64_encoded)
            video_feed = {"base_64": str_img_base_64}
            # sending frames to detect.py as bytes form
            socket.send_string(str_img_base_64)
            # receiving frame after tracking from the detect.py
            message = socket.recv_pyobj()
            counter = message["counter"]
            video = {"base_64": message["image_data"]}
            if counter == 3:
                break
            # emitting frame as bytes into html page
            emit("capture_2", video)

        except:
            pass
    cap.release()
    emit('redirect', {'url': 'medium_timer'})


@socketio.on('medium_pushup')
def medium_pushup(data):
    print(data)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:7002")
    cap = cv2.VideoCapture("static/sample_videos/pushup.mp4")
    while cap.isOpened():
        try:
            ret, im0 = cap.read()
            im0_small = cv2.resize(im0.copy(), (720, 360))
            im0_small = cv2.imencode('.jpg', im0_small)[1].tobytes()
            base_64_encoded = base64.b64encode(im0_small).decode('utf-8')
            str_img_base_64 = "data:image/jpeg;base64,{}".format(base_64_encoded)
            video_feed = {"base_64": str_img_base_64}
            # sending frames to detect.py as bytes form
            socket.send_string(str_img_base_64)
            # receiving frame after tracking from the detect.py
            message = socket.recv_pyobj()
            counter = message["counter"]
            video = {"base_64": message["image_data"]}
            if counter == 3:
                break
            # emitting frame as bytes into html page
            emit("capture_2", video)

        except:
            pass
    cap.release()
    emit('redirect', {'url': '/thanks'})


@socketio.on('heavy_pushup')
def heavy_pushup(data):
    print(data)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:7002")
    cap = cv2.VideoCapture("static/sample_videos/pushup.mp4")
    while cap.isOpened():
        try:
            ret, im0 = cap.read()
            im0_small = cv2.resize(im0.copy(), (720, 360))
            im0_small = cv2.imencode('.jpg', im0_small)[1].tobytes()
            base_64_encoded = base64.b64encode(im0_small).decode('utf-8')
            str_img_base_64 = "data:image/jpeg;base64,{}".format(base_64_encoded)
            video_feed = {"base_64": str_img_base_64}
            # sending frames to detect.py as bytes form
            socket.send_string(str_img_base_64)
            # receiving frame after tracking from the detect.py
            message = socket.recv_pyobj()
            counter = message["counter"]
            video = {"base_64": message["image_data"]}
            if counter == 5:
                break
            # emitting frame as bytes into html page
            emit("capture_2", video)

        except:
            pass
    cap.release()
    emit('redirect', {'url': '/thanks'})


@app.route('/logout')
def logout():
    dt = datetime.now()
    name = session['username']
    cur, conn = connections()
    cur.execute('INSERT INTO  user_logindetails(username,logout) VALUES(%s,%s) ', (name, dt))
    conn.commit()
    conn.close()
    return render_template("home.html")


if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False, host='192.168.29.20')  # '192.168.29.20'
