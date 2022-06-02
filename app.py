from flask import Flask, render_template
import face_recognition
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register():
    import cv2

    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            # print(check)  # prints true as long as the webcam is running
            # print(frame)  # prints matrix values of each framecd
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(filename='registered_img.jpg', img=frame)
                webcam.release()
                # img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                # img_new = cv2.imshow("Captured Image", img_new)
                # cv2.waitKey(1650)
                cv2.destroyAllWindows()
                # print("Processing image...")
                # img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                # # print("Converting RGB image to grayscale...")
                # gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                # # print("Converted RGB image to grayscale...")
                # # print("Resizing image to 28x28 scale...")
                # img_ = cv2.resize(gray, (28, 28))
                # print("Resized...")
                # img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
                # print("Image saved!")

                break
            elif key == ord('q'):
                # print("Turning off camera.")
                webcam.release()
                # print("Camera off.")
                # print("Program ended.")
                cv2.destroyAllWindows()
                break

        except(KeyboardInterrupt):
            # print("Turning off camera.")
            webcam.release()
            # print("Camera off.")
            # print("Program ended.")
            cv2.destroyAllWindows()
            break

    return render_template('index.html')


@app.route('/login')
def login():
    import cv2
    results = False
    # key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            # print(check)  # prints true as long as the webcam is running
            # print(frame)  # prints matrix values of each framecd
            cv2.imshow("Capturing", frame)
            # key = cv2.waitKey(1)
            if True:
                cv2.imwrite(filename='login_img.jpg', img=frame)
                # webcam.release()
                # img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                # img_new = cv2.imshow("Captured Image", img_new)
                # cv2.waitKey(1650)
                # cv2.destroyAllWindows()
                # # print("Processing image...")
                # img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                # # print("Converting RGB image to grayscale...")
                # gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                # # print("Converted RGB image to grayscale...")
                # # print("Resizing image to 28x28 scale...")
                # img_ = cv2.resize(gray, (28, 28))
                # # print("Resized...")
                # # img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
                # # print("Image saved!")

            if cv2.waitKey(10) & 0xff == ord('q'):
                # print("Turning off camera.")
                webcam.release()
                # print("Camera off.")
                # print("Program ended.")
                cv2.destroyAllWindows()
                break

        except(KeyboardInterrupt):
            # print("Turning off camera.")
            webcam.release()
            # print("Camera off.")
            # print("Program ended.")
            cv2.destroyAllWindows()
            break

        known_image = face_recognition.load_image_file("registered_img.jpg")
        unknown_image = face_recognition.load_image_file("login_img.jpg")
        modi_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        results = face_recognition.compare_faces([modi_encoding], unknown_encoding)
        print(results)
        if results:
            webcam.release()
            # print("Camera off.")
            # print("Program ended.")
            cv2.destroyAllWindows()
            return render_template("exercise.html")

    return render_template('index.html', )


@app.route('/exercise')
def exercise():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(-1)

    # Curl counter variables
    counter = 0
    stage = None

    ## Setup mediapipe instance

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            print("camersa")
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            cv2.imshow('Mediapipe Feed', image)
            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (960, 640))

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
                print(angle)
                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 40 and stage == 'down':
                    stage = "up"
                    counter += 1
                    # print(counter)

            except:
                print("in ecxept block")
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

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return render_template("/exercise.html", count=counter)


if __name__ == "__main__":
    app.run(debug=True)
