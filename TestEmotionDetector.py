import cv2
import numpy as np
from keras.models import model_from_json
import time
import pyttsx3
import threading
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
def speak_alert(text):
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech).start()


# Define emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load emotion model
with open('model/emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded emotion model from disk")

# Load Haar cascades
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Drowsiness settings
EYE_AR_THRESH = 2
DROWSY_TIME = 2.0
drowsy_start = None

# Input source (use "webcam" or path to image/video)
file_path = "webcam"


def resize_frame(frame, max_width=800):
    height, width = frame.shape[:2]
    if width > max_width:
        scaling_factor = max_width / width
        frame = cv2.resize(frame, (int(width * scaling_factor), int(height * scaling_factor)))
    return frame

def process_frame(frame):
    global drowsy_start
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Emotion Detection
        resized_face = cv2.resize(roi_gray, (48, 48)) / 255.0  # Normalize
        cropped_img = np.expand_dims(np.expand_dims(resized_face, -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img, verbose=0)

        maxindex = int(np.argmax(emotion_prediction))
        emotion = emotion_dict[maxindex]
        cv2.putText(frame, f"Emotion: {emotion}", (x + 5, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4, cv2.LINE_AA)

        # Drowsiness Detection
        eyes = eye_detector.detectMultiScale(roi_gray)
        eye_count = len(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        if eye_count < EYE_AR_THRESH:
            if drowsy_start is None:
                drowsy_start = time.time()
            elif time.time() - drowsy_start >= DROWSY_TIME:
                cv2.putText(frame, "DROWSY ALERT!", (x, y + h + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                speak_alert("You are feeling drowsy. Please take a break.")
        else:
            drowsy_start = None

    return frame

# ============================== #
# Webcam / Image / Video Source #
# ============================== #
if file_path.lower() == "webcam":
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        frame = resize_frame(frame)
        cv2.imshow("Emotion + Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

elif os.path.exists(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        frame = cv2.imread(file_path)
        if frame is not None:
            frame = process_frame(frame)
            frame = resize_frame(frame)
            cv2.imshow("Emotion + Drowsiness Detection - Image", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not load image.")
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame)
            frame = resize_frame(frame)
            cv2.imshow("Emotion + Drowsiness Detection - Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unsupported file type.")
else:
    print("File not found.")
