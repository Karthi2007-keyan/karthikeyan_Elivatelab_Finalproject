import cv2
import numpy as np
from tensorflow.keras.models import load_model
from threading import Thread
import os

# Load model and face detector
model = load_model("model/mask_detector.keras")
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

# Labels and colors
labels_dict = {0: 'Mask', 1: 'No Mask'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

# Load alert sound
def alert_sound():
    try:
        wave_obj = sa.WaveObject.from_wave_file("alert.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"[ERROR] Playing sound failed: {e}")

# Open webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam mask detection. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 3))

        result = model.predict(reshaped, verbose=0)
        label = np.argmax(result[0])

        label_text = labels_dict[label]
        color = color_dict[label]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Alert only if no mask detected
        if label == 1:
            Thread(target=alert_sound, daemon=True).start()

    # Show frame
    cv2.imshow("Live Mask Detection", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
