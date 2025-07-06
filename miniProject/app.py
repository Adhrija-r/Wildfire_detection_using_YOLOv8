from flask import Flask, render_template, request, redirect, url_for, Response, flash
from ultralytics import YOLO
import cv2
import cvzone
import math
import os

app = Flask(__name__)
app.secret_key = 'secret'

# Load YOLO model
model = YOLO('best.pt')


# Route for the homepage where users can upload a video
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle video upload
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file:
        filepath = os.path.join('static', file.filename)
        file.save(filepath)
        return redirect(url_for('detect_fire', filename=file.filename))


# Route to process the video and detect fire
@app.route('/detect/<filename>')
def detect_fire(filename):
    filepath = os.path.join('static', filename)

    def generate_frames():
        cap = cv2.VideoCapture(filepath)
        classnames = ['fire']

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            result = model(frame, stream=True)

            # Loop through each detection
            fire_detected = False
            for info in result:
                boxes = info.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    Class = int(box.cls[0])
                    if confidence > 50:
                        fire_detected = True  # Fire detected
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%',
                                           [x1 + 8, y1 + 100], scale=1.5, thickness=2)

            if fire_detected:
                # Log fire alert (could be an SMS/email notification)
                print("ALERT: Fire detected!")

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
