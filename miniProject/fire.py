from ultralytics import YOLO
import cvzone
import cv2
import math
from tkinter import Tk
from tkinter.filedialog import askopenfilename



Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
video_path = askopenfilename(filetypes=[("Video files", ".mp4;.avi;*.mkv")])

if video_path:
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)

model = YOLO('best.pt')

# Class name for fire
classnames = ['fire']

while True:
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if the video ends or no frame is captured

    frame = cv2.resize(frame, (640, 480))  # Resize the frame for faster processing
    results = model(frame, stream=True)  # Run the model inference

    fire_detected = False  # Flag to track if fire was detected in the current frame

    # Iterate over the results to extract bounding box, confidence, and class
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)  # Convert to percentage
            Class = int(box.cls[0])

            # If the confidence is higher than a certain threshold (e.g., 60)
            if confidence > 60:
                fire_detected = True  # Set flag to True if fire is detected
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw a rectangle around the detected fire region
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # Display the class name and confidence percentage
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 5, y1 - 10],
                                   scale=1.5, thickness=2, offset=5, colorR=(255, 0, 0))

                # Print the detection results in the terminal for validation
                print(f'Detected: {classnames[Class]}, Confidence: {confidence}%')

    # If no fire was detected in the current frame
    if not fire_detected:
        print('No Fire Detected')

    # Display the video frame
    cv2.imshow('Fire Detection', frame)

    # Add a delay of 1ms to allow the frame to be displayed properly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit if 'q' is pressed

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
