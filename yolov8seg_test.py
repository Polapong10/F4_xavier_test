import os
from ultralytics import YOLO
import sys
from pathlib import Path
import cv2

model = YOLO('yolov8n-seg.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    results = model.predict(source= frame, conf = 0.7, save = False)

    frame_result = results[0].plot()

    # cv2.imshow("Frame", frame)
    cv2.imshow("Results", frame_result)

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break



