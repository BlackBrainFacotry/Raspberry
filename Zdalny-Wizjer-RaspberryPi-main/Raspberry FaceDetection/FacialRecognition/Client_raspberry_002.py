import asyncio
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
from imutils import paths
import websockets
from websockets import connect

CASCADE_PATH = "haarcascade_frontalface_default.xml"
TRAINER_PATH = 'trainer/trainer.yml'
DATASET_PATH = '/home/pi/Desktop/piFace-master/FacialRecognition/dataset/'
WEBSOCKET_URI = "ws://192.168.0.108:5050/"

async def start_video_stream(websocket):
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    min_w = 0.1 * cam.get(3)
    min_h = 0.1 * cam.get(4)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        frame = process_frame(frame, face_cascade, recognizer, font, min_w, min_h)
        await send_frame(websocket, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def process_frame(frame, face_cascade, recognizer, font, min_w, min_h):
    frame = cv2.flip(frame, -1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(min_w), int(min_h)))
    names = get_names()

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 100:
            name = names[id] if id < len(names) else "unknown" 
            confidence_text = f"{confidence}%"
        else:
            name = "unknown"
            confidence_text = f"{confidence}%"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    return frame

async def send_frame(websocket, frame):
    _, buffer = cv2.imencode('.jpg', frame)
    await websocket.send(buffer.tobytes())

def get_names():
    return list(paths.list_images(DATASET_PATH))

async def websocket_handler():
    async with connect(WEBSOCKET_URI, ping_interval=None) as websocket:
        await start_video_stream(websocket)

if __name__ == "__main__":
    print('Starting...')
    asyncio.run(websocket_handler())
