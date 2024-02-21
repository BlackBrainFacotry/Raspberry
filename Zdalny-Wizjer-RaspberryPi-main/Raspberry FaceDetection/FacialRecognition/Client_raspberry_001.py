import asyncio
import cv2
import numpy as np
import os
import struct
from PIL import Image
from imutils import paths
from websockets import connect

# Configuration Constants
CASCADE_PATH = "haarcascade_frontalface_default.xml"
TRAINER_PATH = 'trainer/trainer.yml'
URI = "ws://192.168.0.108:5050/"
DATASET_PATH = 'dataset'

# Setup Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)
faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
font = cv2.FONT_HERSHEY_SIMPLEX

# Function to process each frame and detect faces
async def process_frame(websocket):
    cam = cv2.VideoCapture(0)
    cam.set(3, 480)  # Video width
    cam.set(4, 640)  # Video height
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.flip(frame, -1)  # Flip vertically
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 100:
                name = names[id]
                confidence_text = f"  {round(100 - confidence)}%"
            else:
                name = "unknown"
                confidence_text = f"  {round(100 - confidence)}%"

            cv2.putText(frame, str(name), (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence_text), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

        picture_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        command = "{:<100}".format("Get_picture")
        message = command.encode() + picture_bytes
        await websocket.send(message)

        cv2.imshow("TRANSMITTING TO CACHE SERVER", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


async def train_model(client_socket):
	# Path for face image database
	path = 'dataset'
	if not os.path.exists('trainer'):os.makedirs('trainer')
		
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

	

	print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
	faces,ids = getImagesAndLabels(path)
	recognizer.train(faces, np.array(ids))

	# Save the model into trainer/trainer.yml
	recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

	# Print the numer of faces trained and end program
	print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

	message = "{:<100}".format("Get_train_model")
	await client_socket.send(message.encode())      


# Function to handle commands from websocket
async def handle_commands(websocket):
    await return_name_client(websocket)
    while True:
        data = await websocket.recv()
        command = data[:100].decode().strip()
        if command == "Get_picture":
            await process_frame(websocket)
        elif command == "Get_person_list":
            await return_profile_list(websocket)
        elif command == "Get_train_model":
            await train_model(websocket)

# Main async function to start the websocket connection
async def main():
    async with connect(URI, ping_interval=None) as websocket:
        await handle_commands(websocket)

# Start the asyncio event loop
if __name__ == "__main__":
    print('START')
    asyncio.run(main())
