import cv2

# Constants for the script
CASCADE_PATH = 'Cascades/haarcascade_frontalface_default.xml'
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
ESC_KEY_CODE = 27

def main():
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print("Error loading cascade file. Check the path.")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(3, VIDEO_WIDTH)  # set video width
    cap.set(4, VIDEO_HEIGHT)  # set video height

    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        img = cv2.flip(img, -1)  # Flip the image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('video', img)

        k = cv2.waitKey(30) & 0xff
        if k == ESC_KEY_CODE:  # Press 'ESC' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
