import cv2
import os

def create_profile_directory(profile_name, base_path):
    """Create a directory for a new profile if it doesn't already exist."""
    profile_path = os.path.join(base_path, profile_name)
    if not os.path.exists(profile_path):
        os.makedirs(profile_path)
        print('Profile created: {}'.format(profile_name))
        return profile_path
    else:
        print('This profile already exists in the database.')
        return None

def capture_images(profile_path):
    """Capture images from the webcam and save them in the specified profile directory."""
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Press SPACE to take a photo, ESC to exit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Press SPACE to take a photo, ESC to exit", 640, 480)

    img_counter = 0

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, -1)  # Flip vertically
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Press SPACE to take a photo, ESC to exit", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = os.path.join(profile_path, "{}_{:04d}.jpg".format(profile_name, img_counter))
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

def main():
    base_path = '/home/pi/Desktop/piFace-master/FacialRecognition/dataset'
    profiles = os.listdir(base_path)
    print('Profile list:')
    for profile in profiles:
        print(profile)
    print('')

    profile_name = input("Write the name of your profile: ")
    profile_path = create_profile_directory(profile_name, base_path)
    if profile_path:
        capture_images(profile_path)

if __name__ == "__main__":
    main()
