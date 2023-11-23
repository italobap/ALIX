import cv2
import time
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml') # insert the full path to haarcascade file if you encounter any problem






presence = False
presence_time = 10




consecutive_face_count = 0  # Track consecutive face detections
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open camera")

face_time = time.time()

while True:
    # Capture a frame from the webcam
    ret, image = cam.read()
    if not ret:
        print("Error: Could not read frame")
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_classifier.detectMultiScale(gray)
    print(faces)

    if len(faces) > 0:
        consecutive_face_count += 1
    else:
        consecutive_face_count = 0  # Reset count if no face detected

    if consecutive_face_count >= 5:
        cam.release()
        cv2.destroyAllWindows()
        print("Detected face consistently. You can respond by pressing the button.")
        presence = True
        break
 
    # Check if the face detection time has exceeded the limit
    if time.time() - face_time > presence_time:
        # Release the camera and close the OpenCV window
        cam.release()
        cv2.destroyAllWindows()
        print("Não te encontrei, finalizando atividade.")
        presence = False
        break
