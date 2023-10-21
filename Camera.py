import cv2

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml') # insert the full path to haarcascade file if you encounter any problem


while True:
    cam = cv2.VideoCapture(0)
    ret, image = cam.read()
    cv2.imshow('webcam',image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_classifier.detectMultiScale(gray)
    if(faces != ()):
        cam.release()
        cv2.destroyAllWindows()
        print("Rosto encontrado")
        break
