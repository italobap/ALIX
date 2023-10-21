import cv2

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml') # insert the full path to haarcascade file if you encounter any problem


cam = cv2.VideoCapture(0)

while True:
    ret, image = cam.read()
    cv2.imshow('webcam',image)
    key = cv2.waitKey(1) & 0xFF
    k = cv2.waitKey(1)
    if key == ord('p'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_classifier.detectMultiScale(gray)
        if(faces == ()):
            print("rosto não encontrado")
        else:
            print("rosto encontrado")
            cv2.imwrite('/home/pi/testimage.jpg', image)

    if key == ord('q'):
        cam.release()
        cv2.destroyAllWindows()
        print("Programa encerrado")
        break
