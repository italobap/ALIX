import cv2
from gtts import gTTS
import os
import time 
import subprocess
import playsound

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml') # face detection
language="pt-br"
limit_input_time = 13

def speak(text):
    tts = gTTS(text= text, lang=language)
    filename = "output.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove("output.mp3")

def presence_detection():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera")
        return

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

        if len(faces) > 0:
            # Release the camera and close the OpenCV window
            cam.release()
            cv2.destroyAllWindows()
            speak("Eba, você ainda está estudando. Você pode me responder apertando o botão.")
            break

        # Check if the face detection time has exceeded the limit
        if time.time() - face_time > limit_input_time:
            # Release the camera and close the OpenCV window
            cam.release()
            cv2.destroyAllWindows()
            speak("Não te encontrei. Encerrando atividade")
            break
        
if __name__ == '__main__':
    speak("Será que você ainda está ai? Vou te procurar.")
    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Conversation/baseRotation.py',shell=True, preexec_fn=os.setsid)
    presence_detection()
        
    
