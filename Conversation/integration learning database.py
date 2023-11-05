import os
import numpy as np
import pyaudio
import openai
from pydub import AudioSegment
from gtts import gTTS
import playsound
import time
from time import sleep
import cv2
from senha import API_KEY
import speech_recognition as sr
import requests
import json
import keyboard
push_button = Button(10)
magnetic_sensor = Button(12)
solenoid_pin = 15

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml') # face detection
limit_input_time = 7
short_pomodoro = 1500 #ciclo de 25 min
longer_pomodoro = 7200 #4 ciclos de 30 min

language="pt-br"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def get_transcription_from_whisper():
    # Set the audio parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 4096
    SILENCE_THRESHOLD = 1000  # Silence threshold
    dev_index = 0  # Device index found by p.get_device_info_by_index(ii)
    SPEECH_END_TIME = 1.0  # Time of silence to mark the end of speech

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Initialize variables to track audio detection

    try:
        # Start Recording rasp
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input_device_index=dev_index,
            input=True,
            frames_per_buffer=CHUNK
        )

        print("Recording... Waiting for speech to begin.")
        detection_time = time.time()

        frames = []
        silence_frames = 0
        is_speaking = False

        while True:
            current_time = time.time()
            data = stream.read(CHUNK)
            frames.append(data)

            # Convert audio chunks to integers
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Check if the user has started speaking
            if np.abs(audio_data).mean() > SILENCE_THRESHOLD:
                is_speaking = True

            # Detect if the audio chunk is silence
            if is_speaking:
                if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
                    silence_frames += 1
                else:
                    silence_frames = 0

            # End of speech detected
            if is_speaking and silence_frames > int(SPEECH_END_TIME * (RATE / CHUNK)):
                print("End of speech detected.")
                combined_audio_data = b''.join(frames)
                audio_segment = AudioSegment(
                    data=combined_audio_data,
                    sample_width=audio.get_sample_size(FORMAT),
                    frame_rate=RATE,
                    channels=CHANNELS
                )

                audio_segment.export("output_audio_file.mp3", format="mp3", bitrate="32k")

                with open("output_audio_file.mp3", "rb") as f:
                    transcript = openai.Audio.transcribe("whisper-1", f,language = language_whisper)

                transcription = transcript['text']
                print("Transcript:", transcription.lower())
                return transcription
                break

            # Check if it has been more than 10 seconds without speech
            if current_time - detection_time >= limit_input_time:
                print("No speech detected within the last 10 seconds. Stopping recording.")
                speak("Não escutei o que você falou. Aperte o botão de novo para falar comigo.")
                break

    except Exception as e:
        print("Error during audio recording:", str(e))
    finally:
        # Always stop and close the stream and terminate audio
        stream.stop_stream()
        stream.close()
        audio.terminate()

def speak(text):
    tts = gTTS(text= text, lang=language)
    filename = "output.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove("output.mp3")

def speake(text):
    tts = gTTS(text= text, lang='en')
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
            speak("Você ainda está aí. Você pode me responder apertando o botão.")
            break

        # Check if the face detection time has exceeded the limit
        if time.time() - face_time > limit_input_time:
            # Release the camera and close the OpenCV window
            cam.release()
            cv2.destroyAllWindows()
            speak("Não te encontrei.")
            break

def getQuestion(lesson, i):
    f = open(f"Questionnaires/{lesson}", "r")
    content = f.readlines()
    end = content[i].find(',')
    return content[i][0:end]
  
def getAnswer(lesson, i):
    f = open(f"Questionnaires/{lesson}", "r")
    content = f.readlines()
    begin = content[i].find(',') + 1
    end = content[i].find('/n')
    return content[i][begin:end]

def learning_mode():
    while True:
        if keyboard.read_key() == "r":
                frase = get_transcription_from_whisper()
                if frase is not None:
                    if "colors" and "1" in frase:
                        speak(getQuestion("Colors",1))
                        speak_time = time.time()
                        total_time = time.time()
                        sleep(0.50)
                        while True:
                            if keyboard.read_key() == "r":
                                frase = get_transcription_from_whisper()
                                if getAnswer("Colors",1).lower() in frase:
                                    speake("That is correct.")
                                    sleep(0.50)
                                    speak("Atividade finalizada.Parabéns!")
                                    notal = 10-erro
                                    final_time = (time.time() - total_time)/60 #Jogar final_time no banco de dados
                                    break
                                else:
                                    speake("That is incorrect. Try again")
                                    erro =+1
                            if time.time() - speak_time > limit_input_time:
                                speak("Você ainda está ai?")
                                presence_detection()
                                speak_time = time.time()
                            if time.time() - speak_time > short_pomodoro:
                                speak("Está na hora da sua pausa de 5 minutos.")
                                sleep(300)
                                speak_time = time.time()
                            if time.time() - speak_time > longer_pomodoro:
                                speak("Está na hora da sua pausa de 15 minutos.")
                                sleep(900)
                                speak_time = time.time()
                    if "parar" in frase:
                        speak("Certo, finalizando modo de estudo.")
                        break

def GPIO_Init():
    solenoid_pin = 15
    push_button_pin = 19 #gpio10
    magnetic_sensor_pin = 32 #gpio12
    limit_time = 5
    GPIO.setwarnings(False) # Ignore warning for now
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    GPIO.setup(solenoid_pin, GPIO.OUT) 
    GPIO.setup(magnetic_sensor_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(push_button_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    
def lockable_compartment():
    GPIO.output(solenoid_pin, True)
    while True:
        if magnetic_sensor.is_pressed:
            GPIO.output(solenoid_pin, True)
        else:
            GPIO.output(solenoid_pin, False)
            break
    while True:
        if magnetic_sensor.is_pressed:
            speak("Compartimento de recompensas ativado.")
            break

if __name__ == '__main__':
    GPIO_Init()
    while True:
        if keyboard.read_key() == "r":
            frase = get_transcription_from_whisper()
            if frase is not None:
                if "study" in frase:
                    speak("Certo. Vamos aprender inglês.")
                    sleep(0.50)
                    speak("Você vai utilizar o compartimento de recompensas?")
                    while True:
                        if keyboard.read_key() == "r":
                            if frase is not None:
                                if "sim" in frase:
                                    speak("Certo. Pode abrir a porta para utilizar o compartimento de recompensas")
                                    lockable_compartment()
                                    break
                                if "não" in frase:
                                    speak("Ok. Compartimento de recompensas desativado.")
                                    break

                    speak("Você gostaria de usar a camera para detecção de presença durante as atividades?")
                    while True:
                        if keyboard.read_key() == "r":
                            frase = get_transcription_from_whisper()
                            if frase is not None:
                                if "sim" in frase:
                                    presence_detection()
                                    speak("Ok, detecção de prenseça ativado.")
                                    break
                                else:
                                    speak("Tudo bem, não irei utilizar a detecção de presença.")
                                    break

                    speak("Qual atividade você vai realizar?")
                    learning_mode()            

                if "bye" in frase:
                    speak("Até mais, mal posso esperar para conversar com você de novo.")
                    break