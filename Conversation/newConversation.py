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
import subprocess
import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml') # face detection
limit_input_time = 7
language="pt-br"
language_whisper="pt"
# Set your OpenAI API key
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
link = "https://api.openai.com/v1/chat/completions"

solenoid_pin = 15
push_button_pin = 19 #gpio10
magnetic_sensor_pin = 32 #gpio12

expressionAddress = '/home/alix/Documents/ALIX/ALIX/DisplayLab/'
processRun = True
p = subprocess.Popen('exec ' + expressionAddress + 'standby', shell=True, preexec_fn=os.setsid)
    
def runProcess(expressionName):
    global processRun
    global p
    executando = 'exec ' + expressionAddress + expressionName
    if processRun:
        p.kill()
        
    p=subprocess.Popen(executando, shell=True, preexec_fn=os.setsid)
    processRun = True
    
def get_transcription_from_whisper():
    # Set the audio parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 4096
    SILENCE_THRESHOLD = 1000  # Silence threshold
    dev_index = 1  # Device index found by p.get_device_info_by_index(ii)
    SPEECH_END_TIME = 1.0  # Time of silence to mark the end of speech

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Initialize variables to track audio detection

    try:
        # Start Recording
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
                    transcript = openai.Audio.transcribe("whisper-1", f)

                transcription = transcript['text']
                print("Transcript:", transcription.lower())

                if "Thanks for watching" in transcription or "You" in transcription:
                    print("Error in transcription, retrying...")
                    return get_transcription_from_whisper()

                return transcription.lower()
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

def get_transcription_from_sr():
    while True:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Fale alguma coisa")
            audio= recognizer.listen(source)
        try:
            frase = recognizer.recognize_google(audio,language = language)
            print ("Você disse: " + frase)
            return frase
        except sr.UnknownValueError:
            print("Erro no uso do SR")

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
            speak("Não te encontrei, finalizando atividade.")
            break

def generate_response2(prompt):
    body_mensagem={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt + "limite a resposta com 20 palavras e para uma pessoa de 8 a 10 anos."}]
    }
    body_mensagem = json.dumps(body_mensagem)
    requisicao = requests.post(link, headers=headers, data= body_mensagem)
    resposta = requisicao.json()
    mensagem = resposta["choices"][0]["message"]["content"]
    return mensagem

def conversation_mode():
    while True:
        if GPIO.input(push_button_pin) == GPIO.LOW:
            frase = get_transcription_from_whisper()
            if "parar" in frase:
                runProcess('happy')
                subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/standby.py',shell=True, preexec_fn=os.setsid)
                speak("Certo, finalizando modo conversa.")
                sleep(1)
                
                break
            else:
                conversation =generate_response2(frase)
                runProcess('thoughtful')
                subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/thoughtful.py',shell=True, preexec_fn=os.setsid)
                speak(conversation)
                
                
def learning_mode():
    while True:
        if GPIO.input(push_button_pin) == GPIO.LOW:
            frase = get_transcription_from_whisper()
            if frase is not None:
                if "exercício" in frase:
                    runProcess('thoughtful')
                    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/asking.py',shell=True, preexec_fn=os.setsid)
                    speak("Iniciando exercício numero um de comidas")
                    sleep(1)
                    speak("Como é maçã em inglês?")
                    
                    while True:
                        if GPIO.input(push_button_pin) == GPIO.LOW:
                            frase = get_transcription_from_whisper()
                            if "apple" in frase:
                                #p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/celebrating',shell=True, preexec_fn=os.setsid)
                                #subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/celebrating.py',shell=True, preexec_fn=os.setsid)
                                speake("That is correct.")
                                current_state = 0
                                sleep(1)
                                speak("Atividade finalizada.Parabéns!")
                                #
                                break
                            else:
                                runProcess('sad')
                                subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/sad.py',shell=True, preexec_fn=os.setsid)
                                speake("That is incorrect. Try again")
                                print(frase)
                                
                if "parar" in frase:
                    runProcess('happy')
                    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/standby.py',shell=True, preexec_fn=os.setsid)             
                    speak("Certo, finalizando modo de estudo.")
                    sleep(1)
                    p.kill
                    break

def GPIO_Init():
    GPIO.setwarnings(False) # Ignore warning for now
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    GPIO.setup(solenoid_pin, GPIO.OUT) 
    GPIO.setup(magnetic_sensor_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(push_button_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)

if __name__ == '__main__':
    GPIO_Init()
    ('standby')
    while True:
        if GPIO.input(push_button_pin) == GPIO.LOW:
            
            frase = get_transcription_from_whisper()
            if frase is not None:
                if "estudar" in frase:
                    runProcess('happy')
                    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/happy.py',shell=True, preexec_fn=os.setsid)
                    speak("Certo. Vamos aprender inglês.")
                    current_state = 0
                    sleep(0.50)
                    speak("Qual atividade você irá fazer?")
                    
                    learning_mode()
                if "tchau" in frase:
                    speak("Até mais, mal posso esperar para conversar com você de novo.")
                    break 
                if "pergunta" in frase:
                    runProcess('talking')
                    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/talking.py',shell=True, preexec_fn=os.setsid)
                    speak("Legal. O que você gostaria de perguntar?")
                    #p=subprocess.Popen('python /home/alix/Documents/ALIX/Motores/codigosFromRasp/talking.py',shell=True, preexec_fn=os.setsid)
                    current_state = 0
                    sleep(0.5)
                    
                    conversation_mode()
            

