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
#import keyboard
import pygame 
import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import subprocess

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml') # face detection
#absence_time = 300 # 5min
#short_pomodoro = 1500 #ciclo de 25 min
limit_input_time = 7
absence_time = 100
short_pomodoro = 10 #ciclo de 25 min#

solenoid_pin = 15
push_button_pin = 19 #gpio10
magnetic_sensor_pin = 32 #gpio12

music_path = "/home/alix/Documents/ALIX/ALIX/alix songs/"
#music_path = "C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/alix songs/"

language="pt-br"
#language_whisper_pt="pt"
#language_whisper_en="en"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def get_transcription_from_whisper(language_whisper):
    # Set the audio parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 4096
    SILENCE_THRESHOLD = 1000  # Silence threshold
    dev_index = 1  # Device index found by p.get_device_info_by_index(ii) --------trocar para 0----------
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
                return transcription.lower()

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
            speak("Encontrei você. Que bom, continue estudando.")
            break

        # Check if the face detection time has exceeded the limit
        if time.time() - face_time > limit_input_time:
            # Release the camera and close the OpenCV window
            cam.release()
            cv2.destroyAllWindows()
            speak("Não te encontrei.")
            break

def getLesson(i):
    f = open("Lessons", "r")
    content = f.readlines()
    return content[i][0:content[i].find(',')]

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
    speak("Qual tarefa você gostaria de fazer? Temos tarefas de leitura, de escuta e avaliação para fixar o conhecimento.")
    while True:
        if GPIO.input(push_button_pin) == GPIO.LOW:
                frase = get_transcription_from_whisper("pt")
                if frase is not None:
                    if "leitura" in frase:
                        reading_mode()
                    if "escuta" in frase:
                        listening_mode()
                    if "avaliação" in frase:
                        assessment_mode()
                    if "parar" in frase:
                        speak("Certo, finalizando modo de estudo.")
                        break

def reading_mode():
    break_count = 0
    p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/thoughtful',shell=True, preexec_fn=os.setsid)
    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/thoughtful.py',shell=True, preexec_fn=os.setsid)
    speak("Qual capítulo você irá ler?")
    p.kill()
    while True:
        if GPIO.input(push_button_pin) == GPIO.LOW:
            frase = get_transcription_from_whisper()
            if frase is not None:
                if "capítulo" in frase or "capitulo" in frase or "sentimentos" in frase:
                    p.kill()
                    p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/happy',shell=True, preexec_fn=os.setsid)
                    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/happy.py',shell=True, preexec_fn=os.setsid)
                    speak("Legal. Quando terminar a leitura, lembre de me avisar.")
                    a_time = time.time()
                    spomodoro_time = time.time()
                    total_time = time.time()
                    p.kill()
                    break
                if "parar" in frase:
                    p.kill()
                    p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/happy',shell=True, preexec_fn=os.setsid)
                    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/happy.py',shell=True, preexec_fn=os.setsid)
                    speak("Certo, finalizando modo de estudo da leitura.")
                    total_time = (time.time() - total_time) / 60  # Calculate total reading time in minutes
                    print("Tempo total = " + str(total_time))
                    break
    while True:
        current_time = time.time()
        if current_time- a_time > absence_time:
            print(time.time() - a_time)
            speak("Será que você ainda está ai? Vou te procurar.")
            subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Conversation/baseRotation.py',shell=True, preexec_fn=os.setsid)
            presence_detection()
            a_time = time.time() 
        
        if current_time - spomodoro_time > short_pomodoro:
            if(break_count < 4):
                print(time.time() - spomodoro_time)
                print(short_pomodoro)
                p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/talking',shell=True, preexec_fn=os.setsid)
                subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/talking.py',shell=True, preexec_fn=os.setsid)
                speak("Está na hora da sua pausa de 5 minutos.")
                sleep(5)
                speak("Pausa finalizada. Está na hora de voltar")
                spomodoro_time = time.time() 
                break_count += 1
                p.kill()
            else:
                p.kill()
                p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/happy',shell=True, preexec_fn=os.setsid)
                subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/happy.py',shell=True, preexec_fn=os.setsid)
                speak("Está na hora da sua pausa de 15 minutos.")
                sleep(10)
                speak("Pausa finalizada. Está na hora de voltar")
                spomodoro_time = time.time()
                p.kill()
                break_count = 0  # Reset the break count after a long break

#adjectives primeira pergunta tá errada
def assessment_mode():
    speak("Qual capítulo você irar fazer atividades?")
    while True:
        if GPIO.input(push_button_pin) == GPIO.LOW:
                frase = get_transcription_from_whisper("pt")
                if frase is not None:
                    if "sentimentos" in frase:
                        chapter = "Feelings"
                    #for j in range(10):
                    #    chapter = "Feelings"
                    #    if chapter in frase:
                        speak("Vamos fazer as atividades de sentimentos")
                    #        break
                    error_count = 0 
                    for i in range(6):
                        speak(getQuestion(chapter,i))
                        skip_question = False
                        while True:
                            if GPIO.input(push_button_pin) == GPIO.LOW:
                                frase = get_transcription_from_whisper("en")
                                if frase is not None:
                                    if getAnswer(chapter, i).lower() in frase:
                                        if(i<5):
                                            p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/happy',shell=True, preexec_fn=os.setsid)
                                            subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/happy.py',shell=True, preexec_fn=os.setsid)
                                            speak("Acertou, vamos para a próxima pergunta")
                                            p.kill()
                                            error_count = 0 
                                            break
                                        else:
                                            p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/happy',shell=True, preexec_fn=os.setsid)
                                            subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/happy.py',shell=True, preexec_fn=os.setsid)
                                            speak("Você finalizou a atividade. Parabéns")
                                            p.kill()
                                            lockable_compartment()
                                            break
                                    else:
                                        p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/sad',shell=True, preexec_fn=os.setsid)
                                        subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/sad.py',shell=True, preexec_fn=os.setsid)
                                        speak("Está errado tente outra vez")
                                        error_count += 1
                                        p.kill()
                                        if error_count >=3:
                                            speak("Parece que você está com dificuldades. Gostaria de pular essa questão?")
                                            while True:
                                                if GPIO.input(push_button_pin) == GPIO.LOW:
                                                    frase = get_transcription_from_whisper("pt")
                                                    if frase is not None:
                                                        if "sim" in frase:
                                                            speak("Tudo bem, vamos para a próxima pergunta")
                                                            error_count = 0
                                                            skip_question = True
                                                            break
                                                        if "não" in frase:
                                                            speak(getQuestion(chapter,i))
                                                            break
                            if skip_question:
                                skip_question = False
                                break

                    if "stop" in frase:
                        speak("Certo, finalizando modo de estudo.")
                        break

def listening_mode():
    p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/talking',shell=True, preexec_fn=os.setsid)
    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/talking.py',shell=True, preexec_fn=os.setsid)
    speak("Qual capítulo você quer praticar a escuta?") #ver com o vinicius
    p.kill()
    while True:
        if GPIO.input(push_button_pin) == GPIO.LOW:
                frase = get_transcription_from_whisper()
                if frase is not None:
                    if "adjetivos" in frase:
                        play_music("adjectives")
                    if "alfabeto" in frase:
                        play_music("alphabet")
                    if "animais" in frase:
                        play_music("animals")
                    if "cores" in frase:
                        play_music("colors")
                    if "sentimentos" in frase:
                        p.kill()
                        p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/celebrating',shell=True, preexec_fn=os.setsid)
                        speak("Muito bem. Escute com atenção e divirta-se.")
                        play_music("feelings")
                        p.kill()
                        p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/happy',shell=True, preexec_fn=os.setsid)
                        subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/happy.py',shell=True, preexec_fn=os.setsid)
                        speak("Espero que você tenha aprendido a pronunciar muitas palavras novas. Escute quantas vezes você quiser.")
                        p.kill()
                    if "comidas" in frase:
                        play_music("food")
                    if "cumprimentos" in frase:
                        play_music("greetings")
                    if "numeros" in frase:
                        play_music("numbers")
                    if "preposições" in frase:
                        play_music("prepositions")
                    if "formatos" in frase:
                        play_music("shapes")
                    if "parar" in frase:
                        speak("Certo, finalizando modo de estudo do listen.")
                        break

def play_music(music_name):
	pygame.mixer.music.load(music_path + music_name +".mp3")
	pygame.mixer.music.set_volume(1.0)
	pygame.mixer.music.play()

	while pygame.mixer.music.get_busy() == True:
		continue

def lockable_compartment():
    while True:
        if GPIO.input(push_button_pin) == GPIO.LOW:
            print("Button is pressed")
            GPIO.output(solenoid_pin, 1)
            while (GPIO.input(magnetic_sensor_pin) == GPIO.LOW):
                print(GPIO.input(magnetic_sensor_pin))
                GPIO.output(solenoid_pin, 1)
            
            sleep(2)
            GPIO.output(solenoid_pin, 0)
            break

    while (GPIO.input(magnetic_sensor_pin) == GPIO.HIGH):
        print("Trava aberta")

    speak("Compartimento de segurança fechado com sucesso")

def GPIO_Init():
    GPIO.setwarnings(False) # Ignore warning for now
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    GPIO.setup(solenoid_pin, GPIO.OUT) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
    GPIO.setup(magnetic_sensor_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
    GPIO.setup(push_button_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)


if __name__ == '__main__':
    pygame.init()
    pygame.mixer.init()
    #GPIO_Init()
    p = subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/standby',shell=True, preexec_fn=os.setsid)
    while True:
        if GPIO.input(push_button_pin) == GPIO.LOW:
            frase = get_transcription_from_whisper("pt")
            if frase is not None:
                if "estudar" in frase:
                    p.kill()
                    p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/talking',shell=True, preexec_fn=os.setsid)
                    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/talking.py',shell=True, preexec_fn=os.setsid)
                    '''speak("Certo. Precisamos realizar umas configurações antes de iniciar as atividades.")
                    sleep(0.50)
                    speak("Você vai utilizar o compartimento de recompensas?")
                    p.kill()
                    while True:
                        if GPIO.input(push_button_pin) == GPIO.LOW:
                            frase = get_transcription_from_whisper()
                            if frase is not None:
                                if "sim" in frase:
                                    p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/talking',shell=True, preexec_fn=os.setsid)
                                    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/talking.py',shell=True, preexec_fn=os.setsid)
                                    speak("Certo. Aperte o botão para destravar o compartimento e abra a porta")
                                    p.kill()
                                    while True:
                                        if GPIO.input(push_button_pin) == GPIO.LOW:
                                            print("Button is pressed")
                                            GPIO.output(solenoid_pin, 1)
                                            while (GPIO.input(magnetic_sensor_pin) == GPIO.LOW):
                                                print(GPIO.input(magnetic_sensor_pin))
                                                GPIO.output(solenoid_pin, 1)
                                            
                                            sleep(1)
                                            GPIO.output(solenoid_pin, 0)
                                            break

                                    while (GPIO.input(magnetic_sensor_pin) == GPIO.HIGH):
                                        print("Trava aberta")

                                    speak("Compartimento de recompensas ativado.")
                                    break
                                if "não" in frase:
                                    speak("Ok. Compartimento de recompensas desativado.")
                                    break

                    p.kill()
                    p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/talking',shell=True, preexec_fn=os.setsid)
                    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/talking.py',shell=True, preexec_fn=os.setsid)
                    speak("Você gostaria de usar a camera para detecção de presença durante as atividades?")
                    p.kill()
                    while True:
                        if GPIO.input(push_button_pin) == GPIO.LOW:
                            frase = get_transcription_from_whisper()
                            if frase is not None:
                                if "sim" in frase:
                                    #presence_detection()
                                    speak("Ok, detecção de presença ativado.")
                                    break
                                if "não" in frase:
                                    speak("Tudo bem, não irei utilizar a detecção de presença durante a atividade.")
                                    break
                                else:
                                    speak("Não é uma opção, diga sim ou não")
                    p.kill()
                    p=subprocess.Popen('exec /home/alix/Documents/ALIX/ALIX/DisplayLab/happy',shell=True, preexec_fn=os.setsid)
                    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Expressions/final_movements/happy.py',shell=True, preexec_fn=os.setsid)'''
                    speak("Vamos aprender inglês.")
                    p.kill()
                    learning_mode() 
                    #assessment_mode()

                if "tchau" in frase:
                    speak("Até mais, mal posso esperar para conversar com você de novo.")
                    break
                else:
                    speak("Não entendi o que você falou, fale outra vez.")
