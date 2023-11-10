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
import pygame 
#import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import subprocess
from google.cloud import texttospeech_v1
from pydub import AudioSegment
from pydub.playback import play

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'speech_gtts_cloud_key.json'

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

language="pt"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}



def ttsCloud(message):
    # Instantiates a client
    client = texttospeech_v1.TextToSpeechClient()
    # Set the text input to be synthesized
    synthesis_input = texttospeech_v1.SynthesisInput(text=message)

    voice = texttospeech_v1.VoiceSelectionParams(
        name = 'pt-BR-Standard-C',
        language_code = "pt-BR"
    )
    audio_config = texttospeech_v1.AudioConfig(
        audio_encoding=texttospeech_v1.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, 
        voice=voice, 
        audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)

    song = AudioSegment.from_mp3("output.mp3")
    play(song)

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
                ttsCloud("Não escutei o que você falou. Aperte o botão de novo para falar comigo.")
                break

    except Exception as e:
        print("Error during audio recording:", str(e))
    finally:
        # Always stop and close the stream and terminate audio
        stream.stop_stream()
        stream.close()
        audio.terminate()

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
            ttsCloud("Encontrei você. Que bom, continue estudando.")
            break

        # Check if the face detection time has exceeded the limit
        if time.time() - face_time > limit_input_time:
            # Release the camera and close the OpenCV window
            cam.release()
            cv2.destroyAllWindows()
            ttsCloud("Não te encontrei.")
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
    ttsCloud("Qual tarefa você gostaria de fazer? Temos tarefas de leitura, de escuta e avaliação para fixar o conhecimento.")
    while True:
        if keyboard.read_key() == "r":
                frase = get_transcription_from_whisper("pt")
                if frase is not None:
                    if "leitura" in frase:
                        reading_mode()
                    if "escuta" in frase:
                        listening_mode()
                    if "avaliação" in frase:
                        assessment_mode()
                    if "parar" in frase:
                        ttsCloud("Certo, finalizando modo de estudo.")
                        break

def reading_mode():
    break_count = 0
    ttsCloud("Qual capítulo você irá ler?")
    while True:
        if keyboard.read_key() == "r":
            frase = get_transcription_from_whisper()
            if frase is not None:
                if "capítulo" in frase or "capitulo" in frase or "sentimentos" in frase:
                    ttsCloud("Legal. Quando terminar a leitura, lembre de me avisar.")
                    a_time = time.time()
                    spomodoro_time = time.time()
                    total_time = time.time()
                    break
                if "parar" in frase:
                    ttsCloud("Certo, finalizando modo de estudo da leitura.")
                    total_time = (time.time() - total_time) / 60  # Calculate total reading time in minutes
                    print("Tempo total = " + str(total_time))
                    break
    while True:
        current_time = time.time()
        if current_time- a_time > absence_time:
            print(time.time() - a_time)
            ttsCloud("Será que você ainda está ai? Vou te procurar.")
            presence_detection()
            a_time = time.time() 
        
        if current_time - spomodoro_time > short_pomodoro:
            if(break_count < 4):
                print(time.time() - spomodoro_time)
                print(short_pomodoro)
                ttsCloud("Está na hora da sua pausa de 5 minutos.")
                sleep(5)
                ttsCloud("Pausa finalizada. Está na hora de voltar")
                spomodoro_time = time.time() 
                break_count += 1
            else:
                ttsCloud("Está na hora da sua pausa de 15 minutos.")
                sleep(10)
                ttsCloud("Pausa finalizada. Está na hora de voltar")
                spomodoro_time = time.time()
                break_count = 0  # Reset the break count after a long break

#adjectives primeira pergunta tá errada
def assessment_mode():
    ttsCloud("Qual capítulo você irar fazer atividades?")
    while True:
        if keyboard.read_key() == "r":
                frase = get_transcription_from_whisper("pt")
                if frase is not None:
                    for j in range(10):
                        if getLesson(j) in frase:
                            chapter = getLesson(j)
                            ttsCloud("Vamos fazer as atividades de " + chapter)
                            break
                    error_count = 0 
                    for i in range(6):
                        ttsCloud(getQuestion(chapter,i))
                        skip_question = False
                        while True:
                            if keyboard.read_key() == "r":
                                frase = get_transcription_from_whisper("en")
                                if frase is not None:
                                    if getAnswer(chapter, i).lower() in frase:
                                        if(i<5):
                                            ttsCloud("Acertou, vamos para a próxima pergunta")
                                            error_count = 0 
                                            break
                                        else:
                                            ttsCloud("Você finalizou a atividade. Parabéns")
                                            #lockable_compartment()
                                            break
                                    else:
                                        ttsCloud("Está errado tente outra vez")
                                        error_count += 1
                                        if error_count >=3:
                                            ttsCloud("Parece que você está com dificuldades. Gostaria de pular essa questão?")
                                            while True:
                                                if keyboard.read_key() == "r":
                                                    frase = get_transcription_from_whisper("pt")
                                                    if frase is not None:
                                                        if "sim" in frase:
                                                            ttsCloud("Tudo bem, vamos para a próxima pergunta")
                                                            error_count = 0
                                                            skip_question = True
                                                            break
                                                        if "não" in frase:
                                                            ttsCloud(getQuestion(chapter,i))
                                                            break
                            if skip_question:
                                skip_question = False
                                break

                    if "stop" in frase:
                        ttsCloud("Certo, finalizando modo de estudo.")
                        break

def listening_mode():
    ttsCloud("Qual capítulo você quer praticar a escuta?") #ver com o vinicius
    while True:
        if keyboard.read_key() == "r":
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
                        ttsCloud("Muito bem. Escute com atenção e divirta-se.")
                        play_music("feelings")
                        ttsCloud("Espero que você tenha aprendido a pronunciar muitas palavras novas. Escute quantas vezes você quiser.")
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
                        ttsCloud("Certo, finalizando modo de estudo do listen.")
                        break

def play_music(music_name):
	pygame.mixer.music.load(music_path + music_name +".mp3")
	pygame.mixer.music.set_volume(1.0)
	pygame.mixer.music.play()

	while pygame.mixer.music.get_busy() == True:
		continue

'''def lockable_compartment():
    while True:
        if keyboard.read_key() == "r":
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

    ttsCloud("Compartimento de segurança fechado com sucesso")

def GPIO_Init():
    GPIO.setwarnings(False) # Ignore warning for now
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    GPIO.setup(solenoid_pin, GPIO.OUT) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
    GPIO.setup(magnetic_sensor_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
    GPIO.setup(push_button_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)'''


if __name__ == '__main__':
    pygame.init()
    pygame.mixer.init()
    #GPIO_Init()
    while True:
        if keyboard.read_key() == "r":
            frase = get_transcription_from_whisper("pt")
            if frase is not None:
                if "estudar" in frase:
                    '''ttsCloud("Certo. Precisamos realizar umas configurações antes de iniciar as atividades.")
                    sleep(0.50)
                    ttsCloud("Você vai utilizar o compartimento de recompensas?")
                    while True:
                        if keyboard.read_key() == "r":
                            frase = get_transcription_from_whisper()
                            if frase is not None:
                                if "sim" in frase:
                                    ttsCloud("Certo. Aperte o botão para destravar o compartimento e abra a porta")
                                    while True:
                                        if keyboard.read_key() == "r":
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

                                    ttsCloud("Compartimento de recompensas ativado.")
                                    break
                                if "não" in frase:
                                    ttsCloud("Ok. Compartimento de recompensas desativado.")
                                    break

                    ttsCloud("Você gostaria de usar a camera para detecção de presença durante as atividades?")
                    while True:
                        if keyboard.read_key() == "r":
                            frase = get_transcription_from_whisper()
                            if frase is not None:
                                if "sim" in frase:
                                    #presence_detection()
                                    ttsCloud("Ok, detecção de presença ativado.")
                                    break
                                if "não" in frase:
                                    ttsCloud("Tudo bem, não irei utilizar a detecção de presença durante a atividade.")
                                    break
                                else:
                                    ttsCloud("Não é uma opção, diga sim ou não")'''
                    ttsCloud("Vamos aprender inglês.")
                    learning_mode() 
                    #assessment_mode()

                if "tchau" in frase:
                    ttsCloud("Até mais, mal posso esperar para conversar com você de novo.")
                    break
                else:
                    ttsCloud("Não entendi o que você falou, fale outra vez.")
