import os
import numpy as np
import pyaudio
import openai
from pydub import AudioSegment
import time
import random
from time import sleep
import cv2
import speech_recognition as sr
import requests
import json
import subprocess
from datetime import datetime 
from google.cloud import texttospeech_v1
from pydub import AudioSegment
from pydub.playback import play
import pygame
import RPi.GPIO as GPIO 
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import threading
from threading import Event

from PIL import Image
from st7789v.interface import RaspberryPi
from st7789v import Display

address_default = "/home/alix/Documents/ALIX/ALIX/"

#Google cloud tts credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f"{address_default}Conversation/speech_gtts_cloud_key.json"

#face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')


#Rasp Pins
solenoid_pin = 15
push_button_pin = 31 #gpio6
magnetic_sensor_pin = 32 #gpio12

#Times
record_time = 10 # 10 seconds 
presence_time = 15 # 15 seconds
absence_time = 50 #seconds
short_pomodoro = 30 # 20seconds
break_count_limit = 4

#Global Variables
numero_maximo_imagens = 19
freq_mudanca_de_imagem = 0.1
max_bounce_time = 400
push_button_is_pressed = False
presence_detection_on = False
address_expression = f"{address_default}DisplayLab/images/"
event_expression = Event()

presence = False
presence_use = False
lock_use = False

# Display bounds
minY = 24
minX = 32
maxY = 216
maxX = 288

# Expressoes
happy = 'happy'
sad = 'sad'
celebrating = 'celebrating'
standby = 'standby'
talking = 'talking'
thoughtful = 'thoughtful'

movement_address = f"{address_default}Expressions/final_movements/"
expression = standby

#Musics
music_path = f"{address_default}alix songs/"
# music_path = "C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/alix songs/"

# Whisper and GPT-3.5 Turbo keys and credentials
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
link = "https://api.openai.com/v1/chat/completions"

#Database Connection
cred = credentials.Certificate(f"{address_default}Conversation/credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

#-------------------------------APIs--------------------------------------
# Whisper = speech to text
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
            if current_time - detection_time >= record_time:
                print("No speech detected within the last 10 seconds. Stopping recording.")
                run_expression(talking)
                ttsCloud("Não escutei o que você falou. Aperte o botão de novo para falar comigo.")
                break

    except Exception as e:
        print("Error during audio recording:", str(e))
    finally:
        # Always stop and close the stream and terminate audio
        stream.stop_stream()
        stream.close()
        audio.terminate()

#Google cloud tts
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

#Chat GPT 3.5-Turbo response
def generate_response(prompt):
    body_mensagem={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt + "limite a resposta com 20 palavras e para uma pessoa de 8 a 10 anos."}]
    }
    body_mensagem = json.dumps(body_mensagem)
    requisicao = requests.post(link, headers=headers, data= body_mensagem)
    resposta = requisicao.json()
    mensagem = resposta["choices"][0]["message"]["content"]
    return mensagem

#---------------------------Database Functions-----------------------------
def getQuestion(lesson, i):
    f = open(f"{address_default}Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    end = content[i].find(',')
    return content[i][0:end]
  
def getAnswer(lesson, i):
    f = open(f"{address_default}Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    begin = content[i].find(',') + 1
    end = content[i].find('/n')
    return content[i][begin:end]

def getLesson(i):
    f = open(f"{address_default}Conversation/Lessons", "r")
    content = f.readlines()
    return content[i][0:content[i].find(',')]

def getRangeLesson(lesson):
    f = open(f"{address_default}Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    n = 0
    for line in content:
        if lesson in line:
            i=n
        n = n+1
    return n

def getRangeCustom(custom):
    f = open(f"{address_default}Conversation/Questionnaires/{custom}", "r")
    content = f.readlines()
    n = 0
    for line in content:
        if custom in line:
            i=n
        n = n+1
    return n

def getCustomQuestion(custom, i):
    f = open(f"{address_default}Conversation/Questionnaires/{custom}", "r")
    content = f.readlines()
    end = content[i].find(',')
    return content[i][0:end]
    
def getCustomAnswer(custom, i):
    f = open(f"{address_default}Conversation/Questionnaires/{custom}", "r")
    content = f.readlines()
    begin = content[i].find(',') + 1
    end = content[i].find('/n')
    return content[i][begin:end]
    
def getCustoms(i):
    f = open(f"{address_default}Conversation/Customs", "r")
    content = f.readlines()
    return content[i][0:content[i].find('/n')]

def addAbsence(timeDate):
    absenceData = {"notified": False, "timeOfOccurence": timeDate}
    db.collection("Absences").add(absenceData)
    print(f"Added Absence")
    
def addResults(duration, grade, lesson):
    docs = (db.collection("Lesson").where("name", "==", lesson).stream())
    for doc in docs:
        db.collection("Lesson").document(doc.id).update({"duration":duration})
        db.collection("Lesson").document(doc.id).update({"grade":grade})
    print(f"Added Results")
 
#-----------------------Functions of learning mode---------------------------
def learning_mode():
    global push_button_is_pressed
    run_expression(thoughtful)
    ttsCloud("Qual capítulo você gostaria de aprender?")
    push_button_is_pressed = False
    while True:
        if push_button_is_pressed:
            push_button_is_pressed = False
            frase = get_transcription_from_whisper("pt")
            if frase is not None:
                if "capítulo" in frase:
                    start_time = time.time()
                    notfind = True
                    for j in range(10):
                        if getLesson(j).lower() in frase:
                            notfind = False
                            chapter = getLesson(j).lower()
                            run_expression(happy)
                            ttsCloud("Vamos fazer as atividades de " + chapter)
                            reading_mode(chapter)
                            listening_mode(chapter)
                            nota = assessment_mode(chapter)
                            print(nota)
                            run_expression(celebrating)
                            ttsCloud("Você terminou o capítulo. Muito bem")
                            final_time = time.time()
                            #Tempo gasto na atividade
                            total_time = (final_time - start_time)/60
                            addResults(int(total_time), nota, chapter)
                            print(addResults)
                            if lock_use == True:
                                run_expression(talking)
                                ttsCloud("Aperte o botão para abrir o compartimento de recompensas.")
                                lockable_compartment()
                            push_button_is_pressed = False
                            break
                    if notfind:
                        run_expression(thoughtful)
                        ttsCloud("Esse capítulo não existe. Olhe no material de estudo para ver os capítulos disponíveis.")
                elif "customizadas" in frase:
                    run_expression(happy)
                    ttsCloud("Vamos fazer as questões customizadas.")
                    customlearning()                 
                elif "parar" in frase:
                    run_expression(standby)
                    ttsCloud("Certo, finalizando modo de estudo.")
                    break
                else:
                    run_expression(thoughtful)
                    ttsCloud("Não entendi o que você disse. Olhe no material de estudo para ver os capítulos disponíveis.")
                    
def reading_mode(chapter):
    global push_button_is_pressed
    if presence_use == True:
        break_count = 0
        current_time = time.time()
        a_time = current_time
        spomodoro_time = current_time
        run_expression(talking)
        ttsCloud("Você já pode iniciar  a leitura do capítulo de " + chapter)
        ttsCloud("Ao terminar de ler o capítulo, lembre-se de me avisar.")
        push_button_is_pressed = False
        while True:
            current_time = time.time()
            if push_button_is_pressed:
                push_button_is_pressed = False
                a_time = time.time() 
                frase = get_transcription_from_whisper("pt")
                if frase is not None:
                    if "terminei" in frase or "acabei" in frase or "sim" in frase or "finalizei" in frase:
                        run_expression(celebrating)
                        ttsCloud("Certo, finalizando modo de estudo de leitura.")
                        break
                    elif "ainda" in frase or "lendo" in frase:
                        run_expression(standby)
                        ttsCloud("Tudo bem. Quando terminar de ler, lembre de me avisar.")
                    else:
                        run_expression(thoughtful)
                        ttsCloud("Não entendi o que você disse. Você já terminou a leitura?")
                    
            if current_time - a_time > absence_time:
                print(time.time() - a_time)
                run_expression(thoughtful)
                ttsCloud("Será que você ainda está aí? Vou te procurar.")
                presence = presence_detection()
                if presence == True:
                    a_time = time.time() 
                elif presence == False:
                    break
            
            if current_time - spomodoro_time > short_pomodoro:
                if(break_count < break_count_limit):
                    print(time.time() - spomodoro_time)
                    print(short_pomodoro)
                    run_expression(standby)
                    ttsCloud("Está na hora da sua pausa de 5 minutos.")
                    sleep(5)
                    run_expression(talking)
                    ttsCloud("Pausa finalizada. Está na hora de voltar")
                    spomodoro_time = time.time() 
                    break_count += 1
                else:
                    run_expression(standby)
                    ttsCloud("Está na hora da sua pausa de 15 minutos.")
                    sleep(10)
                    run_expression(talking)
                    ttsCloud("Pausa finalizada. Está na hora de voltar")
                    spomodoro_time = time.time()
                    break_count = 0  # Reset the break count after a long break
    
    elif presence_use == False:
        break_count = 0
        current_time = time.time()
        a_time = current_time 
        spomodoro_time = current_time
        run_expression(talking)
        ttsCloud("Você já pode iniciar  a leitura do caítulo" + chapter)
        ttsCloud("Lembre que ao finalizar a leitura do capítulo, me avise apertando o botão.")
        while True:
            current_time = time.time()
            if push_button_is_pressed:
                push_button_is_pressed = False
                frase = get_transcription_from_whisper("pt")
                if frase is not None:
                    if "terminei" in frase or "acabei" in frase or "sim" in frase or "finalizei" in frase:
                        run_expression(celebrating)
                        ttsCloud("Certo, finalizando modo de estudo da leitura.")
                        break
                    elif "ainda" in frase or "lendo" in frase:
                        run_expression(standby)
                        ttsCloud("Tudo bem. Quando terminar de ler, lembre de me avisar.")
                    else:
                        run_expression(thoughtful)
                        ttsCloud("Não entendi o que você disse. Você já terminou a leitura?")
            
            if current_time - spomodoro_time > short_pomodoro:
                if(break_count < break_count_limit):
                    print(time.time() - spomodoro_time)
                    print(short_pomodoro)
                    run_expression(standby)
                    ttsCloud("Está na hora da sua pausa de 5 minutos.")
                    sleep(5)
                    run_expression(talking)
                    ttsCloud("Pausa finalizada. Está na hora de voltar")
                    spomodoro_time = time.time() 
                    break_count += 1
                else:
                    run_expression(standby)
                    ttsCloud("Está na hora da sua pausa de 15 minutos.")
                    sleep(10)
                    run_expression(talking)
                    ttsCloud("Pausa finalizada. Está na hora de voltar")
                    spomodoro_time = time.time()
                    break_count = 0  # Reset the break count after a long break

def listening_mode(chapter):
    global push_button_is_pressed
    run_expression(talking)
    ttsCloud("Vamos praticar a atividade de escuta do capítulo de ?" + chapter)
    current_time = time.time()
    a_time = current_time
    push_button_is_pressed = False
    if presence_use == True:
        while True:
            current_time = time.time()
            if push_button_is_pressed:
                push_button_is_pressed = False
                a_time = time.time() 
                frase = get_transcription_from_whisper("pt")
                if frase is not None:
                    if "sim" in frase:
                        run_expression(celebrating)
                        ttsCloud("Muito bem. Escute com atenção e divirta-se.")
                        play_music(chapter)
                        run_expression(celebrating)
                        ttsCloud("Espero que você tenha aprendido a pronunciar muitas palavras novas. Escute quantas vezes você quiser.")
                        break
                    if "não" in frase:
                        run_expression(talking)
                        ttsCloud("Tudo bem, vamos para a atividade de avaliação.")
                        break
                    else:
                        run_expression(thoughtful)
                        ttsCloud("Não entendi o que você disse. Me responda Sim ou Não para fazer atividade de escuta.")
                        break
            
            if current_time - a_time > absence_time:
                print(time.time() - a_time)
                run_expression(thoughtful)
                ttsCloud("Será que você ainda está aí? Vou te procurar.")
                presence = presence_detection()
                if presence == True:
                    a_time = time.time()
                elif presence == False:
                    break
    
    elif presence_use == False:
        while True:
            if push_button_is_pressed:
                push_button_is_pressed = False
                frase = get_transcription_from_whisper("pt")
                if frase is not None:
                    if "sim" in frase:
                        run_expression(celebrating)
                        ttsCloud("Muito bem. Escute com atenção e divirta-se.")
                        play_music(chapter)
                        run_expression(celebrating)
                        ttsCloud("Espero que você tenha aprendido a pronunciar muitas palavras novas. Escute quantas vezes você quiser.")
                        break
                    if "não" in frase:
                        run_expression(talking)
                        ttsCloud("Tudo bem, vamos para a atividade de avaliação.")
                        break
                    else:
                        run_expression(thoughtful)
                        ttsCloud("Não entendi o que você disse. Me responda Sim ou Não para fazer atividade de escuta.")
                        break

#adjectives primeira pergunta tá errada
def assessment_mode(chapter):
    global push_button_is_pressed
    run_expression(talking)
    ttsCloud("Vamos praticar a avaliação do capítulo de " + chapter + "?")
    outer_break = False
    break_count = 0
    current_time = time.time()
    a_time = current_time 
    spomodoro_time = current_time
    push_button_is_pressed = False
    while True:
        if push_button_is_pressed:
            push_button_is_pressed = False
            a_time = time.time() 
            frase = get_transcription_from_whisper("pt")
            if frase is not None:
                frase_lower = frase.lower()
                if "sim" in frase_lower:
                    if presence_use == True:
                        #run_expression(thoughtful)
                        ttsCloud("Vamos começar.")
                        error_count = 0
                        nota = 0 
                        for i in range(getRangeLesson(chapter)):
                            run_expression(thoughtful)
                            ttsCloud(getQuestion(chapter,i))
                            skip_question = False
                            push_button_is_pressed = False
                            while True:
                                current_time = time.time()
                                if push_button_is_pressed:
                                    push_button_is_pressed = False
                                    a_time = time.time() 
                                    frase = get_transcription_from_whisper("en")
                                    if frase is not None:
                                        if getAnswer(chapter, i) in frase:
                                            if(i < ((getRangeLesson(chapter))-1)):
                                                run_expression(happy)
                                                ttsCloud("Acertou, vamos para a próxima pergunta")
                                                error_count = 0
                                                nota += 1
                                                break
                                            else:
                                                run_expression(celebrating)
                                                ttsCloud("Você finalizou a atividade. Parabéns")
                                                nota += 1
                                                #nota
                                                media = (nota/getRangeLesson(chapter)) * 10
                                                outer_break = True
                                                return media
                                        else:
                                            run_expression(sad)
                                            ttsCloud("Está errado tente outra vez")
                                            push_button_is_pressed = False
                                            error_count += 1
                                            if error_count >=3:
                                                run_expression(talking)
                                                ttsCloud("Parece que você está com dificuldades. Gostaria de pular essa questão?")
                                                push_button_is_pressed = False
                                                while True:
                                                    if push_button_is_pressed:
                                                        push_button_is_pressed = False
                                                        a_time = time.time() 
                                                        frase = get_transcription_from_whisper("pt")
                                                        if frase is not None:
                                                            if "sim" in frase:
                                                                run_expression(talking)
                                                                ttsCloud("Tudo bem, vamos para a próxima pergunta")
                                                                error_count = 0
                                                                skip_question = True
                                                                break
                                                            if "não" in frase:
                                                                run_expression(thoughtful)
                                                                ttsCloud(getQuestion(chapter,i))
                                                                break
                                                            else:
                                                                run_expression(thoughtful)
                                                                ttsCloud("Não entendi. Me responda se você quer pular a questão com Sim ou Não.")
                                        if skip_question:
                                            skip_question = False
                                            break
                                
                                if current_time - a_time > absence_time:
                                    print(time.time() - a_time)
                                    run_expression(thoughtful)
                                    ttsCloud("Será que você ainda está aí? Vou te procurar.")
                                    presence = presence_detection()
                                    if presence == True:
                                        a_time = time.time() 
                                    elif presence == False:
                                        break 
                                if current_time - spomodoro_time > short_pomodoro:
                                    if(break_count < break_count_limit):
                                        print(time.time() - spomodoro_time)
                                        print(short_pomodoro)
                                        run_expression(standby)
                                        ttsCloud("Está na hora da sua pausa de 5 minutos.")
                                        sleep(5)
                                        run_expression(talking)
                                        ttsCloud("Pausa finalizada. Está na hora de voltar")
                                        spomodoro_time = time.time() 
                                        break_count += 1
                                    else:
                                        run_expression(standby)
                                        ttsCloud("Está na hora da sua pausa de 15 minutos.")
                                        sleep(10)
                                        run_expression(talking)
                                        ttsCloud("Pausa finalizada. Está na hora de voltar")
                                        spomodoro_time = time.time()
                                        break_count = 0  # Reset the break count after a long break

                                if outer_break:
                                    break 
                    
                    elif presence_use == False:
                        #run_expression(thoughtful)
                        ttsCloud("Vamos começar.")
                        error_count = 0
                        nota = 0 
                        for i in range(getRangeLesson(chapter)):
                            run_expression(thoughtful)
                            ttsCloud(getQuestion(chapter,i))
                            skip_question = False
                            push_button_is_pressed = False
                            while True:
                                current_time = time.time()
                                if push_button_is_pressed:
                                    push_button_is_pressed = False
                                    frase = get_transcription_from_whisper("en")
                                    if frase is not None:
                                        if getAnswer(chapter, i).lower() in frase:
                                            if(i < ((getRangeLesson(chapter))-1)):
                                                run_expression(happy)
                                                ttsCloud("Acertou, vamos para a próxima pergunta")
                                                error_count = 0
                                                nota += 1
                                                break
                                            else:
                                                run_expression(celebrating)
                                                ttsCloud("Você finalizou a atividade. Parabéns")
                                                nota += 1
                                                #nota
                                                media = (nota/getRangeLesson(chapter)) * 10
                                                outer_break = True
                                                return media
                                        else:
                                            run_expression(sad)
                                            ttsCloud("Está errado tente outra vez")
                                            error_count += 1
                                            if error_count >=3:
                                                run_expression(talking)
                                                ttsCloud("Parece que você está com dificuldades. Gostaria de pular essa questão?")
                                                push_button_is_pressed = False
                                                while True:
                                                    if push_button_is_pressed:
                                                        push_button_is_pressed = False
                                                        frase = get_transcription_from_whisper("pt")
                                                        if frase is not None:
                                                            if "sim" in frase:
                                                                run_expression(talking)
                                                                ttsCloud("Tudo bem, vamos para a próxima pergunta")
                                                                error_count = 0
                                                                skip_question = True
                                                                break
                                                            if "não" in frase:
                                                                run_expression(thoughtful)
                                                                ttsCloud(getQuestion(chapter,i))
                                                                break
                                                            else:
                                                                run_expression(thoughtful)
                                                                ttsCloud("Não entendi. Me responda se você quer pular a questão com Sim ou Não.")
                                        if skip_question:
                                            skip_question = False
                                            break
                                
                                if current_time - spomodoro_time > short_pomodoro:
                                    if(break_count < break_count_limit):
                                        print(time.time() - spomodoro_time)
                                        print(short_pomodoro)
                                        run_expression(standby)
                                        ttsCloud("Está na hora da sua pausa de 5 minutos.")
                                        sleep(5)
                                        run_expression(talking)
                                        ttsCloud("Pausa finalizada. Está na hora de voltar")
                                        spomodoro_time = time.time() 
                                        break_count += 1
                                    else:
                                        run_expression(standby)
                                        ttsCloud("Está na hora da sua pausa de 15 minutos.")
                                        sleep(10)
                                        run_expression(talking)
                                        ttsCloud("Pausa finalizada. Está na hora de voltar")
                                        spomodoro_time = time.time()
                                        break_count = 0  # Reset the break count after a long break

                                if outer_break:
                                    break 

                elif "não" in frase:
                    run_expression(talking)
                    ttsCloud("Certo, finalizando modo de estudo.")
                    break
                else:
                    run_expression(thoughtful)
                    ttsCloud("Não entendi o que você disse. Me responda Sim ou Não para fazer atividade de avaliação.")
        if outer_break:
            break  # This break will exit the outer while loop

#Play song for listening mode
def play_music(music_name):
	pygame.mixer.music.load(music_path + music_name +".mp3")
	pygame.mixer.music.set_volume(1.0)
	pygame.mixer.music.play()

	while pygame.mixer.music.get_busy() == True:
		continue

#-----------------------Functions of conversation mode----------------------
def conversation_mode():
    continue_conversation = True
    while continue_conversation:
        if GPIO.input(push_button_pin) == GPIO.LOW:
            frase = get_transcription_from_whisper("pt")
            if "parar" in frase:
                run_expression(happy, standby)
                ttsCloud("Certo, finalizando modo conversa.")
                sleep(1)
                continue_conversation = False
            else:
                run_expression(talking)
                conversation =generate_response(frase)
                ttsCloud(conversation)

#----------------------------Custom Questions------------------------------
def customlearning():
    global push_button_is_pressed
    chapter = "curiosidades" 
    qtde = getRangeCustom(chapter)
    if qtde <= 5:
        for i in range (qtde):
            run_expression(thoughtful)
            ttsCloud(getQuestion(chapter,i))
            skip_question = False
            push_button_is_pressed = False
            while True:
                if push_button_is_pressed:
                    push_button_is_pressed = False
                    frase = get_transcription_from_whisper("pt")
                    if frase is not None:
                        if getAnswer(chapter, i) in frase:
                            if(i < ((getRangeCustom(chapter))-1)):
                                run_expression(happy)
                                ttsCloud("Acertou, vamos para a próxima pergunta")
                                error_count = 0
                                nota += 1
                                break
                            else:
                                run_expression(celebrating)
                                ttsCloud("Você finalizou a atividade. Parabéns")
                                break
                        else:
                            run_expression(sad)
                            ttsCloud("Está errado tente outra vez")
                            push_button_is_pressed = False
                            error_count += 1
                            if error_count >=3:
                                run_expression(thoughtful)
                                ttsCloud("Parece que você está com dificuldades. Gostaria de pular essa questão?")
                                push_button_is_pressed = False
                                while True:
                                    if push_button_is_pressed:
                                        push_button_is_pressed = False
                                        frase = get_transcription_from_whisper("pt")
                                        if frase is not None:
                                            if "sim" in frase:
                                                run_expression(talking)
                                                ttsCloud("Tudo bem, vamos para a próxima pergunta")
                                                error_count = 0
                                                skip_question = True
                                                break
                                            if "não" in frase:
                                                run_expression(thoughtful)
                                                ttsCloud(getQuestion(chapter,i))
                                                break
                                            else:
                                                run_expression(thoughtful)
                                                ttsCloud("Não entendi. Me responda se você quer pular a questão com Sim ou Não.")
                        if skip_question:
                            skip_question = False
                            break
                
    else:
        List = [i for i in range(0, qtde)]
        UpdatedList = random.sample(List, k = 5)
        for j in UpdatedList:
            run_expression(thoughtful)
            ttsCloud(getQuestion(chapter,j))
            skip_question = False
            push_button_is_pressed = False
            while True:
                if push_button_is_pressed:
                    push_button_is_pressed = False
                    frase = get_transcription_from_whisper("pt")
                    if frase is not None:
                        if getAnswer(chapter, j) in frase:
                            if(j < ((getRangeCustom(chapter))-1)):
                                run_expression(happy)
                                ttsCloud("Acertou, vamos para a próxima pergunta")
                                error_count = 0
                                nota += 1
                                break
                            else:
                                run_expression(celebrating)
                                ttsCloud("Você finalizou a atividade. Parabéns")
                                break
                        else:
                            run_expression(sad)
                            ttsCloud("Está errado tente outra vez")
                            push_button_is_pressed = False
                            error_count += 1
                            if error_count >=3:
                                run_expression(thoughtful)
                                ttsCloud("Parece que você está com dificuldades. Gostaria de pular essa questão?")
                                push_button_is_pressed = False
                                while True:
                                    if push_button_is_pressed:
                                        push_button_is_pressed = False
                                        frase = get_transcription_from_whisper("pt")
                                        if frase is not None:
                                            if "sim" in frase:
                                                run_expression(talking)
                                                ttsCloud("Tudo bem, vamos para a próxima pergunta")
                                                error_count = 0
                                                skip_question = True
                                                break
                                            if "não" in frase:
                                                run_expression(thoughtful)
                                                ttsCloud(getQuestion(chapter,i))
                                                break
                                            else:
                                                run_expression(thoughtful)
                                                ttsCloud("Não entendi. Me responda se você quer pular a questão com Sim ou Não.")
                        if skip_question:
                            skip_question = False
                            break
                
#--------------------------Other functions-----------------------------------
def run_expression(expressionName, movementName = None):
    global expression

    if not movementName:
        movementName = expressionName
    
    expression = expressionName
    subprocess.Popen('python ' + movement_address + movementName + '.py',shell=True, preexec_fn=os.setsid)
    
def GPIO_Init():
    pygame.init()
    pygame.mixer.init()
    GPIO.setwarnings(False) # Ignore warning for now
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    GPIO.setup(solenoid_pin, GPIO.OUT) 
    GPIO.setup(magnetic_sensor_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(push_button_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.add_event_detect(push_button_pin, GPIO.FALLING, 
        callback=push_button_handler, bouncetime=max_bounce_time)

def presence_detection():
    global presence_detection_on
    global presence
    presence_detection_on = True
    
    subprocess.Popen(f"python {address_default}Conversation/baseRotation.py",shell=True, preexec_fn=os.setsid)
    consecutive_face_count = 0  # Track consecutive face detections
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera")
        presence_detection_on = False
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
            consecutive_face_count += 1
        else:
            consecutive_face_count = 0  # Reset count if no face detected

        if consecutive_face_count >= 5:
            cam.release()
            cv2.destroyAllWindows()
            run_expression(happy)
            ttsCloud("Você ainda está aí. Você pode me responder apertando o botão.")
            presence = True
            break

        # Check if the face detection time has exceeded the limit
        if time.time() - face_time > presence_time:
            # Release the camera and close the OpenCV window
            cam.release()
            cv2.destroyAllWindows()
            run_expression(sad)
            ttsCloud("Não te encontrei, finalizando atividade.")
            #data e hora de ausência
            timestamp = time.time()
            date_time = datetime.fromtimestamp(timestamp)
            addAbsence(date_time)
            presence = False
            break

    #start_thread_expression()
    presence_detection_on = False
    return presence

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

    run_expression(standby)
    ttsCloud("Compartimento de recompensas fechado com sucesso.")

def push_button_handler(sig):
    global push_button_is_pressed
    #GPIO.cleanup()
    push_button_is_pressed = True

#----------------------------Thread functions----------------------------
def start_thread_expression():
    global event_expression
    event_expression.clear()
    thread = threading.Thread(target=thread_expression)
    thread.daemon = True
    thread.start()

def stop_thread_expression():
    global event_expression
    event_expression.set()

def thread_expression():
    internal_expression = expression
    with RaspberryPi() as ipr:#rpi:
        display = Display(ipr)
        display.initialize(color_mode=444)
        frame = Image.open(address_expression+'preto'+'/preto'+'.png')
        data = list(frame.convert('RGB').getdata())
        display.draw_rgb_bytes(data)
        display.initialize(color_mode=444, bounds=(minY,minX,maxY,maxX))
        while True:
            for i in range(numero_maximo_imagens):
                if presence_detection_on:
                    continue
                # Se a expressao mudar, reinicia o loop de imagens da pasta
                if internal_expression != expression:
                    i = 0
                    internal_expression = expression
                    
                frame = Image.open(address_expression+expression+'/frame'+str(i)+'.png')
                data = list(frame.convert('RGB').getdata())
                display.draw_rgb_bytes(data)
                time.sleep(freq_mudanca_de_imagem)

#----------------------------Main function----------------------------
if __name__ == '__main__':
    GPIO_Init()
    t = threading.Thread(target=thread_expression)
    t.daemon = True
    t.start()
    ttsCloud("Olá, eu sou ALIX. O que vamos aprender hoje?")
    while True:
        if GPIO.input(push_button_pin) == GPIO.LOW:
            frase = get_transcription_from_whisper("pt")
            if frase is not None:
                if "estudar" in frase:
                    run_expression(talking)
                    ttsCloud("Certo. Precisamos realizar umas configurações antes de iniciar as atividades.")
                    ttsCloud("Você vai utilizar o compartimento de recompensas?")
                    push_button_is_pressed = False
                    while True:
                        if GPIO.input(push_button_pin) == GPIO.LOW:
                            frase = get_transcription_from_whisper("pt")
                            if frase is not None:
                                if "sim" in frase:
                                    run_expression(happy)
                                    ttsCloud("Certo. Aperte o botão para destravar o compartimento e abra a porta")
                                    lockable_compartment()
                                    lock_use = True
                                    run_expression(standby)
                                    ttsCloud("Compartimento de recompensas ativado.")
                                    break
                                if "não" in frase:
                                    run_expression(standby)
                                    ttsCloud("Ok. Compartimento de recompensas desativado.")
                                    lock_use = False
                                    break
                                else:
                                    run_expression(thoughtful)
                                    ttsCloud("Não entendi. Me responda se você quer usar o compartimento de recompensas com Sim ou Não.")
                                    
                    run_expression(talking)
                    ttsCloud("Você gostaria de usar a camera para detecção de presença durante as atividades?")
                    while True:
                        if GPIO.input(push_button_pin) == GPIO.LOW:
                            frase = get_transcription_from_whisper("pt")
                            if frase is not None:
                                if "sim" in frase:
                                    presence_use = True
                                    run_expression(happy)
                                    ttsCloud("Ok, detecção de presença ativado.")
                                    break
                                if "não" in frase:
                                    run_expression(standby)
                                    ttsCloud("Ok. Detecção de presença desativado")
                                    presence_use = False
                                    break
                                else:
                                    run_expression(thoughtful)
                                    ttsCloud("Não entendi. Me responda se você quer usar a detecção de presença com Sim ou Não.")
                    run_expression(happy)
                    ttsCloud("Vamos aprender inglês!!!")
                    #presence_use = True
                    #lock_use = False
                    learning_mode()
                elif "pergunta" in frase:
                    run_expression(thoughtful)
                    ttsCloud("Legal. O que você gostaria de perguntar?")
                    conversation_mode()
                elif "parar" in frase:
                    run_expression(standby)
                    ttsCloud("Tchau")
                    break
                elif "desligar" in frase:
                    run_expression(standby)
                    ttsCloud("Até mais, mal posso esperar para conversar com você de novo.")
                    os.system("sudo shutdown -h now")  
                else:
                    run_expression(thoughtful)
                    ttsCloud("Não entendi o que você falou.")


    
                