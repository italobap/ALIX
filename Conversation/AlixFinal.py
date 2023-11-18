import os
import numpy as np
import pyaudio
import openai
from pydub import AudioSegment
from gtts import gTTS
import time
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

#Google cloud tts credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/alix/Documents/ALIX/ALIX/Conversation/speech_gtts_cloud_key.json'

#face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')


#Rasp Pins
solenoid_pin = 15
push_button_pin = 19 #gpio10
magnetic_sensor_pin = 32 #gpio12

#Times
record_time = 10 # 10 seconds 
presence_time = 15 # 15 seconds
absence_time = 100 #seconds
short_pomodoro = 30 # 20seconds

#Global Variables
presence = False
push_button_is_pressed = False
expressionAddress = '/home/alix/Documents/ALIX/ALIX/DisplayLab/'
movementAddress = '/home/alix/Documents/ALIX/ALIX/Expressions/final_movements/'
processRun = True
p = subprocess.Popen('exec ' + expressionAddress + 'standby', shell=True, preexec_fn=os.setsid)

#Musics
#music_path = "/home/alix/Documents/ALIX/ALIX/alix songs/"
music_path = "C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/alix songs/"

# Whisper and GPT-3.5 Turbo keys and credentials
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
link = "https://api.openai.com/v1/chat/completions"

#Database Connection
cred = credentials.Certificate("/home/alix/Documents/ALIX/ALIX/Conversation/credentials.json")
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
                runProcess('thoughtful')
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
    f = open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    end = content[i].find(',')
    return content[i][0:end]
  
def getAnswer(lesson, i):
    f = open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    begin = content[i].find(',') + 1
    end = content[i].find('/n')
    return content[i][begin:end]

def getLesson(i):
    f = open("/home/alix/Documents/ALIX/ALIX/Conversation/Lessons", "r")
    content = f.readlines()
    return content[i][0:content[i].find(',')]

def getRange(lesson):
    f = open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    n = 0
    for line in content:
        if lesson in line:
            i=n
        n = n+1
    return n

def getCustomQuestion(custom, i):
    f = open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{custom}", "r")
    content = f.readlines()
    end = content[i].find(',')
    return content[i][0:end]
    
def getCustomAnswer(custom, i):
    f = open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{custom}", "r")
    content = f.readlines()
    begin = content[i].find(',') + 1
    end = content[i].find('/n')
    return content[i][begin:end]
    
def getCustoms(i):
    f = open("/home/alix/Documents/ALIX/ALIX/Conversation/Customs", "r")
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
def learning_mode(lock_use, presence_use):
    global push_button_is_pressed
    runProcess('thoughtful')
    ttsCloud("Qual capítulo você gostaria de aprender?")
    push_button_is_pressed = False
    while True:
        if push_button_is_pressed:
            push_button_is_pressed = False
            frase = get_transcription_from_whisper("pt")
            if frase is not None:
                if "capítulo" in frase:
                    start_time = time.time()
                    for j in range(10):
                        if getLesson(j).lower() in frase:
                            chapter = getLesson(j).lower()
                            runProcess('happy')
                            ttsCloud("Vamos fazer as atividades de " + chapter)
                            #reading_mode(chapter,presence_use)
                            #listening_mode(chapter, presence_use)
                            nota = assessment_mode(chapter,presence_use)
                            print(nota)
                            runProcess('happy')
                            ttsCloud("Você terminou o capítulo. Muito bem")
                            final_time = time.time()
                            #Tempo gasto na atividade
                            total_time = final_time - start_time
                            addResults(total_time, nota, chapter)
                            print(total_time)
                            if lock_use == True:
                                #runProcess('talking')
                                ttsCloud("Aperte o botão para abrir o compartimento de recompensas.")
                                lockable_compartment()
                            break
                if "parar" in frase:
                    runProcess('talking')
                    ttsCloud("Certo, finalizando modo de estudo.")
                    break

def reading_mode(chapter,presence_use):
    global push_button_is_pressed
    if presence_use == True:
        break_count = 0
        current_time = time.time()
        a_time = current_time
        spomodoro_time = current_time
        runProcess('thoughtful')
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
                        runProcess('happy')
                        ttsCloud("Certo, finalizando modo de estudo da leitura.")
                        break
                    else:
                        runProcess('thoughtful')
                        ttsCloud("Não entendi o que você disse. Você já terminou a leitura?")
                    
            if current_time - a_time > absence_time:
                print(time.time() - a_time)
                runProcess('thoughtful')
                ttsCloud("Será que você ainda está ai? Vou te procurar.")
                presence = presence_detection()
                if presence == True:
                    a_time = time.time() 
                elif presence == False:
                    break
            
            if current_time - spomodoro_time > short_pomodoro:
                if(break_count < 4):
                    print(time.time() - spomodoro_time)
                    print(short_pomodoro)
                    runProcess('standby')
                    ttsCloud("Está na hora da sua pausa de 5 minutos.")
                    sleep(5)
                    runProcess('standby')
                    ttsCloud("Pausa finalizada. Está na hora de voltar")
                    spomodoro_time = time.time() 
                    break_count += 1
                else:
                    runProcess('standby')
                    ttsCloud("Está na hora da sua pausa de 15 minutos.")
                    sleep(10)
                    runProcess('standby')
                    ttsCloud("Pausa finalizada. Está na hora de voltar")
                    spomodoro_time = time.time()
                    break_count = 0  # Reset the break count after a long break
    
    elif presence_use == False:
        break_count = 0
        current_time = time.time()
        a_time = current_time 
        spomodoro_time = current_time
        runProcess('thoughtful')
        ttsCloud("Você já pode iniciar  a leitura do caítulo" + chapter)
        ttsCloud("Lembre que ao finalizar a leitura do capítulo, me avise apertando o botão.")
        while True:
            current_time = time.time()
            if push_button_is_pressed:
                push_button_is_pressed = False
                frase = get_transcription_from_whisper("pt")
                if frase is not None:
                    if "terminei" in frase or "acabei" in frase or "sim" in frase or "finalizei" in frase:
                        runProcess('happy')
                        ttsCloud("Certo, finalizando modo de estudo da leitura.")
                        break
                    else:
                        runProcess('thoughtful')
                        ttsCloud("Não entendi o que você disse. Você já terminou a leitura?")
            
            if current_time - spomodoro_time > short_pomodoro:
                if(break_count < 4):
                    print(time.time() - spomodoro_time)
                    print(short_pomodoro)
                    runProcess('standby')
                    ttsCloud("Está na hora da sua pausa de 5 minutos.")
                    sleep(5)
                    runProcess('happy')
                    ttsCloud("Pausa finalizada. Está na hora de voltar")
                    spomodoro_time = time.time() 
                    break_count += 1
                else:
                    runProcess('standby')
                    ttsCloud("Está na hora da sua pausa de 15 minutos.")
                    sleep(10)
                    runProcess('happy')
                    ttsCloud("Pausa finalizada. Está na hora de voltar")
                    spomodoro_time = time.time()
                    break_count = 0  # Reset the break count after a long break

def listening_mode(chapter,presence_use):
    global push_button_is_pressed
    runProcess('thoughtful')
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
                        runProcess('thoughtful')
                        ttsCloud("Muito bem. Escute com atenção e divirta-se.")
                        play_music(chapter)
                        runProcess('thoughtful')
                        ttsCloud("Espero que você tenha aprendido a pronunciar muitas palavras novas. Escute quantas vezes você quiser.")
                        break
                    if "não" in frase:
                        runProcess('thoughtful')
                        ttsCloud("Tudo bem, vamos para a atividade de avaliação.")
                        break
                    else:
                        runProcess('thoughtful')
                        ttsCloud("Não entendi o que você disse. Me responda Sim ou Não para fazer atividade de escuta.")
                        break
            
            if current_time - a_time > absence_time:
                print(time.time() - a_time)
                runProcess('thoughtful')
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
                        runProcess('thoughtful')
                        ttsCloud("Muito bem. Escute com atenção e divirta-se.")
                        play_music(chapter)
                        runProcess('thoughtful')
                        ttsCloud("Espero que você tenha aprendido a pronunciar muitas palavras novas. Escute quantas vezes você quiser.")
                        break
                    if "não" in frase:
                        runProcess('thoughtful')
                        ttsCloud("Tudo bem, vamos para a atividade de avaliação.")
                        break
                    else:
                        runProcess('thoughtful')
                        ttsCloud("Não entendi o que você disse. Me responda Sim ou Não para fazer atividade de escuta.")
                        break

#adjectives primeira pergunta tá errada
def assessment_mode(chapter,presence_use):
    global push_button_is_pressed
    runProcess('thoughtful')
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
                        #runProcess('thoughtful')
                        ttsCloud("Vamos começar.")
                        error_count = 0
                        nota = 0 
                        for i in range(getRange(chapter)):
                            runProcess('thoughtful')
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
                                            if(i < ((getRange(chapter))-1)):
                                                runProcess('thoughtful')
                                                ttsCloud("Acertou, vamos para a próxima pergunta")
                                                error_count = 0
                                                nota += 1
                                                break
                                            else:
                                                runProcess('thoughtful')
                                                ttsCloud("Você finalizou a atividade. Parabéns")
                                                nota += 1
                                                #nota
                                                media = (nota/getRange(chapter)) * 10
                                                #data e hora de termino
                                                timestamp = time.time()
                                                date_time = datetime.fromtimestamp(timestamp)
                                                str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
                                                outer_break = True
                                                return media
                                        else:
                                            runProcess('thoughtful')
                                            ttsCloud("Está errado tente outra vez")
                                            push_button_is_pressed = False
                                            error_count += 1
                                            if error_count >=3:
                                                runProcess('thoughtful')
                                                ttsCloud("Parece que você está com dificuldades. Gostaria de pular essa questão?")
                                                push_button_is_pressed = False
                                                while True:
                                                    if push_button_is_pressed:
                                                        push_button_is_pressed = False
                                                        a_time = time.time() 
                                                        frase = get_transcription_from_whisper("pt")
                                                        if frase is not None:
                                                            if "sim" in frase:
                                                                runProcess('thoughtful')
                                                                ttsCloud("Tudo bem, vamos para a próxima pergunta")
                                                                error_count = 0
                                                                skip_question = True
                                                                break
                                                            if "não" in frase:
                                                                runProcess('thoughtful')
                                                                ttsCloud(getQuestion(chapter,i))
                                                                break
                                                            else:
                                                                runProcess('thoughtful')
                                                                ttsCloud("Não entendi. Me responda se você quer pular a questão com Sim ou Não.")
                                        if skip_question:
                                            skip_question = False
                                            break
                                
                                if current_time - a_time > absence_time:
                                    print(time.time() - a_time)
                                    runProcess('thoughtful')
                                    ttsCloud("Será que você ainda está ai? Vou te procurar.")
                                    presence = presence_detection()
                                    if presence == True:
                                        a_time = time.time() 
                                    elif presence == False:
                                        break 
                                if current_time - spomodoro_time > short_pomodoro:
                                    if(break_count < 4):
                                        print(time.time() - spomodoro_time)
                                        print(short_pomodoro)
                                        runProcess('thoughtful')
                                        ttsCloud("Está na hora da sua pausa de 5 minutos.")
                                        sleep(5)
                                        runProcess('thoughtful')
                                        ttsCloud("Pausa finalizada. Está na hora de voltar")
                                        spomodoro_time = time.time() 
                                        break_count += 1
                                    else:
                                        runProcess('thoughtful')
                                        ttsCloud("Está na hora da sua pausa de 15 minutos.")
                                        sleep(10)
                                        runProcess('thoughtful')
                                        ttsCloud("Pausa finalizada. Está na hora de voltar")
                                        spomodoro_time = time.time()
                                        break_count = 0  # Reset the break count after a long break

                                if outer_break:
                                    break 
                    
                    elif presence_use == False:
                        #runProcess('thoughtful')
                        ttsCloud("Vamos começar.")
                        error_count = 0
                        nota = 0 
                        for i in range(getRange(chapter)):
                            runProcess('thoughtful')
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
                                            if(i < ((getRange(chapter))-1)):
                                                runProcess('thoughtful')
                                                ttsCloud("Acertou, vamos para a próxima pergunta")
                                                error_count = 0
                                                nota += 1
                                                break
                                            else:
                                                runProcess('thoughtful')
                                                ttsCloud("Você finalizou a atividade. Parabéns")
                                                nota += 1
                                                #nota
                                                media = (nota/getRange(chapter)) * 10
                                                #data e hora de termino
                                                timestamp = time.time()
                                                date_time = datetime.fromtimestamp(timestamp)
                                                str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
                                                outer_break = True
                                                return media
                                        else:
                                            runProcess('thoughtful')
                                            ttsCloud("Está errado tente outra vez")
                                            error_count += 1
                                            if error_count >=3:
                                                runProcess('thoughtful')
                                                ttsCloud("Parece que você está com dificuldades. Gostaria de pular essa questão?")
                                                push_button_is_pressed = False
                                                while True:
                                                    if push_button_is_pressed:
                                                        push_button_is_pressed = False
                                                        frase = get_transcription_from_whisper("pt")
                                                        if frase is not None:
                                                            if "sim" in frase:
                                                                runProcess('thoughtful')
                                                                ttsCloud("Tudo bem, vamos para a próxima pergunta")
                                                                error_count = 0
                                                                skip_question = True
                                                                break
                                                            if "não" in frase:
                                                                runProcess('thoughtful')
                                                                ttsCloud(getQuestion(chapter,i))
                                                                break
                                                            else:
                                                                runProcess('thoughtful')
                                                                ttsCloud("Não entendi. Me responda se você quer pular a questão com Sim ou Não.")
                                        if skip_question:
                                            skip_question = False
                                            break
                                
                                if current_time - spomodoro_time > short_pomodoro:
                                    if(break_count < 4):
                                        print(time.time() - spomodoro_time)
                                        print(short_pomodoro)
                                        runProcess('thoughtful')
                                        ttsCloud("Está na hora da sua pausa de 5 minutos.")
                                        sleep(5)
                                        runProcess('thoughtful')
                                        ttsCloud("Pausa finalizada. Está na hora de voltar")
                                        spomodoro_time = time.time() 
                                        break_count += 1
                                    else:
                                        runProcess('thoughtful')
                                        ttsCloud("Está na hora da sua pausa de 15 minutos.")
                                        sleep(10)
                                        runProcess('thoughtful')
                                        ttsCloud("Pausa finalizada. Está na hora de voltar")
                                        spomodoro_time = time.time()
                                        break_count = 0  # Reset the break count after a long break

                                if outer_break:
                                    break 

                elif "não" in frase:
                    runProcess('thoughtful')
                    ttsCloud("Certo, finalizando modo de estudo.")
                    break
                else:
                    runProcess('thoughtful')
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
                runProcess('happy', 'standby')
                ttsCloud("Certo, finalizando modo conversa.")
                sleep(1)
                continue_conversation = False
            else:
                runProcess('thoughtful')
                conversation =generate_response(frase)
                ttsCloud(conversation)

#--------------------------Other functions-----------------------------------
def runProcess(expressionName, movementName = None):
    global processRun
    global p

    if not movementName:
        movementName = expressionName

    executando = 'exec ' + expressionAddress + expressionName
    if processRun:
        p.kill()
        
    p=subprocess.Popen(executando, shell=True, preexec_fn=os.setsid)
    processRun = True
    subprocess.Popen('python ' + movementAddress + movementName + '.py',shell=True, preexec_fn=os.setsid)
    
def GPIO_Init():
    pygame.init()
    pygame.mixer.init()
    GPIO.setwarnings(False) # Ignore warning for now
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    GPIO.setup(solenoid_pin, GPIO.OUT) 
    GPIO.setup(magnetic_sensor_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(push_button_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.add_event_detect(push_button_pin, GPIO.FALLING, 
        callback=push_button_handler, bouncetime=100)

def presence_detection():
    subprocess.Popen('python /home/alix/Documents/ALIX/ALIX/Conversation/baseRotation.py',shell=True, preexec_fn=os.setsid)
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
            runProcess('thoughtful')
            ttsCloud("Você ainda está aí. Você pode me responder apertando o botão.")
            presence = True
            return presence

        # Check if the face detection time has exceeded the limit
        if time.time() - face_time > presence_time:
            # Release the camera and close the OpenCV window
            cam.release()
            cv2.destroyAllWindows()
            runProcess('thoughtful')
            ttsCloud("Não te encontrei, finalizando atividade.")
            #data e hora de ausência
            timestamp = time.time()
            date_time = datetime.fromtimestamp(timestamp)
            str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
            addAbsence(str_date_time)
            presence = False
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

    runProcess('thoughtful')
    ttsCloud("Compartimento de segurança fechado com sucesso.")

def push_button_handler(sig):
    global push_button_is_pressed
    #GPIO.cleanup()
    push_button_is_pressed = True

def pomodoro():
    print("Pomodoro")
#----------------------------Main function----------------------------
if __name__ == '__main__':
    GPIO_Init()
    while True:
        if GPIO.input(push_button_pin) == GPIO.LOW:
            frase = get_transcription_from_whisper("pt")
            if frase is not None:
                if "estudar" in frase:
                    runProcess('thoughtful')
                    ttsCloud("Certo. Precisamos realizar umas configurações antes de iniciar as atividades.")
                    runProcess('thoughtful')
                    ttsCloud("Você vai utilizar o compartimento de recompensas?")
                    push_button_is_pressed = False
                    while True:
                        if GPIO.input(push_button_pin) == GPIO.LOW:
                            frase = get_transcription_from_whisper("pt")
                            if frase is not None:
                                if "sim" in frase:
                                    runProcess('thoughtful')
                                    ttsCloud("Certo. Aperte o botão para destravar o compartimento e abra a porta")
                                    lockable_compartment()
                                    lock_use = True
                                    runProcess('thoughtful')
                                    ttsCloud("Compartimento de recompensas ativado.")
                                    break
                                if "não" in frase:
                                    runProcess('thoughtful')
                                    ttsCloud("Ok. Compartimento de recompensas desativado.")
                                    lock_use = False
                                    break
                                else:
                                    runProcess('thoughtful')
                                    ttsCloud("Não entendi. Me responda se você quer usar o compartimento de recompensas com Sim ou Não.")
                                    
                    runProcess('thoughtful')
                    ttsCloud("Você gostaria de usar a camera para detecção de presença durante as atividades?")
                    while True:
                        if GPIO.input(push_button_pin) == GPIO.LOW:
                            frase = get_transcription_from_whisper("pt")
                            if frase is not None:
                                if "sim" in frase:
                                    presence_use = True
                                    runProcess('thoughtful')
                                    ttsCloud("Ok, detecção de presença ativado.")
                                    break
                                if "não" in frase:
                                    runProcess('thoughtful')
                                    ttsCloud("Tudo bem, não irei utilizar a detecção de presença durante a atividade.")
                                    presence_use = False
                                    break
                                else:
                                    runProcess('thoughtful')
                                    ttsCloud("Não é uma opção, diga sim ou não")
                    #runProcess('thoughtful')
                    ttsCloud("Vamos aprender inglês!!!")
                    learning_mode(lock_use, presence_use)
                    
                elif "pergunta" in frase:
                    runProcess('thoughtful')
                    ttsCloud("Legal. O que você gostaria de perguntar?")
                    conversation_mode()
                elif "tchau" in frase:
                    runProcess('thoughtful')
                    ttsCloud("Até mais, mal posso esperar para conversar com você de novo.")
                    os.system("sudo shutdown -h now")  
                else:
                    runProcess('thoughtful')
                    ttsCloud("Não entendi o que você falou. ")


    
                