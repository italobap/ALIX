import os
import numpy as np
import pyaudio
import openai
import wave
from pydub import AudioSegment
from gtts import gTTS
import playsound
import time
import cv2
from neuralintents import GenericAssistant
from senha import API_KEY
import speech_recognition as sr
import requests
import json

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml') # insert the full path to haarcascade file if you encounter any problem
limit_input_time = 5
language="pt-br"
language_whisper="pt"

# Set your OpenAI API key
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
link = "https://api.openai.com/v1/chat/completions"

#assistant = GenericAssistant("intents.json")
#assistant.train_model()

def get_transcription_from_whisper():

    # Set the audio parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 4096
    SILENCE_THRESHOLD = 300  # Silence threshold
    dev_index = 2 # device index found by p.get_device_info_by_index(ii)
    SPEECH_END_TIME = 1.0  # Time of silence to mark the end of speech

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input_device_index = dev_index,input = True,
                        frames_per_buffer=CHUNK)

    print("Recording...Waiting for speech to begin.")
    detection_time = time.time()  # Initialize the detection time to allow the first detection
    frames = []
    silence_frames = 0
    is_speaking = False
    total_frames = 0

    while True:
        current_time = time.time()
        data = stream.read(CHUNK)
        frames.append(data)
        total_frames += 1

        # Convert audio chunks to integers
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Check if user has started speaking
        if np.abs(audio_data).mean() > SILENCE_THRESHOLD:
            is_speaking = True

        # Detect if the audio chunk is silence
        if is_speaking:
            if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
                silence_frames += 1
            else:
                silence_frames = 0

        # End of speech detected
        if is_speaking and silence_frames > SPEECH_END_TIME * (RATE / CHUNK):
            print("End of speech detected.")
            detection_time = current_time
            # Stop Recording
            stream.stop_stream()
            stream.close()
            audio.terminate()

            print("Finished recording.")
            combined_audio_data = b''.join(frames)
            # Convert raw data to an AudioSegment object
            audio_segment = AudioSegment(
                data=combined_audio_data,
                sample_width=audio.get_sample_size(FORMAT),
                frame_rate=RATE,
                channels=CHANNELS
            )

            # Export as a compressed MP3 file with a specific bitrate
            audio_segment.export("output_audio_file.mp3", format="mp3", bitrate="32k")

            # Open the saved file to send to the API
            with open("output_audio_file.mp3", "rb") as f:
                transcript = openai.Audio.transcribe("whisper-1", f)
                #transcript = openai.Audio.transcribe("whisper-1", f, language = language_whisper)

            transcription = transcript['text']
            print("Transcript:", transcription)
            response_time = time.time()
            speak(transcription)
            print("--- %s seconds ---" % (time.time()-response_time))
            break

        # Check if it has been more than 5 seconds without speech
        if current_time - detection_time >= limit_input_time:
            print("No speech detected. Stopping recording.")
            # Stop Recording
            stream.stop_stream()
            stream.close()
            audio.terminate()

            print("Finished recording.")
            presence_detection()
            break

def speak(text):
    tts = gTTS(text= text, lang=language)
    filename = "output.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove("output.mp3")

def presence_detection():
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

def generate_response(prompt):
    response = openai.Completion.create(
        engine = "gpt-3.5-turbo",
        prompt = prompt,
        max_tokens = 30,
        n =1,
        stop=None,
        temperature = 0.5,
    )
    gpt_response = response["choices"][0]["text"]
    return gpt_response

def generate_response2(prompt):
    body_mensagem={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt + "em poucas palavras"}]
    }
    body_mensagem = json.dumps(body_mensagem)
    requisicao = requests.post(link, headers=headers, data= body_mensagem)
    resposta = requisicao.json()
    mensagem = resposta["choices"][0]["message"]["content"]
    return mensagem

if __name__ == '__main__':
    while True:
        #get_transcription_from_whisper()
        frase = get_transcription_from_sr()
        if "conversação" in frase:
            speak("iniciando modo conversação")

        if "parar" in frase:
            speak("Tchau, até mais")
            break 

        if "pergunta" in frase:
            speak("Faça uma pergunta")
            while True:
                frase = get_transcription_from_sr()
                current_time = time.time()
                if "parar" in frase:
                    speak("Voltando para programa inicial")
                    break
                else:
                    conversation =generate_response2(frase)
                    print("--- %s seconds ---" % (time.time()-current_time))
                    speak(conversation)