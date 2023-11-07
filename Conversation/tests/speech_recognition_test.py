import speech_recognition as sr
from datetime import date
from time import sleep

r = sr.Recognizer()
mic = sr.Microphone()

print("hello")

def start():
    while True:
        print("Diga Alix para começar a utilizar o programa")
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                if transcription.lower() == "start":
                    filename = "input.wav"
                    print("Speak with me:...")    
                    with sr.Microphone() as source:
                        recognizer = sr.Recognizer()
                        source.pause_threshold = 1
                        audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())

                        text = get_transcription_from_sr(filename)
                        if text:
                            print(f"You said: {text}")

                            response = generate_response(text)
                            print(f"GPT-3 says: {response}")

                            speak(response)
                    if transcription.lower() == "stop":
                        print("Finalizando programa")
                        break
                    if transcription.lower() == "hello":
                        speak("How are you")

            except Exception as e:
                print("Aconteceu um erro : {}".format(e))