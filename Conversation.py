import os
import numpy as np
import pyaudio
import openai
import wave
from pydub import AudioSegment
from gtts import gTTS
import playsound
import time
# Set your OpenAI API key
openai.api_key = "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"

def get_transcription_from_whisper():


    # Set the audio parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 4096
    SILENCE_THRESHOLD = 300  # Silence threshold
    dev_index = 2 # device index found by p.get_device_info_by_index(ii)
    SPEECH_END_TIME = 1.0  # Time of silence to mark the end of speech

    dev_index = 2 # device index found by p.get_device_info_by_index(ii)
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input_device_index = dev_index,input = True,
                        frames_per_buffer=CHUNK)

    print("Recording...Waiting for speech to begin.")

    frames = []
    silence_frames = 0
    is_speaking = False
    total_frames = 0

    while True:
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
            break

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

    # Return the transcript text
    return transcript['text']



def get_transcription_from_audio_file(file_name):

    with open(file_name, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)

    # Return the transcript text
    return transcript['text']

def speak(text):
    tts = gTTS(text= text, lang ="pt-br")
    filename = "output.mp3"
    tts.save(filename)
    playsound.playsound(filename)

# Example usage of the function
if __name__ == '__main__':
    
    transcription = get_transcription_from_whisper()
    print("Transcript:", transcription)
    start_time = time.time()
    speak(transcription)
    print("--- %s seconds ---" % (time.time()-start_time))