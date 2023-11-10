import os
from google.cloud import texttospeech_v1
from pydub import AudioSegment
from pydub.playback import play

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'speech_gtts_cloud_key.json'

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