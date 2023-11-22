import pygame
import os
#os.environ['SDL_AUDIODRIVER'] = 'pulseaudio'
music_path = "/home/alix/Documents/ALIX/ALIX/alix songs/"
#music_path = "C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/alix songs/"

pygame.init()
pygame.mixer.init()

def play_music(music_name):
	pygame.mixer.music.load(music_path + music_name +".mp3")
	pygame.mixer.music.set_volume(1.0)
	pygame.mixer.music.play()

	while pygame.mixer.music.get_busy() == True:
		continue

name = "saudações"
play_music(name)