import pygame

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("/home/alix/Documents/ALIX/ALIX/alix songs/food.mp3")
pygame.mixer.music.set_volume(1.0)
pygame.mixer.music.play()

while pygame.mixer.music.get_busy() == True:
	continue