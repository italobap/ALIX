import time
import keyboard
from time import sleep

absence_time = 100
short_pomodoro = 5 #ciclo de 25 min#
global a_time
global spomodoro_time 

def pomodoro():
    break_count = 0

    current_time = time.time()
    if current_time- a_time > absence_time:
        print(time.time() - a_time)
        print("Será que você ainda está ai? Vou te procurar.")
        a_time = time.time() 
    
    if current_time - spomodoro_time > short_pomodoro:
        if(break_count < 4):
            print(time.time() - spomodoro_time)
            print(short_pomodoro)
            print("Está na hora da sua pausa de 5 minutos.")
            sleep(5)
            print("Pausa finalizada. Está na hora de voltar")
            spomodoro_time = time.time() 
            break_count += 1
        else:
            print("Está na hora da sua pausa de 15 minutos.")
            sleep(10)
            print("Pausa finalizada. Está na hora de voltar")
            spomodoro_time = time.time()
            break_count = 0  # Reset the break count after a long break

if __name__ == '__main__':
    print("Iniciando pomodoro")
    pomodoro()
    print("Pomodoro inicializado")
    spomodoro_time = time.time()
    a_time = time.time()
    while True:
        pomodoro()