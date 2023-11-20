import time 
from time import sleep
import threading
from threading import Event

#Times
record_time = 10 # 10 seconds 
presence_time = 7.5 # 15 seconds
absence_time = 5 #seconds
short_pomodoro = 2 # 20seconds
break_count_limit = 4

#Global Variables
presence = False
presence_use = False
push_button_is_pressed = False
thread = threading.Thread()

# create the event
event = Event()

def presence_detection():
    print("Detectado")

def thread_time(presence_use):
    break_count = 0
    current_time = time.time()
    a_time = current_time + absence_time
    spomodoro_time = current_time + short_pomodoro

    while True:
        if event.is_set():
            print("Parando a Thread")
            break

        current_time = time.time()

        if presence_use == True:
            if current_time - a_time > absence_time:
                print(time.time() - a_time)
                print("Será que você ainda está ai? Vou te procurar.")
                presence = presence_detection()
                if presence == True:
                    a_time = time.time() 
                elif presence == False:
                    break

        if current_time - spomodoro_time > short_pomodoro:
            if(break_count < break_count_limit):
                print(time.time() - spomodoro_time)
                print(short_pomodoro)
                print("Está na hora da sua pausa de 5 minutos.")
                sleep(5)
                print("Pausa finalizada. Está na hora de voltar.")
                spomodoro_time = time.time() 
                break_count += 1
            else:
                print("Está na hora da sua pausa de 15 minutos.")
                sleep(10)
                print("Pausa finalizada. Está na hora de voltar.")
                spomodoro_time = time.time()
                break_count = 0  # Reset the break count after a long break

def start_thread():
    thread = threading.Thread(target=thread_time, args=(event,))
    thread.daemon = True
    thread.start()

def learning_mode():
    global event
    print("Qual capítulo você gostaria de aprender?")
    start_thread()
    print("learning_mode")
    sleep(10)
    event.set()

    sleep(10)
    event.clear()
    print("learning_mode")

    start_thread()
    sleep(10)
    event.set()

if __name__ == '__main__':
    # create and configure a new thread
    
    learning_mode()