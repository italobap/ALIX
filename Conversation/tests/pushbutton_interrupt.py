import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from gpiozero import Button
import time 
from time import sleep

solenoid_pin = 15
push_button_pin = 19 #gpio10
push_button_is_pressed = False
magnetic_sensor_pin = 32 #gpio12
limit_time = 5
GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(push_button_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP)

def push_button_handler(sig, frame):
    global push_button_is_pressed
    GPIO.cleanup()
    push_button_is_pressed = True


def reading_mode():
    global push_button_is_pressed
    break_count = 0
    absence_time = 100
    short_pomodoro = 10 #ciclo de 25 min#
    a_time = time.time() * 50
    spomodoro_time = a_time
    print(a_time)
    print("Qual capítulo você irá ler?")
    frase = "capitulo"
    while True:
        if push_button_is_pressed:
            push_button_is_pressed = False
            if frase is not None:
                frase = None
                if "capítulo" in frase or "capitulo" in frase or "sentimentos" in frase:
                    print("Legal. Quando terminar a leitura, lembre de me avisar.")
                    # Valor inicial de absence_time
                    a_time = time.time()
                    # Valor inicial tempo de break reminder
                    spomodoro_time = time.time()
                    # Valor inicial do tempo de atividade
                    start_time = time.time()
                    # continue
                    
                if "parar" in frase:
                    print("Certo, finalizando modo de estudo da leitura.")
                    total_time = (time.time() - start_time) / 60  # Calculate total reading time in minutes
                    print("Tempo total = " + str(total_time))
                    break
                
        current_time = time.time()
        if current_time - a_time > absence_time:
            print(time.time() - a_time)
            print("Será que você ainda está ai? Vou te procurar.")
            sleep(5)
            # Reinicia o valor inicial de absence_time
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
                print(time.time() - spomodoro_time)
                print("Está na hora da sua pausa de 15 minutos.")
                sleep(10)
                print("Pausa finalizada. Está na hora de voltar")
                spomodoro_time = time.time()
                break_count = 0  # Reset the break count after a long break

if __name__ == "__main__":
    GPIO.add_event_detect(push_button_pin, GPIO.FALLING, 
            callback=push_button_handler, bouncetime=100)
    reading_mode()
