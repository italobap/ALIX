from adafruit_servokit import ServoKit
import time
import threading

# the robot will raise its arms and display a happy face.
kit = ServoKit(channels=16)

bracoDireito = 0
ombroDireito = 1
bracoEsquerdo = 2
ombroEsquerdo = 3
cabeca = 4
velocidade = 0.01

# neutro
anguloBracoDireitoNeutro = 155

anguloBracoEsquerdoNeutro = 30

# sad
anguloBracoDireito = 180
anguloOmbroDireito = 180

anguloBracoEsquerdo = 0
anguloOmbroEsquerdo = 30
anguloCabeca = 90

kit.servo[ombroDireito].angle = anguloOmbroDireito
kit.servo[ombroEsquerdo].angle = anguloOmbroEsquerdo
kit.servo[cabeca].angle = anguloCabeca

def moveEsquerdo():
    for i in range (anguloBracoEsquerdoNeutro, anguloBracoEsquerdo, -1):
        kit.servo[bracoEsquerdo].angle = i
        time.sleep(velocidade)

    time.sleep(3)
    for i in range (anguloBracoEsquerdo, anguloBracoEsquerdoNeutro):
        kit.servo[bracoEsquerdo].angle = i
        time.sleep(velocidade)

def moveDireito():
    for i in range (anguloBracoDireitoNeutro, anguloBracoDireito):
        kit.servo[bracoDireito].angle = i
        time.sleep(velocidade)
    
    time.sleep(3)

    for i in range (anguloBracoDireito, anguloBracoDireitoNeutro, -1):
        kit.servo[bracoDireito].angle = i
        time.sleep(velocidade)

if __name__ == "__main__":
    threads = []  # Lista para armazenar as threads

    t = threading.Thread(target=moveEsquerdo, args=())
    threads.append(t)  # Adicione a thread à lista

    t = threading.Thread(target=moveDireito, args=())
    threads.append(t)  # Adicione a thread à lista

    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    print("Done!")

