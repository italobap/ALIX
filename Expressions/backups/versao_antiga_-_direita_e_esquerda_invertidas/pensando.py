from adafruit_servokit import ServoKit
import threading
import time

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
anguloOmbroDireitoNeutro = 180

anguloBracoEsquerdo = 30
anguloOmbroEsquerdo = 30

# Pensando
anguloBracoDireito = 180
anguloOmbroDireito = 100

anguloBracoEsquerdo = 30
anguloOmbroEsquerdo = 30
anguloCabeca = 90

kit.servo[bracoEsquerdo].angle = anguloBracoEsquerdo
kit.servo[ombroEsquerdo].angle = anguloOmbroEsquerdo
kit.servo[cabeca].angle = anguloCabeca

def moveBracoDireito():
    for i in range (anguloBracoDireitoNeutro, anguloBracoDireito):
        kit.servo[bracoDireito].angle = i
        time.sleep(velocidade)

    time.sleep(3)
    
    for i in range (anguloBracoDireito, anguloBracoDireitoNeutro, -1):
        kit.servo[bracoDireito].angle = i
        time.sleep(velocidade)

def moveOmbroDireito():
    for i in range (anguloOmbroDireitoNeutro, anguloOmbroDireito, -1):
        kit.servo[ombroDireito].angle = i
        time.sleep(velocidade)

    time.sleep(3)

    for i in range (anguloOmbroDireito, anguloOmbroDireitoNeutro):
        kit.servo[ombroDireito].angle = i
        time.sleep(velocidade)

if __name__ == "__main__":
    threads = []  # Lista para armazenar as threads

    t = threading.Thread(target=moveBracoDireito, args=())
    threads.append(t)  # Adicione a thread à lista

    t = threading.Thread(target=moveOmbroDireito, args=())
    threads.append(t)  # Adicione a thread à lista

    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    print("Done!")



