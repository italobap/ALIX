# (extra) The robot open its arms and display a happy or concentratedq face.

from adafruit_servokit import ServoKit
import time
import threading

kit = ServoKit(channels=16)

bracoEsquerdo = 0 #Esquerda do ALIX
ombroEsquerdo = 1 #Esquerda do ALIX
bracoDireito = 2 #Direita do ALIX
ombroDireito = 3 #Direita do ALIX
cabeca = 4

anguloCabeca = 90
velocidade = 0.01

# Standby
anguloBracoEsquerdoStandby = 155
anguloBracoDireitoStandby = 30

# asking limitado
anguloBracoEsquerdo = 150
anguloOmbroEsquerdo = 180
anguloBracoDireito = 35
anguloOmbroDireito = 30

# asking original
# anguloBracoEsquerdo = 125
# anguloOmbroEsquerdo = 180
# anguloBracoDireito = 60
# anguloOmbroDireito = 30

kit.servo[ombroEsquerdo].angle = anguloOmbroEsquerdo
kit.servo[ombroDireito].angle = anguloOmbroDireito
kit.servo[cabeca].angle = anguloCabeca

def moveDireito():
    for i in range (anguloBracoDireitoStandby, anguloBracoDireito):
        kit.servo[bracoDireito].angle = i
        time.sleep(velocidade)

    time.sleep(3)
    for i in range (anguloBracoDireito, anguloBracoDireitoStandby, -1):
        kit.servo[bracoDireito].angle = i
        time.sleep(velocidade)

def moveEsquerdo():
    for i in range (anguloBracoEsquerdoStandby, anguloBracoEsquerdo, -1):
        kit.servo[bracoEsquerdo].angle = i
        time.sleep(velocidade)
    
    time.sleep(3)

    for i in range (anguloBracoEsquerdo, anguloBracoEsquerdoStandby):
        kit.servo[bracoEsquerdo].angle = i
        time.sleep(velocidade)

if __name__ == "__main__":
    threads = []  # Lista para armazenar as threads

    t = threading.Thread(target=moveDireito, args=())
    threads.append(t)  # Adicione a thread a lista

    t = threading.Thread(target=moveEsquerdo, args=())
    threads.append(t)  # Adicione a thread a lista

    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    print("Done!")
