# The robot will raise its arm and look as if touching it to its head with a concentrated facial expression.

from adafruit_servokit import ServoKit
import threading
import time

kit = ServoKit(channels=16)

bracoEsquerdo = 0 #Esquerda do ALIX
ombroEsquerdo = 1 #Esquerda do ALIX
bracoDireito = 2 #Direita do ALIX
ombroDireito = 3 #Direita do ALIX
cabeca = 4

# Standby
anguloBracoEsquerdoStandby = 155
anguloOmbroEsquerdoStandby = 180
anguloBracoDireito = 30
anguloOmbroDireito = 30

# Pensando
anguloBracoEsquerdo = 180
anguloOmbroEsquerdo = 100
anguloBracoDireito = 30
anguloOmbroDireito = 30

anguloCabeca = 90
velocidade = 0.01

kit.servo[bracoDireito].angle = anguloBracoDireito
kit.servo[ombroDireito].angle = anguloOmbroDireito
kit.servo[cabeca].angle = anguloCabeca

def moveBracoEsquerdo():
    for i in range (anguloBracoEsquerdoStandby, anguloBracoEsquerdo):
        kit.servo[bracoEsquerdo].angle = i
        time.sleep(velocidade)

    time.sleep(3)
    
    for i in range (anguloBracoEsquerdo, anguloBracoEsquerdoStandby, -1):
        kit.servo[bracoEsquerdo].angle = i
        time.sleep(velocidade)

def moveOmbroEsquerdo():
    for i in range (anguloOmbroEsquerdoStandby, anguloOmbroEsquerdo, -1):
        kit.servo[ombroEsquerdo].angle = i
        time.sleep(velocidade)

    time.sleep(3)

    for i in range (anguloOmbroEsquerdo, anguloOmbroEsquerdoStandby):
        kit.servo[ombroEsquerdo].angle = i
        time.sleep(velocidade)

if __name__ == "__main__":
    threads = []  # Lista para armazenar as threads

    t = threading.Thread(target=moveBracoEsquerdo, args=())
    threads.append(t)  # Adicione a thread a lista

    t = threading.Thread(target=moveOmbroEsquerdo, args=())
    threads.append(t)  # Adicione a thread a lista

    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    print("Done!")
