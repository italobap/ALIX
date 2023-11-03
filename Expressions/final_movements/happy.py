# The robot will raise its arms and display a happy face.

from adafruit_servokit import ServoKit
import threading
import time

kit = ServoKit(channels=16)

bracoEsquerdo = 0 #Esquerda do ALIX
ombroEsquerdo = 1 #Esquerda do ALIX
bracoDireito = 2 #Direita do ALIX
ombroDireito = 3 #Direita do ALIX
cabeca = 4

anguloBracoEsquerdo = 155
anguloOmbroEsquerdo = 180
anguloBracoDireito = 30
anguloOmbroDireito = 30
anguloCabeca = 110

variacao = 150
velocidade = 0.005


kit.servo[bracoEsquerdo].angle = anguloBracoEsquerdo
kit.servo[ombroEsquerdo].angle = anguloOmbroEsquerdo

kit.servo[bracoDireito].angle = anguloBracoDireito
kit.servo[ombroDireito].angle = anguloOmbroDireito
kit.servo[cabeca].angle = anguloCabeca

def moveDireito(servo, startAngle, variacaoAngle, delay):
        # while True:
        for j in range(2):
            endAngle = startAngle + variacaoAngle
            for i in range (startAngle, endAngle):
                kit.servo[servo].angle = i
                time.sleep(delay)
            time.sleep(1)
            for i in range (endAngle, startAngle, -1):
                kit.servo[servo].angle = i
                time.sleep(delay)

def moveEsquerdo(servo, startAngle, variacaoAngle, delay):
        # while True:
        for j in range(2):
            endAngle = startAngle + variacaoAngle
            for i in range (startAngle, endAngle, -1):
                kit.servo[servo].angle = i
                time.sleep(delay)
            time.sleep(1)
            for i in range (endAngle, startAngle):
                kit.servo[servo].angle = i
                time.sleep(delay)

if __name__ == "__main__":
    threads = []  # Lista para armazenar as threads

    t = threading.Thread(target=moveDireito, args=(ombroDireito, anguloBracoDireito, variacao, velocidade))
    threads.append(t)  # Adicione a thread a lista

    t = threading.Thread(target=moveEsquerdo, args=(ombroEsquerdo, anguloOmbroEsquerdo, -1*variacao, velocidade))
    threads.append(t)  # Adicione a thread a lista

    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    print("Done!")
