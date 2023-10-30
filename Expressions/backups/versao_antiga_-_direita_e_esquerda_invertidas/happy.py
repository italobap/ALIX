from adafruit_servokit import ServoKit
import threading
import time

anguloBracoDireito = 155
anguloOmbroDireito = 180

anguloBracoEsquerdo = 30
anguloOmbroEsquerdo = 30

anguloCabeca = 110

bracoEsquerdo = 2
ombroEsquerdo = 3
bracoDireito = 0
ombroDireito = 1
cabeca = 4
variacao = 150
velocidade = 0.005
# the robot will raise its arms and display a happy face.
kit = ServoKit(channels=16)

kit.servo[bracoDireito].angle = anguloBracoDireito
kit.servo[ombroDireito].angle = anguloOmbroDireito

kit.servo[bracoEsquerdo].angle = anguloBracoEsquerdo
kit.servo[ombroEsquerdo].angle = anguloOmbroEsquerdo
kit.servo[cabeca].angle = anguloCabeca

def moveEsquerdo(servo, startAngle, variacaoAngle, delay):
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

def moveDireito(servo, startAngle, variacaoAngle, delay):
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

    t = threading.Thread(target=moveEsquerdo, args=(ombroEsquerdo, anguloBracoEsquerdo, variacao, velocidade))
    threads.append(t)  # Adicione a thread à lista

    t = threading.Thread(target=moveDireito, args=(ombroDireito, anguloOmbroDireito, -1*variacao, velocidade))
    threads.append(t)  # Adicione a thread à lista

    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    print("Done!")


