from adafruit_servokit import ServoKit
import threading
import time

anguloBracoEsquerdo = 180
anguloBracoDireito = 0
anguloCabeca = 110

bracoEsquerdo = 0
ombroEsquerdo = 1
bracoDireito = 2
ombroDireito = 3
cabeca = 4
variacao = 90
velocidade = 0.01
# the robot will raise its arms and display a happy face.
kit = ServoKit(channels=16)

kit.servo[bracoEsquerdo].angle = anguloBracoEsquerdo
kit.servo[ombroEsquerdo].angle = anguloBracoEsquerdo

kit.servo[bracoDireito].angle = anguloBracoDireito
kit.servo[ombroDireito].angle = anguloBracoDireito
#kit.servo[cabeca].angle = anguloCabeca

def moveMotor(servo, startAngle, variacaoAngle, delay):
        while True:
            endAngle = startAngle + variacaoAngle
            kit.servo[servo].angle = endAngle
            time.sleep(.9)
            kit.servo[servo].angle = startAngle
            #for i in range (startAngle, endAngle):
            #    kit.servo[servo].angle = i
            #    time.sleep(delay)
            #time.sleep(2)
            #for i in range (endAngle, startAngle, -1):
            #    kit.servo[servo].angle = i
            #    time.sleep(delay)

if __name__ == "__main__":
    threads = []  # Lista para armazenar as threads

    # visao de frente
    # ombro direito
    #t = threading.Thread(target=moveMotor, args=(ombroDireito, anguloBracoDireito, variacao, velocidade))
    #threads.append(t)  # Adicione a thread à lista

    t = threading.Thread(target=moveMotor, args=(ombroEsquerdo, anguloBracoEsquerdo-variacao, variacao, velocidade))
    threads.append(t)  # Adicione a thread à lista

    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    print("Done!")

