from adafruit_servokit import ServoKit
import threading
import time

anguloBracoEsquerdo = 31
anguloBracoDireito = 150
anguloCabeca = 110

bracoEsquerdo = 2
ombroEsquerdo = 3
bracoDireito = 0
ombroDireito = 1
cabeca = 4
variacao = 30
velocidade = 0.01
# the robot move both arms up and down and rotate its head sideways while displaying a happy face.
kit = ServoKit(channels=16)

kit.servo[bracoEsquerdo].angle = anguloBracoEsquerdo
kit.servo[ombroEsquerdo].angle = anguloBracoEsquerdo

kit.servo[bracoDireito].angle = anguloBracoDireito
kit.servo[ombroDireito].angle = anguloBracoDireito

def moveMotor(servo, startAngle, variacaoAngle, delay):
    for j in range(5):
        
        # Exemplo: começa em startAngle(150) vai movimentar até 180 (startAngle + variacaoAngle (30))
        endAngle = startAngle + variacaoAngle
        for i in range (startAngle, endAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)

        # Exemplo: começa em 180, final do movimento anterior, e vai até 120
        for i in range (endAngle, startAngle - variacaoAngle, -1):
            kit.servo[servo].angle = i
            time.sleep(delay)
        
        # cabeca
        for i in range (startAngle - variacaoAngle, startAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)

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

    #visao de frente
    #braco direito
    for i in range(2):
        t = threading.Thread(target=moveMotor, args=(i, anguloBracoDireito, variacao, velocidade))  # Crie uma thread para cada servo
        threads.append(t)  # Adicione a thread à lista
        
    #braco esquerdo
    for i in range(2,4):
        t = threading.Thread(target=moveMotor, args=(i, anguloBracoEsquerdo, -1*variacao, velocidade))  # Crie uma thread para cada servo
        threads.append(t)  # Adicione a thread à lista
    
    # cabeca
    t1 = threading.Thread(target=moveMotor, args=(4, anguloCabeca, variacao, velocidade))  # Crie uma thread para cada servo
    threads.append(t1)  # Adicione a thread à lista
        
    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    print("Done!")

