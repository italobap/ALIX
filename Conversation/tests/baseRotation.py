from adafruit_servokit import ServoKit
import threading
import time

kit = ServoKit(channels=16)

base = 7
startBaseAngle = 90

velocidade = 0.05
variacao = 60

def moveMotor(servo, startAngle, variacaoAngle, delay):
        # Exemplo: comeca em startAngle(150) vai movimentar ate 180 (startAngle + variacaoAngle (30))
        endAngle = startAngle + variacaoAngle
        for i in range (startAngle, endAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)

        # Exemplo: comeca em 180, final do movimento anterior, e vai ate 120
        for i in range (endAngle, startAngle - variacaoAngle, -1):
            kit.servo[servo].angle = i
            time.sleep(delay)
        
        for i in range (startAngle - variacaoAngle, startAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)

if __name__ == "__main__":
    threads = []  # Lista para armazenar as threads

    #moveBase
    t = threading.Thread(target=moveMotor, args=(base, startBaseAngle, variacao, velocidade))  # Crie uma thread para cada servo
    threads.append(t)  # Adicione a thread a lista
        
    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    print("Done!")
