from adafruit_servokit import ServoKit
import threading
import time

kit = ServoKit(channels=16)

variacao = 30
velocidade = 0.02
bracoDireito = 0
ombroDireito = 1
bracoEsquerdo = 2
ombroEsquerdo = 3
cabeca = 4

anguloBracoDireito = 140
anguloOmbroDireito = 180

anguloBracoEsquerdo = 30
anguloOmbroEsquerdo = 30

anguloCabeca = 90
repeticoes = 2

kit.servo[bracoDireito].angle = anguloBracoDireito
kit.servo[ombroDireito].angle = anguloOmbroDireito

kit.servo[bracoEsquerdo].angle = anguloBracoEsquerdo
kit.servo[ombroEsquerdo].angle = anguloOmbroEsquerdo
kit.servo[cabeca].angle = anguloCabeca

def moveOmbroBracoEsquerdo(servo, startAngle, variacaoAngle, delay):
    for j in range(repeticoes):
        endAngle = startAngle + variacaoAngle
        for i in range (startAngle, endAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)

        # Exemplo: começa em 180, final do movimento anterior, e vai até 120
        for i in range (endAngle, startAngle - variacaoAngle, -1):
            kit.servo[servo].angle = i
            time.sleep(delay)
        
        for i in range (startAngle - variacaoAngle, startAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)

def moveBracoDireito(servo, startAngle, variacaoAngle, delay):
    for j in range(repeticoes):
        endAngle = startAngle + variacaoAngle #110
        for i in range (startAngle, endAngle, -1):
            kit.servo[servo].angle = i
            time.sleep(delay)

        for i in range (endAngle, startAngle - variacaoAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)
        
        for i in range (startAngle - variacaoAngle, startAngle, -1):
            kit.servo[servo].angle = i
            time.sleep(delay)

def moveEsquerdo(servo, startAngle, variacaoAngle, delay):
    for j in range(repeticoes):
        endAngle = startAngle + variacaoAngle
        for i in range (startAngle, endAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)
        
        for i in range (endAngle, startAngle, -1):
            kit.servo[servo].angle = i
            time.sleep(delay)

def moveOmbroDireito(servo, startAngle, variacaoAngle, delay):
    for j in range(repeticoes):
        endAngle = startAngle + variacaoAngle #120
        for i in range (startAngle, endAngle, -1):
            kit.servo[servo].angle = i
            time.sleep(delay)

        # Exemplo: começa em 180, final do movimento anterior, e vai até 120
        for i in range (endAngle, startAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)

if __name__ == "__main__":
    threads = []  # Lista para armazenar as threads

    #visao de frente
    #braco direito
    
    t1 = threading.Thread(target=moveBracoDireito, args=(bracoDireito, anguloBracoDireito, -1*variacao, velocidade))  # Crie uma thread para cada servo
    t2 = threading.Thread(target=moveOmbroDireito, args=(ombroDireito, anguloOmbroDireito, -2*variacao, velocidade))  # Crie uma thread para cada servo
    threads.append(t1)  # Adicione a thread à lista
    threads.append(t2)  # Adicione a thread à lista
        
    t1 = threading.Thread(target=moveOmbroBracoEsquerdo, args=(bracoEsquerdo, anguloBracoEsquerdo, variacao, velocidade))  # Crie uma thread para cada servo
    t2 = threading.Thread(target=moveOmbroBracoEsquerdo, args=(ombroEsquerdo, anguloOmbroEsquerdo, variacao, velocidade))  # Crie uma thread para cada servo
    threads.append(t1)  # Adicione a thread à lista
    threads.append(t2)  # Adicione a thread à lista
    
    #t1 = threading.Thread(target=moveMotor, args=(cabeca, anguloCabeca, variacao, velocidade))  # Crie uma thread para cada servo
    #threads.append(t1)  # Adicione a thread à lista
        
    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    print("Done!")



