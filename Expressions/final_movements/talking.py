# The robot will move its arms, although not fully in sync with the speech, 
# focusing more on being an animation than a synchronous representation of the speech. 
# The robot's mouth on the LCD will move as something is being spoken and so will the arms.

from adafruit_servokit import ServoKit
import threading
import time

kit = ServoKit(channels=16)

bracoEsquerdo = 0 #Esquerda do ALIX
ombroEsquerdo = 1 #Esquerda do ALIX
bracoDireito = 2 #Direita do ALIX
ombroDireito = 3 #Direita do ALIX
cabeca = 4

anguloBracoEsquerdo = 140 
anguloOmbroEsquerdo = 180 
anguloBracoDireito = 30 
anguloOmbroDireito = 30
anguloCabeca = 90

variacao = 30
velocidade = 0.02
repeticoes = 2


kit.servo[bracoEsquerdo].angle = anguloBracoEsquerdo
kit.servo[ombroEsquerdo].angle = anguloOmbroEsquerdo

kit.servo[bracoDireito].angle = anguloBracoDireito
kit.servo[ombroDireito].angle = anguloOmbroDireito
kit.servo[cabeca].angle = anguloCabeca

def moveOmbroBracoDireito(servo, startAngle, variacaoAngle, delay):
    for j in range(repeticoes):
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

def moveBracoEsquerdo(servo, startAngle, variacaoAngle, delay):
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

def moveDireito(servo, startAngle, variacaoAngle, delay):
    for j in range(repeticoes):
        endAngle = startAngle + variacaoAngle
        for i in range (startAngle, endAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)
        
        for i in range (endAngle, startAngle, -1):
            kit.servo[servo].angle = i
            time.sleep(delay)

def moveOmbroEsquerdo(servo, startAngle, variacaoAngle, delay):
    for j in range(repeticoes):
        endAngle = startAngle + variacaoAngle #120
        for i in range (startAngle, endAngle, -1):
            kit.servo[servo].angle = i
            time.sleep(delay)

        # Exemplo: comeca em 180, final do movimento anterior, e vai ate 120
        for i in range (endAngle, startAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)

if __name__ == "__main__":
    threads = []  # Lista para armazenar as threads

    # lado esquerdo 
    t1 = threading.Thread(target=moveBracoEsquerdo, args=(bracoEsquerdo, anguloBracoEsquerdo, -1*variacao, velocidade))  # Crie uma thread para cada servo
    t2 = threading.Thread(target=moveOmbroEsquerdo, args=(ombroEsquerdo, anguloOmbroEsquerdo, -2*variacao, velocidade))  # Crie uma thread para cada servo
    threads.append(t1)  # Adicione a thread a lista
    threads.append(t2)  # Adicione a thread a lista

    # lado direito    
    t1 = threading.Thread(target=moveOmbroBracoDireito, args=(bracoDireito, anguloBracoDireito, variacao, velocidade))  # Crie uma thread para cada servo
    t2 = threading.Thread(target=moveOmbroBracoDireito, args=(ombroDireito, anguloOmbroDireito, variacao, velocidade))  # Crie uma thread para cada servo
    threads.append(t1)  # Adicione a thread a lista
    threads.append(t2)  # Adicione a thread a lista
    
    #t1 = threading.Thread(target=moveMotor, args=(cabeca, anguloCabeca, variacao, velocidade))  # Crie uma thread para cada servo
    #threads.append(t1)  # Adicione a thread a lista
        
    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    print("Done!")
