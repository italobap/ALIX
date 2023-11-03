# The robot move both arms up and down and rotate its head sideways while displaying a happy face.

from adafruit_servokit import ServoKit
import threading
import time

kit = ServoKit(channels=16)

bracoEsquerdo = 0 #Esquerda do ALIX
ombroEsquerdo = 1 #Esquerda do ALIX
bracoDireito = 2 #Direita do ALIX
ombroDireito = 3 #Direita do ALIX
cabeca = 4

repeticoes = 2
velocidade = 0.01
variacao = 30

for i in range (140, 150):
    kit.servo[bracoEsquerdo].angle = i
    time.sleep(velocidade)
        
for i in range (180, 150, -1):
    kit.servo[ombroEsquerdo].angle = i
    time.sleep(velocidade)
    
def moveMotor(servo, startAngle, variacaoAngle, delay):
    for j in range(repeticoes):
        
        # Exemplo: comeca em startAngle(150) vai movimentar ate 180 (startAngle + variacaoAngle (30))
        endAngle = startAngle + variacaoAngle
        for i in range (startAngle, endAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)
            #print(i, delay)

        # Exemplo: comeca em 180, final do movimento anterior, e vai ate 120
        for i in range (endAngle, startAngle - variacaoAngle, -1):
            kit.servo[servo].angle = i
            time.sleep(delay)
            #print(i, delay)
        
        for i in range (startAngle - variacaoAngle, startAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)
            #print(i, delay)

if __name__ == "__main__":
    threads = []  # Lista para armazenar as threads

    #braco Esquerdo
    for i in range(2):
        t = threading.Thread(target=moveMotor, args=(i, 150, variacao, velocidade))  # Crie uma thread para cada servo
        threads.append(t)  # Adicione a thread a lista
        
    #braco Direito
    for i in range(2,4):
        t = threading.Thread(target=moveMotor, args=(i, 31, variacao, velocidade))  # Crie uma thread para cada servo
        threads.append(t)  # Adicione a thread a lista
    
    t1 = threading.Thread(target=moveMotor, args=(4, 90, variacao, velocidade))  # Crie uma thread para cada servo
    threads.append(t1)  # Adicione a thread a lista
        
    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    for i in range (150, 140, -1):
        kit.servo[bracoEsquerdo].angle = i
        time.sleep(velocidade)
        
    for i in range (150, 180):
        kit.servo[ombroEsquerdo].angle = i
        time.sleep(velocidade)

    kit.servo[bracoDireito].angle = 30
    kit.servo[ombroDireito].angle = 30
    kit.servo[cabeca].angle = 90
    print("Done!")
