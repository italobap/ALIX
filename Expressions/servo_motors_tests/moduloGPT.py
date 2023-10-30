from adafruit_servokit import ServoKit
import threading
import time

kit = ServoKit(channels=16)

def moveMotor(servo, startAngle, variacaoAngle, delay):
    while True:
        
        # Exemplo: começa em startAngle(150) vai movimentar até 180 (startAngle + variacaoAngle (30))
        endAngle = startAngle + variacaoAngle
        for i in range (startAngle, endAngle):
            kit.servo[servo].angle = i
            time.sleep(delay)
            #print(i, delay)

        # Exemplo: começa em 180, final do movimento anterior, e vai até 120
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

    #visao de frente
    #braco direito
    for i in range(2):
        t = threading.Thread(target=moveMotor, args=(i, 150, 30, 0.01))  # Crie uma thread para cada servo
        threads.append(t)  # Adicione a thread à lista
        
    #braco esquerdo
    for i in range(2,4):
        t = threading.Thread(target=moveMotor, args=(i, 31, 30, 0.01))  # Crie uma thread para cada servo
        threads.append(t)  # Adicione a thread à lista
    
    t1 = threading.Thread(target=moveMotor, args=(4, 110, 30, 0.01))  # Crie uma thread para cada servo
    threads.append(t1)  # Adicione a thread à lista
        
    for t in threads:
        t.start()  # Inicie todas as threads

    for t in threads:
        t.join()  # Aguarde todas as threads terminarem

    print("Done!")
