from time import sleep
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)
i = 10
sleeping = 1
vezes = 10
braco1 = 150
braco2 = 15
cabeca=90
while True:
    for i in range (0, 40, 10):
        kit.servo[0].angle = braco1+i
        kit.servo[1].angle = braco1+i
        kit.servo[2].angle = braco2+i
        kit.servo[3].angle = braco2+i
        #kit.servo[4].angle = i*(vezes)
        #kit.servo[3].angle = i*(vezes)
        kit.servo[4].angle = cabeca+i
        kit.servo[8].angle = cabeca+i
        print(i*vezes)
        
        sleep(sleeping)
        
    for i in range (20,-10,-10):
    
        kit.servo[0].angle = braco1+i
        kit.servo[1].angle = braco1+i
        kit.servo[2].angle = braco2+i
        kit.servo[3].angle = braco2+i
        kit.servo[4].angle = cabeca+i
        kit.servo[8].angle = cabeca+i
        #kit.servo[4].angle = i*(vezes)
        #kit.servo[3].angle = i*(vezes)
        print(i*vezes)
        sleep(sleeping)
