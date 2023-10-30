from time import sleep
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)
i = 10
sleeping = 2
vezes = 90
while True:
    for i in range (0,3):
        kit.servo[0].angle = i*vezes
        kit.servo[1].angle = i*vezes
        kit.servo[2].angle = i*vezes
        kit.servo[3].angle = i*vezes
        kit.servo[4].angle = i*vezes
        kit.servo[5].angle = i*vezes
        print(i)
        
        sleep(sleeping)
        
    for i in range (2,-1,-1):
    
        kit.servo[0].angle = i*vezes
        kit.servo[1].angle = i*vezes
        kit.servo[2].angle = i*vezes
        kit.servo[3].angle = i*vezes
        kit.servo[4].angle = i*vezes
        kit.servo[5].angle = i*vezes
        print(i)
        sleep(sleeping)

