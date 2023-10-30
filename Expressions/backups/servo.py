import RPi.GPIO as gpio
from time import sleep

gpio.setmode(gpio.BOARD)
gpio.setup(12,gpio.OUT)

pwmServo = gpio.PWM(12,50)
pwmServo.start(0)

while True:
    pwmServo.ChangeDutyCycle(7) #(angulo/18)+2
    print(90)
    sleep(2)

    pwmServo.ChangeDutyCycle(12)
    print(180)
    sleep(2)

    pwmServo.ChangeDutyCycle(7)
    print(90)
    sleep(2)

    pwmServo.ChangeDutyCycle(2)
    print(0)
    sleep(2)

gpio.cleanup()
