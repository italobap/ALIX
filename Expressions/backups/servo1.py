import RPi.GPIO as gpio
from time import sleep

gpio.setmode(gpio.BOARD)
gpio.setup(12,gpio.OUT)

pwmServo = gpio.PWM(12,50)
pwmServo.start(0)

while True:
    for i in range (0,181):
        pwmServo.ChangeDutyCycle((i/18)+2)
        print(i)
        sleep(0.05)

    for i in range (180,-1,-1):
        pwmServo.ChangeDutyCycle((i/18)+2)
        print(i)
        sleep(0.05)

gpio.cleanup()

