import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from gpiozero import Button
import time 
from time import sleep

solenoid_pin = 15
push_button_pin = 19 #gpio10
magnetic_sensor_pin = 32 #gpio12
limit_time = 5
GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(solenoid_pin, GPIO.OUT) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
GPIO.setup(magnetic_sensor_pin, GPIO.IN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
GPIO.setup(push_button_pin, GPIO.IN, pull_up_down = GPIO.PUD_UP) # Set pin 10 to be an input pin and set initial value to be pulled low (off)


while True:
    if GPIO.input(push_button_pin) == GPIO.LOW:
        print("Button is pressed")
        GPIO.output(solenoid_pin, 1)
        while (GPIO.input(magnetic_sensor_pin) == GPIO.LOW):
            GPIO.output(solenoid_pin, 1)
        
        sleep(1)
        GPIO.output(solenoid_pin, 0)
        current_time = time.time()
        break

while time.time() - current_time > limit_time:
    if (GPIO.input(magnetic_sensor_pin) == GPIO.HIGH):
        print("Trava fechada")
    else:
        print("Trava aberta")
