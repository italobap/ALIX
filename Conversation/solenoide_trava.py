import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from time import sleep

pino = 15

GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(pino, GPIO.OUT) # Set pin 10 to be an input pin and set initial value to be pulled low (off)

#GPIO.output(pino, True)
GPIO.output(pino, False)
print("Solenoide Travada!")

#while True:
##    GPIO.output(pino, True)
 ##   print("Solenoid!")
   # sleep(3)
    #GPIO.output(pino, False)
    #print("Solenoid!")
    #sleep(3)

