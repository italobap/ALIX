from gpiozero import Button

#button = Button(10)
button = Button(12)


push_button = Button(10)
magnetic_sensor = Button(12)
solenoid_pin = 15

while True:
    if push_button.is_pressed:
        print("Button is pressed")
    else:
        print("Button is not pressed")
