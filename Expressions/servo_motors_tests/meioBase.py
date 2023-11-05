from adafruit_servokit import ServoKit

# the robot will raise its arms and display a happy face.
kit = ServoKit(channels=16)

kit.servo[7].angle = 90
