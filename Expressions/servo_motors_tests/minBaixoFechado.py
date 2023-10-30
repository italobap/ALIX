from adafruit_servokit import ServoKit

# the robot will raise its arms and display a happy face.
kit = ServoKit(channels=16)

kit.servo[0].angle = 180
kit.servo[1].angle = 180
kit.servo[2].angle = 0
kit.servo[3].angle = 0
