from adafruit_servokit import ServoKit

# the robot will raise its arms and display a happy face.
kit = ServoKit(channels=16)

op2 = 180-60
op1 = 180
kit.servo[0].angle = op1
#kit.servo[1].angle = op1
#kit.servo[2].angle = 0
#kit.servo[3].angle = op2
#kit.servo[4].angle = 10