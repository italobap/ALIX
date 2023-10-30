from adafruit_servokit import ServoKit
import time

# the robot will raise its arms and display a happy face.
kit = ServoKit(channels=16)

bracoDireito = 0
ombroDireito = 1
bracoEsquerdo = 2
ombroEsquerdo = 3
cabeca = 4
velocidade = 0.01

anguloBracoDireito = 155
anguloOmbroDireito = 180

anguloBracoEsquerdo = 30
anguloOmbroEsquerdo = 30
anguloCabeca = 90

kit.servo[bracoDireito].angle = anguloBracoDireito
kit.servo[ombroDireito].angle = anguloOmbroDireito

kit.servo[bracoEsquerdo].angle = anguloBracoEsquerdo
kit.servo[ombroEsquerdo].angle = anguloOmbroEsquerdo
kit.servo[cabeca].angle = anguloCabeca


