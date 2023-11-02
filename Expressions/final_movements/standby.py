# The robot will keep its arms lowered and display a neutral face on the screen.

from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)

bracoEsquerdo = 0 #Esquerda do ALIX
ombroEsquerdo = 1 #Esquerda do ALIX
bracoDireito = 2 #Direita do ALIX
ombroDireito = 3 #Direita do ALIX
cabeca = 4

anguloBracoEsquerdo = 155
anguloOmbroEsquerdo = 180
anguloBracoDireito = 30
anguloOmbroDireito = 30
anguloCabeca = 90

velocidade = 0.01

kit.servo[bracoEsquerdo].angle = anguloBracoEsquerdo
kit.servo[ombroEsquerdo].angle = anguloOmbroEsquerdo

kit.servo[bracoDireito].angle = anguloBracoDireito
kit.servo[ombroDireito].angle = anguloOmbroDireito
kit.servo[cabeca].angle = anguloCabeca
