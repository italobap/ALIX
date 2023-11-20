import threading
import time
from PIL import Image
from st7789v.interface import RaspberryPi
from st7789v import Display

expression = 'sad'
def thread_expression():
    internal_expression = expression
    with RaspberryPi() as rpi:
        display = Display(rpi)
        display.initialize(color_mode=666)
        while True:
            for i in range(45):
                if internal_expression != expression:
                    i = 0
                    internal_expression = expression

                frame = Image.open('/home/alix/Downloads/'+expression+'/frame'+str(i)+'.png')
                data = list(frame.convert('RGB').getdata())
                display.draw_rgb_bytes(data)
                time.sleep(0.1)

#----------------------------Main function----------------------------
if __name__ == "__main__":
    # global expression
    t = threading.Thread(target=thread_expression)
    t.start()

    while True:
        time.sleep(10)
        expression = 'happy'
        time.sleep(10)
        expression = 'sad'
