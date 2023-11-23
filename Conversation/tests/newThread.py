def thread_expression():
    internal_expression = expression
    with RaspberryPi() as ipr:#rpi:
        display = Display(ipr)
        display.initialize(color_mode=444, bounds=(36,48,204,272))
        while True:
            for i in range(numero_maximo_imagens):
                # Se a expressÃ£o mudar, reinicia o loop de imagens da pasta
                if internal_expression != expression:
                    i = 0
                    internal_expression = expression
                    
                frame = Image.open(address_expression+expression+'/frame'+str(i)+'.png')
                data = list(frame.convert('RGB').getdata())
                display.draw_rgb_bytes(data)
                time.sleep(freq_mudanca_de_imagem)