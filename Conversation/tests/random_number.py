import random

def getRangeLesson(lesson):
    f = open(f"C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    n = 0
    for line in content:
        if lesson in line:
            i=n
        n = n+1
    return n

def getRangeCustom(custom):
    f = open(f"C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires/{custom}", "r")
    content = f.readlines()
    n = 0
    for line in content:
        if custom in line:
            i=n
        n = n+1
    return n

def getQuestion(lesson, i):
    f = open(f"C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    end = content[i].find(',')
    return content[i][0:end]

def getCustomQuestion(custom, i):
    f = open(f"C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires/{custom}", "r")
    content = f.readlines()
    end = content[i].find(',')
    return content[i][0:end]
    
def getCustomAnswer(custom, i):
    f = open(f"C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires/{custom}", "r")
    content = f.readlines()
    begin = content[i].find(',') + 1
    end = content[i].find('/n')
    return content[i][begin:end]

def assessment_mode(chapter,presence_use):
    global push_button_is_pressed
    outer_break = False
    for i in range(getRange(chapter)):
        run_expression('thoughtful')
        ttsCloud(getQuestion(chapter,i))
        skip_question = False
        push_button_is_pressed = False
        while True:
            if push_button_is_pressed:
                push_button_is_pressed = False
                frase = get_transcription_from_whisper("en")
                if frase is not None:
                    if getAnswer(chapter, i) in frase:
                        if(i < ((getRange(chapter))-1)):
                            run_expression('thoughtful')
                            ttsCloud("Acertou, vamos para a próxima pergunta")
                            error_count = 0
                            nota += 1
                            break
                        else:
                            run_expression('thoughtful')
                            ttsCloud("Você finalizou a atividade. Parabéns")
                            outer_break = True
                            break
                    else:
                        run_expression('thoughtful')
                        ttsCloud("Está errado tente outra vez")
                        push_button_is_pressed = False
                        error_count += 1
                        if error_count >=3:
                            run_expression('thoughtful')
                            ttsCloud("Parece que você está com dificuldades. Gostaria de pular essa questão?")
                            push_button_is_pressed = False
                            while True:
                                if push_button_is_pressed:
                                    push_button_is_pressed = False
                                    frase = get_transcription_from_whisper("pt")
                                    if frase is not None:
                                        if "sim" in frase:
                                            run_expression('thoughtful')
                                            ttsCloud("Tudo bem, vamos para a próxima pergunta")
                                            error_count = 0
                                            skip_question = True
                                            break
                                        if "não" in frase:
                                            run_expression('thoughtful')
                                            ttsCloud(getQuestion(chapter,i))
                                            break
                                        else:
                                            run_expression('thoughtful')
                                            ttsCloud("Não entendi. Me responda se você quer pular a questão com Sim ou Não.")
                    if skip_question:
                        skip_question = False
                        break
            

            if outer_break:
                break 
                                
def customlearning():
    chapter = "curiosidades" 
    qtde = getRangeCustom(chapter)
    if qtde <= 5:
        for i in range (qtde):
            run_expression('thoughtful')
            ttsCloud(getQuestion(chapter,i))
            skip_question = False
            push_button_is_pressed = False
            while True:
                if push_button_is_pressed:
                    push_button_is_pressed = False
                    frase = get_transcription_from_whisper("pt")
                    if frase is not None:
                        if getAnswer(chapter, i) in frase:
                            if(i < ((getRange(chapter))-1)):
                                run_expression('thoughtful')
                                ttsCloud("Acertou, vamos para a próxima pergunta")
                                error_count = 0
                                nota += 1
                                break
                            else:
                                run_expression('thoughtful')
                                ttsCloud("Você finalizou a atividade. Parabéns")
                                outer_break = True
                                break
                        else:
                            run_expression('thoughtful')
                            ttsCloud("Está errado tente outra vez")
                            push_button_is_pressed = False
                            error_count += 1
                            if error_count >=3:
                                run_expression('thoughtful')
                                ttsCloud("Parece que você está com dificuldades. Gostaria de pular essa questão?")
                                push_button_is_pressed = False
                                while True:
                                    if push_button_is_pressed:
                                        push_button_is_pressed = False
                                        frase = get_transcription_from_whisper("pt")
                                        if frase is not None:
                                            if "sim" in frase:
                                                run_expression('thoughtful')
                                                ttsCloud("Tudo bem, vamos para a próxima pergunta")
                                                error_count = 0
                                                skip_question = True
                                                break
                                            if "não" in frase:
                                                run_expression('thoughtful')
                                                ttsCloud(getQuestion(chapter,i))
                                                break
                                            else:
                                                run_expression('thoughtful')
                                                ttsCloud("Não entendi. Me responda se você quer pular a questão com Sim ou Não.")
                        if skip_question:
                            skip_question = False
                            break
                

                if outer_break:
                    break 
    else:
        List = [i for i in range(0, qtde)]
        UpdatedList = random.sample(List, k = 5)
        for j in UpdatedList:
            run_expression('thoughtful')
            ttsCloud(getQuestion(chapter,j))
            skip_question = False
            push_button_is_pressed = False
            while True:
                if push_button_is_pressed:
                    push_button_is_pressed = False
                    frase = get_transcription_from_whisper("pt")
                    if frase is not None:
                        if getAnswer(chapter, j) in frase:
                            if(j < ((getRange(chapter))-1)):
                                run_expression('thoughtful')
                                ttsCloud("Acertou, vamos para a próxima pergunta")
                                error_count = 0
                                nota += 1
                                break
                            else:
                                run_expression('thoughtful')
                                ttsCloud("Você finalizou a atividade. Parabéns")
                                outer_break = True
                                break
                        else:
                            run_expression('thoughtful')
                            ttsCloud("Está errado tente outra vez")
                            push_button_is_pressed = False
                            error_count += 1
                            if error_count >=3:
                                run_expression('thoughtful')
                                ttsCloud("Parece que você está com dificuldades. Gostaria de pular essa questão?")
                                push_button_is_pressed = False
                                while True:
                                    if push_button_is_pressed:
                                        push_button_is_pressed = False
                                        frase = get_transcription_from_whisper("pt")
                                        if frase is not None:
                                            if "sim" in frase:
                                                run_expression('thoughtful')
                                                ttsCloud("Tudo bem, vamos para a próxima pergunta")
                                                error_count = 0
                                                skip_question = True
                                                break
                                            if "não" in frase:
                                                run_expression('thoughtful')
                                                ttsCloud(getQuestion(chapter,i))
                                                break
                                            else:
                                                run_expression('thoughtful')
                                                ttsCloud("Não entendi. Me responda se você quer pular a questão com Sim ou Não.")
                        if skip_question:
                            skip_question = False
                            break
                

                if outer_break:
                    break 

chapter = "curiosidades"
print(getRangeCustom(chapter))


