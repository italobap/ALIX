import time
import keyboard
from time import sleep

absence_time = 100
short_pomodoro = 5 #ciclo de 25 min#
global a_time
global spomodoro_time 
chapter = "Cores"

def getLesson(i):
    f = open("Lessons", "r")
    content = f.readlines()
    return content[i][0:content[i].find(',')]

def getQuestion(lesson, i):
    f = open(f"Questionnaires/{lesson}", "r")
    content = f.readlines()
    end = content[i].find(',')
    return content[i][0:end]

def getAnswer(lesson, i):
    f = open(f"Questionnaires/{lesson}", "r")
    content = f.readlines()
    begin = content[i].find(',') + 1
    end = content[i].find('/n')
    return content[i][begin:end]

def getRange(lesson):
    f = open(f"Questionnaires/{lesson}", "r")
    content = f.readlines()
    n = 0
    for line in content:
        if lesson in line:
            i=n
        n = n+1
    return n

def assessment_mode(chapter):
    print("Vamos praticar a avaliação de " + chapter + "?")
    outer_break = False
    while True:
        if keyboard.read_key() == "r":
                frase= input("Escreva sua resposta: ")
                if frase is not None:
                    frase_lower = frase.lower()
                    if "sim" in frase_lower:
                        print("Vamos começar.")
                        error_count = 0
                        nota = 0
                        for i in range(getRange(chapter)):
                            print(getQuestion(chapter,i))
                            skip_question = False
                            while True:
                                if keyboard.read_key() == "r":
                                    frase= input("Escreva sua resposta: ")
                                    if frase is not None:
                                        if getAnswer(chapter, i).lower() in frase:
                                            if(i < ((getRange(chapter))-1)):
                                                print("Acertou, vamos para a próxima pergunta")
                                                error_count = 0
                                                nota += 1
                                                print("Notal atual ="  +  str(nota))
                                                break
                                            else:
                                                print("Você finalizou a atividade. Parabéns")
                                                nota += 1
                                                print("Notal atual ="  +  str(nota))
                                                media = (nota/getRange(chapter)) * 10
                                                #lockable_compartment()
                                                print(media)
                                                outer_break = True
                                                break
                                        else:
                                            print("Está errado tente outra vez")
                                            error_count += 1
                                            if error_count >=3:
                                                print("Parece que você está com dificuldades. Gostaria de pular essa questão?")
                                                while True:
                                                    if keyboard.read_key() == "r":
                                                        frase = input("Escreva sua resposta: ")
                                                        if frase is not None:
                                                            if "sim" in frase:
                                                                print("Tudo bem, vamos para a próxima pergunta")
                                                                error_count = 0
                                                                skip_question = True
                                                                break
                                                            if "não" in frase:
                                                                print(getQuestion(chapter,i))
                                                                break
                                        if skip_question:
                                            skip_question = False
                                            break
                                if outer_break:
                                    break 

                    elif "não" in frase:
                        print("Certo, finalizando modo de estudo.")
                        break
                    else:
                        print("Não entendi o que você disse. Me responda Sim ou Não para fazer atividade de avaliação.")
        if outer_break:
            break  # This break will exit the outer while loop


if __name__ == '__main__':
    assessment_mode(chapter)
