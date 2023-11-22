import random

def getRange(lesson):
    f = open(f"C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    n = 0
    for line in content:
        if lesson in line:
            i=n
        n = n+1
    return n

def getQuestion(lesson, i):
    f = open(f"C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    end = content[i].find(',')
    return content[i][0:end]

def customquestion(chapter):
    qtde = getRange(chapter)
    if qtde <= 5:
        for i in range (qtde):
            print(getQuestion(chapter,i))
    else:
        List = [i for i in range(0, qtde)]
        UpdatedList = random.sample(List, k = 5)
        
        for j in UpdatedList:
            print(getQuestion(chapter,j))

chapter = "números"
customquestion(chapter)


