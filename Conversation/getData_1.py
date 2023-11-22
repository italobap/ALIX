def getQuestion(lesson, i):
    f = open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    end = content[i].find(',')
    return content[i][0:end]
  
def getAnswer(lesson, i):
    f = open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    begin = content[i].find(',') + 1
    end = content[i].find('/n')
    return content[i][begin:end]

def getLesson(i):
    f = open("/home/alix/Documents/ALIX/ALIX/Conversation/Lessons", "r")
    content = f.readlines()
    return content[i][0:content[i].find(',')]

def getRange(lesson):
    f = open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{lesson}", "r")
    content = f.readlines()
    n = 0
    for line in content:
        if lesson in line:
            i=n
        n = n+1
    return n

def getCustomQuestion(custom, i):
    f = open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{custom}", "r")
    content = f.readlines()
    end = content[i].find(',')
    return content[i][0:end]
    
def getCustomAnswer(custom, i):
    f = open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{custom}", "r")
    content = f.readlines()
    begin = content[i].find(',') + 1
    end = content[i].find('/n')
    return content[i][begin:end]
    
def getCustoms(i):
    f = open("/home/alix/Documents/ALIX/ALIX/Conversation/Customs", "r")
    content = f.readlines()
    return content[i][0:content[i].find('/n')]

notfind = True
resposta = 'saudações'
for j in range(10):
    if resposta in getLesson(j):
        print("deu certo")
        deucerto = False
        
if deucerto:
    print("deu ruim")


