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

def getLesson(i):
    f = open("Lessons", "r")
    content = f.readlines()
    return content[i][0:content[i].find(',')]

def getAudio(lesson):
    f = open("Lessons", "r")
    content = f.readlines()
    n = 0
    for line in content:
        if lesson in line:
            i=n
        n = n+1
    begin = content[i].find(',') + 1
    end = content[i].find('/n')
    return content[i][begin:end]

#print(getAudio("Colors"))

#print( getLesson(2))
print(getQuestion("Colors",1))
print(getAnswer("Colors",1))
