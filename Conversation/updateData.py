import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


codificacao = 'UTF-8'
if not os.path.exists("/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires"):      
    os.makedirs("/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires") 

# if not os.path.exists("C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires"):      
#    os.makedirs("C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires")

cred = credentials.Certificate("/home/alix/Documents/ALIX/ALIX/Conversation/credentials.json")

firebase_admin.initialize_app(cred)

db = firestore.client()

query = db.collection("Lesson").where("subject","==","english").stream()
f = open('/home/alix/Documents/ALIX/ALIX/Conversation/Lessons', 'w')

nameLesson = ''
for lesson in query:
    nameLesson = str(lesson.get('name')).lower()
    #print(f"{lesson.id} => {lesson.to_dict()}")
    print(f"{nameLesson}")
    
    with open('/home/alix/Documents/ALIX/ALIX/Conversation/Lessons', 'a') as f:
        f.write(f"{nameLesson},{lesson.get('ListeningSection').get('audio')}")
        f.write('\n')
    
    f=open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{nameLesson}" ,'w', encoding=codificacao)
    
    
    questionnaire=lesson.get("AssesmentSection").get("Questionnaire")
    for question in questionnaire:
        questionName = str(question.get('question')).lower()
        expectedAnswer = str(question.get('expectedAnswer')).lower()
        print(f"{questionName},{expectedAnswer}")
        with open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{nameLesson}" ,'a', encoding=codificacao) as f:
            f.write(f"{questionName},{expectedAnswer}")
            f.write('\n')
    
nameLesson = ''    
query = db.collection("CustomQuestionnaire").stream()
f = open('/home/alix/Documents/ALIX/ALIX/Conversation/Customs', 'w')
for custom in query:
    nameLesson = str(lesson.get('name')).lower()
    #print(f"{lesson.id} => {lesson.to_dict()}")
    print(f"{nameLesson}")
    
    with open('/home/alix/Documents/ALIX/ALIX/Conversation/Customs', 'a') as f:
        f.write(f"{nameLesson}")
        f.write('\n')
    
    f=open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{nameLesson}" ,'w', encoding=codificacao)
    
    questionnaire=custom.get("Questionnaire")
    for question in questionnaire:
        questionName = str(question.get('question')).lower()
        expectedAnswer = str(question.get('expectedAnswer')).lower()
        print(f"{question.get('question')},{question.get('answer')}")
        with open(f"/home/alix/Documents/ALIX/ALIX/Conversation/Questionnaires/{custom.get('name')}" ,'a', encoding=codificacao) as f:
            f.write(f"{questionName},{expectedAnswer}")
            f.write('\n')


