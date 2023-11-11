import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


codificacao = 'UTF-8'
if not os.path.exists("/home/alix/Desktop/Questionnaires"):      
    os.makedirs("/home/alix/Desktop/Questionnaires") 

# if not os.path.exists("C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires"):      
#    os.makedirs("C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires")

cred = credentials.Certificate("credentials.json")

firebase_admin.initialize_app(cred)

db = firestore.client()

query = db.collection("Lesson").where("subject","==","english").stream()

f = open('Lessons', 'w')

nameLesson = ''
for lesson in query:
    nameLesson = str(lesson.get('name')).lower()
    #print(f"{lesson.id} => {lesson.to_dict()}")
    print(f"{nameLesson}")
    
    with open('Lessons', 'a') as f:
        f.write(f"{nameLesson},{lesson.get('ListeningSection').get('audio')}")
        f.write('\n')
    
    f=open(f"Questionnaires/{nameLesson}" ,'w', encoding=codificacao)
    
    
    questionnaire=lesson.get("AssesmentSection").get("Questionnaire")
    for question in questionnaire:
        questionName = str(question.get('question')).lower()
        expectedAnswer = str(question.get('expectedAnswer')).lower()
        print(f"{questionName},{expectedAnswer}")
        with open(f"Questionnaires/{nameLesson}" ,'a', encoding=codificacao) as f:
            f.write(f"{questionName},{expectedAnswer}")
            f.write('\n')
    

