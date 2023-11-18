import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

if not os.path.exists("/home/alix/Desktop/Questionnaires"):      
    os.makedirs("/home/alix/Desktop/Questionnaires") 

cred = credentials.Certificate("/home/alix/Desktop/cred.json")

firebase_admin.initialize_app(cred)

db = firestore.client()

query = db.collection("Lesson").where("subject","==","english").stream()

f = open('/home/alix/Desktop/Lessons', 'w')

for lesson in query:
    #print(f"{lesson.id} => {lesson.to_dict()}")
    print(f"{lesson.get('name')}")
    
    with open('/home/alix/Desktop/Lessons', 'a') as f:
        f.write(f"{lesson.get('name')},{lesson.get('ListeningSection').get('audio')}")
        f.write('\n')
    
    f=open(f"/home/alix/Desktop/Questionnaires/{lesson.get('name')}" ,'w')
    
    
    questionnaire=lesson.get("AssesmentSection").get("Questionnaire")
    for question in questionnaire:
        print(f"{question.get('question')},{question.get('expectedAnswer')}")
        with open(f"/home/alix/Desktop/Questionnaires/{lesson.get('name')}" ,'a') as f:
            f.write(f"{question.get('question')},{question.get('expectedAnswer')}")
            f.write('\n')

query = db.collection("CustomQuestionnaire").stream()
f = open('/home/alix/Desktop/Customs', 'w')
for custom in query:
    #print(f"{lesson.id} => {lesson.to_dict()}")
    print(f"{custom.get('name')}")
    
    with open('/home/alix/Desktop/Customs', 'a') as f:
        f.write(f"{custom.get('name')}")
        f.write('\n')
    
    f=open(f"/home/alix/Desktop/Questionnaires/{custom.get('name')}" ,'w')
    
    
    questionnaire=custom.get("Questionnaire")
    for question in questionnaire:
        print(f"{question.get('question')},{question.get('answer')}")
        with open(f"/home/alix/Desktop/Questionnaires/{custom.get('name')}" ,'a') as f:
            f.write(f"{question.get('question')},{question.get('answer')}")
            f.write('\n')

