import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


#if not os.path.exists("/home/alix/Desktop/Questionnaires"):      
#    os.makedirs("/home/alix/Desktop/Questionnaires") 

if not os.path.exists("C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires"):      
    os.makedirs("C:/Users/italo/Documents/UTFPR/2023-2/Oficinas 3/Código/ALIX/Conversation/Questionnaires")

cred = credentials.Certificate("credentials.json")

firebase_admin.initialize_app(cred)

db = firestore.client()

query = db.collection("Lesson").where("subject","==","english").stream()

f = open('Lessons', 'w')

for lesson in query:
    #print(f"{lesson.id} => {lesson.to_dict()}")
    print(f"{lesson.get('name')}")
    
    with open('Lessons', 'a') as f:
        f.write(f"{lesson.get('name')},{lesson.get('ListeningSection').get('audio')}")
        f.write('\n')
    
    f=open(f"Questionnaires/{lesson.get('name')}" ,'w', encoding='Windows-1252')
    
    
    questionnaire=lesson.get("AssesmentSection").get("Questionnaire")
    for question in questionnaire:
        print(f"{question.get('question')},{question.get('expectedAnswer')}")
        with open(f"Questionnaires/{lesson.get('name')}" ,'a', encoding='Windows-1252') as f:
            f.write(f"{question.get('question')},{question.get('expectedAnswer')}")
            f.write('\n')
    

