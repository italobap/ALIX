import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime 
import time 

################# ESTABELECENDO CONEXÃO COM BD ##########################

cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

################# CONEXÃO ESTABELECIDA ##############################

def addAbsence(timeDate):
    absenceData = {"notified": False, "timeOfOccurence": timeDate}
    db.collection("Absences").add(absenceData)
    print(f"Added Absence")
    
def addResults(duration, grade, lesson):
    docs = (db.collection("Lesson").where("name", "==", lesson).stream())
    for doc in docs:
        db.collection("Lesson").document(doc.id).update({"duration":duration})
        db.collection("Lesson").document(doc.id).update({"grade":grade})
    print(f"Added Results")
  
def addAbsenceOld():
    absenceData = {"notified": False, "timeOfOccurence": datetime.now()}
    db.collection("Absences").add(absenceData)
    print(f"Added Absence")

#addResults(30, 8.2, "Saudações")
timestamp = time.time()
date_time = datetime.fromtimestamp(timestamp)
print(date_time)
addAbsence(date_time)



