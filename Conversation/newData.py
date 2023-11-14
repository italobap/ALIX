import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime 
import time 

################# ESTABELECENDO CONEXÃO COM BD ##########################

cred = credentials.Certificate("cred.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

################# CONEXÃO ESTABELECIDA ##############################

def addAbsence():
    absenceData = {"notified": False, "timeOfOccurence": datetime.now()}
    db.collection("Absences").add(absenceData)
    print(f"Added Absence")
    
def addResults(duration, grade, lesson):
    resultsData = {"date": datetime.now() , "duration": duration, "grade": grade, "lesson": lesson}
    db.collection("Results").add(resultsData)
    print(f"Added Results")
  
addAbsence()
addResults(30, 8.2, "Adjetivos")
