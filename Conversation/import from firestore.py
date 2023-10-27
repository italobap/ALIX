import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json

# Setup
cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred)

db=firestore.client()

# Read data
# Get a document with known id
'''result = db.collection('Lesson').document("sfNxpdM3YZfQ6tbbzJS4").get()
if result.exists:
    print(result.to_dict())'''
data = {}

docs = result = db.collection('Lesson').get()
for doc in docs:
    data[doc.id] = doc.to_dict()

json_file = 'firestore_data.json'

with open(json_file, 'w') as file:

    json.dump(data, file, indent=4)

print(f"Data has been exported to {json_file}")