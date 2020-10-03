#The project automates calls for people from the firebase cloud database and the schedular keeps it running and checks for entries
#every 1 hour using aps scedular
#The project can be used to set 5 min before reminder calls to a set of people for doing a particular job
import os
from firebase_admin import credentials, firestore, initialize_app
from datetime import datetime,timedelta
import time
from time import gmtime, strftime
import twilio
from twilio.rest import Client
#twilio credentials
acc_sid=""
auth_token=""
client=Client(acc_sid, auth_token)

#firebase credentials
#key.json is your certificate of firebase project
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
database_reference = db.collection('on_call')

#Here the collection name is on_call which has documents with fields phone , from (%H:%M:%S time to call the person),date 

#gets data from cloud database and calls 5 min prior the time (from time) alloted in the database
def search():

    calling_time = datetime.now()
    one_hours_from_now = (calling_time + timedelta(hours=1)).strftime('%H:%M:%S')  
    current_date=str(strftime("%d-%m-%Y", gmtime()))
    docs = db.collection(u'on_call').where(u'date',u'==',current_date).stream()
    list_of_docs=[]
    for doc in docs:
        
        c=doc.to_dict()
        if (calling_time).strftime('%H:%M:%S')<=c['from']<=one_hours_from_now:
            list_of_docs.append(c)
    print(list_of_docs)

    while(list_of_docs):
        timestamp=datetime.now().strftime('%H:%M')
        five_minutes_prior= (timestamp + timedelta(minutes=5)).strftime('%H:%M')
        for doc in list_of_docs:
            if doc['from'][0:5]==five_minutes_prior:
                phone_number= doc['phone']
                call = client.calls.create(
                to=phone_number,
                from_="add your twilio number",
                url="http://demo.twilio.com/docs/voice.xml"
                )
                list_of_docs.remove(doc)


