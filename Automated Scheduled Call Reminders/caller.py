"""
Firebase-Twilio Automated Reminder System

This script automates reminder calls to individuals stored in a Firebase Cloud Firestore database.
It checks for entries every hour and initiates calls 5 minutes prior to the scheduled time for each entry.
"""

import datetime
from typing import List, Dict, Any
from time import gmtime, strftime
from firebase_admin import credentials, firestore, initialize_app
from twilio.rest import Client

# Twilio credentials (Replace with your actual credentials)
ACC_SID: str = ""
AUTH_TOKEN: str = ""
TWILIO_PHONE_NUMBER: str = "add your twilio number"

# Firebase credentials (key.json should be your Firebase project certificate)
FIREBASE_CERT_PATH: str = "key.json"

# Initialize Firebase and Twilio clients
cred = credentials.Certificate(FIREBASE_CERT_PATH)
default_app = initialize_app(cred)
db = firestore.client()
database_reference = db.collection("on_call")
twilio_client = Client(ACC_SID, AUTH_TOKEN)


def search() -> None:
    """
    Search for scheduled calls in the database and initiate reminders 5 minutes prior to the scheduled time.
    
    This function:
    1. Queries the Firebase database for entries with the current date
    2. Filters entries scheduled within the next hour
    3. Initiates Twilio calls for entries where the scheduled time is 5 minutes from now
    """
    # Current time and cutoff time (1 hour from now)
    current_time: datetime.datetime = datetime.datetime.now()
    one_hour_later: str = (current_time + datetime.timedelta(hours=1)).strftime("%H:%M:%S")
    current_date: str = str(strftime("%d-%m-%Y", gmtime()))
    
    # Fetch documents from Firestore
    docs = db.collection(u"on_call").where(u"date", u"==", current_date).stream()
    scheduled_calls: List[Dict[str, Any]] = []
    
    # Filter documents scheduled within the next hour
    for doc in docs:
        doc_data = doc.to_dict()
        if current_time.strftime("%H:%M:%S") <= doc_data["from"] <= one_hour_later:
            scheduled_calls.append(doc_data)
    
    print(f"Found {len(scheduled_calls)} scheduled calls for {current_date} within the next hour")
    
    # Process each scheduled call to check if it's time to send a reminder
    while scheduled_calls:
        current_timestamp: str = datetime.datetime.now().strftime("%H:%M")
        five_minutes_later: str = (datetime.datetime.now() + datetime.timedelta(minutes=5)).strftime("%H:%M")
        
        for call in scheduled_calls[:]:  # Iterate over a copy to safely remove elements
            scheduled_time = call["from"][0:5]  # Extract HH:MM from HH:MM:SS
            
            if scheduled_time == five_minutes_later:
                phone_number: str = call["phone"]
                
                try:
                    # Initiate Twilio call
                    twilio_client.calls.create(
                        to=phone_number,
                        from_=TWILIO_PHONE_NUMBER,
                        url="http://demo.twilio.com/docs/voice.xml",
                    )
                    print(f"Call initiated to {phone_number} for scheduled time {scheduled_time}")
                    scheduled_calls.remove(call)
                except Exception as e:
                    print(f"Error initiating call to {phone_number}: {str(e)}")


if __name__ == "__main__":
    # Run the search function to check for and initiate calls
    search()