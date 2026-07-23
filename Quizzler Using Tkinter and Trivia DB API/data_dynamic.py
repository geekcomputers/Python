
'''
This file is responsible for fetching quiz questions from the Open Trivia Database API.
'''

import requests

parameters = {
    "amount": 10,
    "type": "multiple",
    "category": 18
}

error_message = ""

try:
    response = requests.get(url="https://opentdb.com/api.php", params=parameters, timeout=10)
    response.raise_for_status()  # Raise an exception for HTTP errors
    question_data = response.json()["results"]
    print("Questions loaded successfully from the API.")
except requests.exceptions.ConnectionError:
    error_message = "Network connection is poor. Please check your internet connection."
    question_data = []
except requests.exceptions.Timeout:
    error_message = "Request timed out. Internet speed might be too slow."
    question_data = []
except requests.exceptions.RequestException as e:
    error_message = f"An error occurred: {e}"
    question_data = []
