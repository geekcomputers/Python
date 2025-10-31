import json

new_data = {
        website_input.get():{
            "email": email_input.get(),
            "password": passw_input.get()
        }
    }

try:
    with open("data.json", "r") as data_file:
        data = json.load(data_file)
except FileNotFoundError:
    with open("data.json", "w") as data_file:
        pass
else:
    with open("data.json", "w") as data_file:
        json.dump(new_data, data_file, indent = 4)