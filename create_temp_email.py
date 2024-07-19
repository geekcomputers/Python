import requests


temp_email_provider = "https://api.guerrillamail.com"


def get_email_address():
    try:
        response = requests.post(
            f'{temp_email_provider}/ajax.php?f=get_email_address',
            json={'ip': '127.0.0.1', 'agent': 'Mozilla/5.0'}
        )

        if response.status_code == 200:
            data = response.json()
            return {
                'email': data['email_addr'],
                'sidToken': data['sid_token'],
            }
    except Exception as error:
        print('Error getting email address:', error)

    return None


def check_email(sid_token):
    try:
        response = requests.get(
            f'{temp_email_provider}/ajax.php?f=check_email&seq=0&sid_token={sid_token}'
        )
        if response.status_code == 200:
            data = response.json()
            return data['list']
    except Exception as error:
        print('Error checking email:', error)

    return []


if __name__ == "__main__":
    email_data = get_email_address()
    if email_data:
        print(f"Email: {email_data['email']}, SID Token: {email_data['sidToken']}")
        print(f"Web inbox: https://www.guerrillamail.com/inbox?sid_token={email_data['sidToken']}")
        email_list = check_email(email_data['sidToken'])
        print(f"Email list length: {len(email_list)}")
        assert len(email_list) == 1
