from __future__ import print_function

# Created by sarathkaul on 11/11/19

import json
import urllib.request

# Set the webhook_url to the one provided by Slack when you create the webhook at https://my.slack.com/services/new/incoming-webhook/
webhook_url = (
    "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
)
slack_data = {"text": "Hi Sarath Kaul"}

response = urllib.request.Request(
    webhook_url,
    data=json.dumps(slack_data),
    headers={"Content-Type": "application/json"},
)
print(response)
# if response.status_code != 200:
#     raise ValueError(
#         'Request to slack returned an error %s, the response is:\n%s'
#         % (response.status_code, response.text)
#     )
