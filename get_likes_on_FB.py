from __future__ import print_function

import json
import sys
import urllib

userId = sys.argv[1]  # USERID
limit = 100

accessToken = "TOKENVALUE"
url = f"https://graph.facebook.com/{userId}/posts?access_token={accessToken}&limit={limit}"

data = json.load(urllib.urlopen(url))
id = 0

print(id)

for item in data["data"]:
    time = item["created_time"][11:19]
    date = item["created_time"][5:10]
    year = item["created_time"][:4]

num_share = item["shares"]["count"] if "shares" in item else 0
num_like = item["likes"]["count"] if "likes" in item else 0
id += 1

print(
    str(id)
    + "\t"
    + time.encode("utf-8")
    + "\t"
    + date.encode("utf-8")
    + "\t"
    + year.encode("utf-8")
    + "\t"
    + str(num_share)
    + "\t"
    + str(num_like)
)
