import httpx

url = "https://api.countrystatecity.in/v1/countries"

headers = {"X-CSCAPI-KEY": "API_KEY"}

response = httpx.get(url, headers=headers)

print(response.text)
