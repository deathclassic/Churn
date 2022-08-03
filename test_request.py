import requests
import json

url='http://localhost:8000/predict'

with open('customer.json', 'r') as f_in:
	customer = json.loads(f_in.read())

res = requests.post(url, json=customer).json()
print(res)
