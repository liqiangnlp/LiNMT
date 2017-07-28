#!/usr/bin/python
# encoding: utf-8
import requests
import json
import time

url = "http://10.119.186.29:8088/translate"
#url = "http://10.119.181.207:8200/controller"
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
sentence = "Visto que os canos de lançamento de efluentes da Estação do Tratamento das Águas Residuais da Península de Macau"


data = {"action":"translate", "sourceLang": "pt", "targetLang":"zh", "text": sentence, "alignmentInfo": "true", "nBestSize": 1}
r = requests.post(url, data=json.dumps(data), headers=headers)
res = r.json()
#print res

for ele in res.get('translation', [{}]):
	print ele.get('translated', [{}])[0].get('text', 'No Result').strip()
