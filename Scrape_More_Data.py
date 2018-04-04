# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 08:48:27 2018

@author: aakas
"""

#import matplotlib.pyplot as plt
#from PIL import Image
#from io import BytesIO
import urllib.request
import requests
import pickle
import os

#subscription_key = "96d05359d76f4e758906539daeab939e"
subscription_key = "af868f73d1f44344b36d566e82bc6191"
assert subscription_key

headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
params  = {"q": '', "textDecorations":True, "textFormat":"HTML", "count": 20, "size": "medium", 
            "maxFileSize": 25192}


search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

#'socks', 'sneakers', 

commonSearchTerms = ['t-shirts', 'neck-tie', 'underwear', 'boxers', 'pyjamas', 'purses', 'bags', 
                     'sun glasses', 'toys', 'vests', 'merchandise', 'keyholders', 'key chains', 'comics']

## Completed terms 
## 'ant man':'Ant-Man',  'aquaman':'Aquaman', 'avengers':'Avengers' , 'batman':'Batman' , 'black panther' :'Black Panther' 
## 'captain america':'Captain America' , 'catwoman':'Catwoman', 'ghost rider':'Ghost Rider' , 'hulk':'Hulk' , 'she hulk':'Hulk'
## 'iron man': 'Iron Man' , 'spiderman':'Spiderman' , 


mainTerms = { 'spidey': 'Spiderman', 'amazing spiderman':'Spiderman' , 'superman':'Superman'          }

# , 
#           , 
#           

PATH = "C:\\CAX_Superhero_Identify\\train_xtra\\"

for term in mainTerms:
    for common in commonSearchTerms:
        search_term = term + ' ' + common
        params["q"] = search_term
        print("SEARCH TERM: "+ search_term)
        
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        response.raise_for_status()

        search_results = response.json()

        contentUrl = [img["contentUrl"] for img in search_results["value"][:]]
        print("Downloading...")
        for idx in range(len(contentUrl)):
            url = contentUrl[idx]
            if idx == 1:
                pass 
            
            if('jpg' in url):
                filename = search_term + '_' + str(idx) + '.jpg'
                filename = PATH + mainTerms[term]+ '\\' + filename
                if not os.path.exists(PATH + mainTerms[term]):
                    os.makedirs(PATH + mainTerms[term])
                    
                print(url + ">>>" +filename)                        
                try:
                    if('image8' not in url):
                        image_data = urllib.request.urlretrieve(url, filename)
                    else:
                        print("Skipping url..")
                    #image_data.raise_for_status()
                    #image = Image.open(BytesIO(image_data.content))  
                    #image.save(filename)
                except:
                    print("ERR: "+ url)
