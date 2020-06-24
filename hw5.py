# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:21:57 2020

@author: Caroline
"""

from bs4 import BeautifulSoup
import pandas as pd
import csv
import time
import requests

user_agent={'User-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'}

url='http://www.pythonscraping.com/pages/page3.html'

response = requests.get(url, user_agent)
print(response)

soup=BeautifulSoup(response.text, 'lxml')

#print(soup.prettify())


title=soup.find('h1').text
#print(type(title))

rows=soup.find_all('tr',{'class': 'gift'})
#print(len(rows))
#for row in rows:
#    if row.attrs['id'] == 'gift1':
#       print(row)
#    else :
#        pass
    
item_list=[]
for row in rows:  
    item_cell=row.find_all('td')
    item=item_cell[0]
    item=item.text.strip()
    print(item)
    item_list.append(item)
print(item_list)
    
    
    
    









