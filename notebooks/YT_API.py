# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: ped-venv
#     language: python
#     name: ped-venv
# ---

# +
import os
from apiclient.discovery import build

# STORE YOUR YOUTUBE API KEY IN THE FILE BELOW!
with open(os.path.join('..', '..', 'tomato-soup.txt'), 'r') as file:
    api_key = file.read()
    
youtube = build('youtube', 'v3', developerKey = api_key)

for i in range(1, 51):
    request = youtube.videoCategories().list(
        part="snippet",
        id=str(i)
    )
    response = request.execute()
    
    try:
        print(f"{i};{response['items'][0]['snippet']['title']}")
    except:
        print(f"{i};?")
