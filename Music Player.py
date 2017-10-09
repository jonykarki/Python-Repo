# Plays the Best match by Youtube® 
# Author :+> Jony
# Internet Required

import os
import webbrowser
import urllib.request
import bs4 as bs

# user_agent for sending headers with the request
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

# header
headers={'User-Agent':user_agent,} 

# query the user
song_name = input("Enter the name of the song to play:  ")

# youtube query
youtube_query = song_name.replace(' ', '+')


# find the best match
# go to youtube and select the first one
url = 'https://www.youtube.com/results?search_query=' + youtube_query


# get the html
request = urllib.request.Request(url, None, headers)
response = urllib.request.urlopen(request)
source_code = response.read()

soup = bs.BeautifulSoup(source_code, 'lxml')

# get the url
soup_two = soup.find(attrs={'class': 'item-section'})  #item-section
#print(soup_two)

for url in soup_two.find_all('a'):
    songs = url.get('href')
    print(songs)


