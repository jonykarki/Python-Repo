# Plays the Best match by Youtube® //** The top Result **//
# Author :+> Jony
# Date: 09 Oct, 2017
# Internet Required

import os
import webbrowser
import urllib.request
import bs4 as bs

# user_agent for sending headers with the request
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

# initialize the header with the fake user-agent
headers={'User-Agent':user_agent,} 

# query the user for the song they want to play
song_name = input("Enter the name of the song to play:  ")

# youtube query
# replace all the spaces with plus(+) sign
youtube_query = song_name.replace(' ', '+')


# find the best match
# go to youtube and select the first one
url = 'https://www.youtube.com/results?search_query=' + youtube_query


# get the html
# send the request along with the headers
request = urllib.request.Request(url, None, headers)
response = urllib.request.urlopen(request)
source_code = response.read()

# format all the code
soup = bs.BeautifulSoup(source_code, 'lxml')  

# get the url
# of all the soup 
soup_two = soup.find(attrs={'class': 'item-section'})  #item-section is the part that contains the youtube results


#print(soup_two)  ----> Test
songs_list = []

# loop through the data inside soup_two
# only search for 'a' tag
for url in soup_two.find_all('a'):
    # find all the hrefs from the links found in soup_two
    songs = url.get('href')
    songs_list.append(songs)


# the main url of the video we want
prefix = 'https://www.youtube.com'


# intent to open the link of the video in the client's web browser
webbrowser.open(prefix + songs_list[0])
