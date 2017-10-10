
# GUI VERSION:/tkinter python
# Author :+> Jony
# Date: 10 Oct, 2017
# Internet Required
# Description: Takes the user inputted song name and plays that song in youtube

import webbrowser
import urllib.request
import bs4 as bs

# import all the tkinter modules as well
from tkinter import *
from tkinter import ttk


# root window for our app
root = Tk()

# title of the windows
root.title("JK Music Player")

# make the window fixed size
# the window is not resizable
root.geometry("400x100+300+300")
root.resizable(width=False, height=False)     # disable resizing


# GUI PART OF THE APP

# main label
Label(root, text="Enter the Name of the song you want to play").grid(row=0, sticky=W, padx=4)

# song should be called later on to find the text inside it
# so we use the following format
song = Entry(root, width=65)
song.grid(row=1, column=0, sticky=W, padx=4)

# this is the label at last
label1 = Label(root)

def play_song(event):

    # user_agent for sending headers with the request
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

    # initialize the header with the fake user-agent
    headers={'User-Agent':user_agent,} 


    # here we fetch the song name
    # data from the label is in string format
    song_name = song.get()

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


    # initialize an empty list for now which will 
    # be filled later on the loop below
    songs_list = []

    # loop through the data inside soup_two
    # only search for 'a' tag
    for url in soup_two.find_all('a'):
        # find all the hrefs from the links found in soup_two
        songs = url.get('href')

        # in each of the above loop go to the empty list 
        # that we initialized before and add the items into it
        # We'll later on get the first item from the list to display 
        songs_list.append(songs)


    # the main url of the video we want 
    # Youtube Main Page
    prefix = 'https://www.youtube.com'

    # intent to open the link of the video in the client's web browser
    webbrowser.open(prefix + songs_list[0])

    # loaded message
    label1['text'] = '''Enjoy the Song'''



# play_button
play_button = Button(root, text="Play The Song")

# Button_Event Handling
play_button.bind("<Button-1>", play_song)  # whenever there's a single left mouse click , call the play_song function
play_button.grid(row=3, column=0, padx=6)


# Info label at the end
label1.grid(row=4, sticky=W, padx=4)
label1['text'] = '''Takes Few Seconds To Load'''

# loop the windows
root.mainloop()


# THE END #
