# Scraper for News from Gooooogle News
# Author =:> Jony
# 09 Oct, 2017
# This script connects to the google news and scrapes the current news and displays it.
# This is the only NEWS program in the Universe with Trump Filter Built In. xD

# Dependency: Beautiful Soup 

import bs4 as bs
import urllib.request

# definition of getTheNews()
def getTheNews():

    # base google news url
    BASE_URL = "https://news.google.com/news/rss?ned=us&hl=en"

    # get the main html of the page
    sauce = urllib.request.urlopen(BASE_URL).read()

    soup = bs.BeautifulSoup(sauce, 'xml')

    # get the list of news first
    list_of_news = soup.find_all("item")

    # TODO: Add Trump Filter
    t_filter = input("Would you like to filter Trump from the list?(y/n)  ")

    if t_filter.lower() == 'n':
        
        # now get the actual news from the list 
        for news in list_of_news:
            # if "Trump" in news.title.text:
            #     print(news.title.text)
            
            print(news.title.text)
            print(news.link.text)
            print(news.pubDate.text)
            print("*"*100)      # Divider
            print(' '*100)      # Spaces for clarity
    else:
        print(' '*100)
        print(' '*100)
        print("The Wall is Being Built.....")
        print(' '*100)
        print(' '*100)

        # The 'trump' filter
        for news in list_of_news:
            if "Trump" not in news.title.text:
                print(news.title.text)
                print(news.link.text)
                print(news.pubDate.text)
                print("*"*100)      # Divider
                print(' '*100)      # Spaces for clarity

# call the function to fetch the current news
getTheNews()
