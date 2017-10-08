# Scraper, daraz.com.np
# version 0.1 /beta/

import bs4 as bs
import urllib.request


BASE_URL = "https://www.daraz.com.np/catalog/?q="


user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

headers={'User-Agent':user_agent,} 

searchTerm = input('What are you looking for?   ')

url = BASE_URL + searchTerm
request = urllib.request.Request(url, None, headers)
response = urllib.request.urlopen(request)
sauce = response.read()

soup = bs.BeautifulSoup(sauce, 'lxml')

data = soup.find_all("span", class_="name")

print(data)

