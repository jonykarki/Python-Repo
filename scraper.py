# Scraper, daraz.com.np
# Author: Jony 8 Oct, 2017
# version 0.1 /beta/

import bs4 as bs
import urllib.request

# base url 
BASE_URL = "https://www.daraz.com.np/catalog/?q="

# user_agent for sending headers with the request
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

# header
headers={'User-Agent':user_agent,} 


#ask the user for search term
# daraz := mobile => smartphones?> ./error?
searchTerm = input('What are you looking for?   ')

# append both url to get the final one
url = BASE_URL + searchTerm

#send request to the server with url and the headers
request = urllib.request.Request(url, None, headers)
response = urllib.request.urlopen(request)
sauce = response.read()

#format the sauce
soup = bs.BeautifulSoup(sauce, 'lxml')

# search for the products
# TODO: Is it an array?
data = soup.find(attrs={'class': 'products'})

alld = data.get_text()

# TODO: Format the above data before printing it.
print(alld)


##for teext in data.find_all('span'):
##    texxt = teext.text
##
##print(texxt)

