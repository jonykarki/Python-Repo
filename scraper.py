# scraper for hamrobazar.com
# /*//*///**////**/**/


import bs4 as bs
import urllib.request


# Item the user wants to find
searchTerm = input("What are you looking for?   ")
print(searchTerm)

# find the results of the searchTerm in hamrobazar
sauce = urllib.request.urlopen('http://hamrobazaar.com/search.php?do_search=Search&searchword=' + searchTerm + '&Search.x=0&Search.y=0&catid_search=0').read()


soup = bs.BeautifulSoup(sauce, 'lxml')

# print all the HTML
# print(soup)

# find the table of datas
table = soup.table

# let's just find the table inside of the li tag
li_tag = soup.li

# find the table inside of the li
for table in li_tag:
    table_rows = table.find_all('tr')

    for tr in table_rows:
        td= tr.find_all('td')
        for i in td:
            row_data = i.text
        print(row_data)


# table_rows = table.find_all('tr')

# get the data from the table rows
##for tr in table_rows:
##    td = tr.find_all('td')
##    for i in td:
##        row = i.text
##    print(row)

    
# print(table)


