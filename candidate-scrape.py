import json
import urllib.request

# user_agent for sending headers with the request
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

# header
headers={'User-Agent':user_agent,} 

district = input("Enter the Name of the district:   ")

url = "http://election.ujyaaloonline.com/api/candidates?district=" + district

request = urllib.request.Request(url, None, headers)
response = urllib.request.urlopen(request)
source = response.read()

# print(source)

data = json.loads(source)

#print(data['candidates']['2']['400'][0]['cName'])

election_area = data['election_areas']


# get all the possible election-areas from the district
# data needed for the database
'''
resultno :> autoincrement
constituencyname :>
stateno :> Remove the column?
districtno :>
candidate :>
gender :> Remove the column???
votes :> set to zero for now

'''

i = 0
j = 0

for key, value in election_area.items():
    area_key = key
    district_name = data['district_slug']

    try:
        for item in data["candidates"]['1'][area_key]:
            print(item['aName'])
            print(item["cName"])
            i = i + 1

    except:
        for item in data["candidates"]['2'][area_key]:
            print(item['aName'])

            print(item["cName"])
            
            j = j + 1

print(data['district_slug'] + " has " + str(i) + " candidates in provincial election")
print(data['district_slug'] + " has " + str(j) + " candidates in federal election")
print("Total: " + str(i + j) + " candidates added to the database")
