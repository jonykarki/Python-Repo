import json
import urllib.request
import MySQLdb
# db = MySQLdb.connect(host="localhost",    # your host, usually localhost
#                      user="root",         # your username
#                      passwd="",  # your password
#                      db="election") 

# cur = db.cursor()

# user_agent for sending headers with the request
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

# header
headers={'User-Agent':user_agent,} 

url = "https://hamropatro-android.appspot.com/kv/get/election-areas-50-result::1"

request = urllib.request.Request(url, None, headers)
response = urllib.request.urlopen(request)
source = response.read()

# print(source)

data = json.loads(source)

'''
Things required for the results table

resultno: autoIncrement
constituencyname: 
stateno:
districtno:
candidate:
gender: DELETE GENDER
votes: 


'''

candi_datum = data['list']
candi_data = candi_datum[0]

fin_data = candi_data['value']

final_data = json.loads(fin_data)

# get other information if we want
# enter the constituencyname

candidate_results = final_data['candidateResults']

for item in candidate_results:
    candidate = item['englishName']
    print(candidate)

    party = item['partyEnglishName']
    print(party)

    votes = item['votes']
    print(votes)

    print('\n\n')
