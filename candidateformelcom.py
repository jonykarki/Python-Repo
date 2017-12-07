import json
import urllib.request
import MySQLdb
db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="",  # your password
                     db="election") 

cur = db.cursor()

# user_agent for sending headers with the request
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

# header
headers={'User-Agent':user_agent,} 

url = "http://result.election.gov.np/JSONFiles/CentralFinalCandidate1.txt?_search=false&nd=1512621126840&rows=10&page=600&sidx=_id&sord=desc"

request = urllib.request.Request(url, None, headers)
response = urllib.request.urlopen(request)
source = response.read()
# print(source)

data = json.loads(source)

for i in range(0, len(data)):
    candidatename  = data[i]['CandidateName'].encode('utf-8')
    gender = data[i]['Gender'].encode('utf-8')
    age = int(data[i]['Age'])
    politicalpartyname = data[i]['PoliticalPartyName'].encode('utf-8')
    districtname = data[i]['DistrictName'].encode('utf-8')
    state = int(data[i]['State'])
    scconstid = data[i]['SCConstID'].encode('utf-8')

    sql = "INSERT INTO `results` (`id`, `candidatename`, `gender`, `age`, `politicalpartyname`, `districtname`, `state`, `scconstid`) VALUES (NULL, %s, %s, %s, %s, %s, %s, %s)"
    cur.execute(sql, (candidatename, gender, age, politicalpartyname, districtname, state, scconstid))

    db.commit()

    print('INSERTED ' + data[i]['CandidateName'] + " into the database")

