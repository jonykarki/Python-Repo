# send messages on facebook by calling the Mitsuku api
# created: Aug 2019
from fbchat import Client
from fbchat.models import *
import requests
import json
import time

payload = {
    'Accept': "*/*",
    'Accept-Encoding': "gzip, deflate, br",
    'Accept-Language': "en-US, en",
    'q': "0.9",
    'Connection': 'keep-alive',
    'Content-Length': '160',
    'Content-type': 'application/x-www-form-urlencoded',
    'Host': "miapi.pandorabots.com",
    'Origin': "https://www.pandorabots.com",
    'Referer': "https://www.pandorabots.com/mitsuku/",
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36"
}
headers = {}

URL = "https://miapi.pandorabots.com/talk"
PARAMS = {'input': 'lol',
          'sessionid': 403726733,
          'channel': 6,
          'botkey': 'n0M6dW2XZacnOgCWTp0FRYUuMjSfCkJGgobNpgPv9060_72eKnu3Yl-o1v2nFGtSXqfwJBG2Ros~',
          'client_name': 'cw16e8e9d0d10'}


def get_reply(message):
    PARAMS['input'] = message
    # to send the form data use session
    session = requests.Session()
    r = session.post(url=URL, data=PARAMS, headers=payload)
    response = json.loads(r.text)['responses'][0]
    print("Response: " + response)
    return response

# while(True):
#     message = input("Enter the message ")
#     PARAMS['input'] = message
#     print(PARAMS)
#     # to send the form data use session
#     session = requests.Session()
#     r = session.post(url=URL, data=PARAMS, headers=payload)
#     print(json.loads(r.text)['responses'][0])


# #########
# REPLACE WITH EMAIL AND PASSWORD
# #########
EMAIL = ""
PASSWORD = ""


class EchoBot(Client):
    def onMessage(self, mid, author_id, message_object, thread_id, thread_type, **kwargs):
        self.markAsRead(author_id)
        msg = ""
        try:
            print("Message: " + message_object.text)
        except:
            pass

        # don't reply on our own messages
        if author_id == 100002928098103:
            msg = get_reply(message_object.text)
            self.reactToMessage(mid, MessageReaction.ANGRY)
            self.send(Message(text=msg), thread_id=thread_id,
                      thread_type=thread_type)


client = EchoBot(EMAIL, PASSWORD)
client.listen()
