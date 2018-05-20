#!/usr/local/bin/python 

from fbchat import Client

EMAIL = "ouremail@email.com"
PASSWORD = "secret"

class EchoBot(Client):
    def onMessage(self, author_id, message_object, thread_id, thread_type, **kwargs):
        self.markAsDelivered(author_id, thread_id)
        self.markAsRead(author_id)

        # don't reply on our own messages
        if author_id != self.uid:
            self.send(message_object, thread_id=thread_id, thread_type=thread_type)

client = EchoBot(USERNAME, PASSWORD)
client.listen()