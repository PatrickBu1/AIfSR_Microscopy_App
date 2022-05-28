import os
from pymongo import MongoClient

COLLECTION_NAME = "users"

class Mongo(object):
    def __init__(self):
        mongo_url = os.environ.get('MONGO_URL')
        self.db = MongoClient(mongo_url).webapp
        
    def find(self, selector):
        return self.db.users.find_one(selector)

    def create(self, user):
        return self.db.users.insert_one(user)

    def update(self, selector, user):
        return self.db.users.replace_one(selector, user).modified_count

    def delete(self, user):
        return self.db.users.delete_one(selector).delete_count