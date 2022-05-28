from mongo import Mongo
from schema import UserSchema

# TODO: check validity of data before doing any DB actions

class DB_Service:
    def __init__(self, username, db_instance=Mongo):
        self.db = db_instance()
        self.username = username

        if not user_id:
            raise Exception("User id not provided")

    def find(self, username):
        data = self.db.find({'username': self.username})
        return self.dump(data)

    def create(self, data):
        data = self.load(data)
        created = self.db.create(data)
        return created > 0

    def update_with(self, username, data):
        data = self.load(data)
        records_affected = self.db.update(data)
        return records_affected > 0

    def delete_for(self, username):
        records_affected = self.db.delete({'username': self.username})
        return records_affected > 0

    def load(self, data):
        return UserSchema().load(data) # load from json

    def dump(self, data):
        return UserSchema().dump(data) # dump form object
