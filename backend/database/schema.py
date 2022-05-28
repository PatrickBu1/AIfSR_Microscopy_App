from marshmallow import Schema, fields

class User:
    def __init__(self, username, password, image_height, image_width):
        self.username = username
        self.password = password
        self.image_height = image_height
        self.image_width = image_width

class UserSchema(Schema):
    username = fields.Str(required=True)
    password = fields.Str(required=True)
    image_height = fields.Int()
    image_width = fields.Int()