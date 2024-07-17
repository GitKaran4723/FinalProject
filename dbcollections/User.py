from db import Database
import hashlib
from bson import ObjectId

class User:
    def __init__(self, email, password, company=None, _id=None):
        self.email = email
        self.password = self.hash_password(password) if not self.is_hashed(password) else password
        self.company = company
        self._id = _id  # Store the MongoDB ObjectId

    @staticmethod
    def is_hashed(password):
        return len(password) == 64 and all(c in '0123456789abcdef' for c in password)

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def check_password(self, password):
        return self.password == self.hash_password(password)

    def save(self):
        db = Database.get_db()
        user_data = self.__dict__.copy()
        user_data.pop('_id', None)  # Remove _id to let MongoDB handle it
        return db['users'].insert_one(user_data)
    
    @staticmethod
    def get_all_users():
        db = Database.get_db()
        return list(db['users'].find())
    
    @staticmethod
    def get_user_by_email(email):
        db = Database.get_db()
        user_data = db['users'].find_one({"email": email})

        if user_data:
            return User(**user_data)
        return None
    
    @staticmethod
    def get_user_by_id(user_id):
        db = Database.get_db()
        user_data = db['users'].find_one({"_id": ObjectId(user_id)})
        if user_data:
            return User(**user_data)
        return None
    
    @staticmethod
    def delete_user_by_email(email):
        db = Database.get_db()
        return db['users'].delete_one({"email": email})
    
    @staticmethod
    def update_user_by_email(email, data):
        if 'password' in data:
            data['password'] = User.hash_password_static(data['password'])
        db = Database.get_db()
        return db['users'].update_one({"email": email}, {"$set": data})
    
    @staticmethod
    def delete_all_users():
        db = Database.get_db()
        return db['users'].delete_many({})
    
    @staticmethod
    def update_user_by_id(user_id, data):
        if 'password' in data:
            data['password'] = User.hash_password_static(data['password'])
        db = Database.get_db()
        return db['users'].update_one({"_id": ObjectId(user_id)}, {"$set": data})
    
    @staticmethod
    def delete_user_by_id(user_id):
        db = Database.get_db()
        return db['users'].delete_one({"_id": ObjectId(user_id)})

    @staticmethod
    def hash_password_static(password):
        return hashlib.sha256(password.encode()).hexdigest()
