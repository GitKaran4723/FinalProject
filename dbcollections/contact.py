from db import Database

class Contact:
    def __init__(self, name, email, message):
        self.name = name
        self.email = email
        self.message = message

    def save(self):
        db = Database.get_db()
        return db['contactus'].insert_one(self.__dict__)

    @staticmethod
    def get_all_contacts():
        db = Database.get_db()
        return list(db['contactus'].find())

    @staticmethod
    def get_contact_by_id(contact_id):
        db = Database.get_db()
        return db['contactus'].find_one({"_id": contact_id})

    @staticmethod
    def update_contact_by_id(contact_id, data):
        db = Database.get_db()
        return db['contactus'].update_one({"_id": contact_id}, {"$set": data})

    @staticmethod
    def delete_contact_by_id(contact_id):
        db = Database.get_db()
        return db['contactus'].delete_one({"_id": contact_id})
