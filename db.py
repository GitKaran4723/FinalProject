from pymongo import MongoClient
import os
import dotenv

dotenv.load_dotenv()

# MongoDB URI from environment variables
MONGO_URI = os.getenv("MONGO_NEW")

class Database:
    _client = None

    @staticmethod
    def get_client():
        if Database._client is None:
            Database._client = MongoClient(MONGO_URI)
        return Database._client

    @staticmethod
    def get_db():
        client = Database.get_client()
        return client['project01']

    @staticmethod
    def close_connection():
        if Database._client:
            Database._client.close()
            Database._client = None

# Example usage
if __name__ == "__main__":
    try:
        db = Database.get_db()
        print(db.list_collection_names())  # Example operation
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        Database.close_connection()
