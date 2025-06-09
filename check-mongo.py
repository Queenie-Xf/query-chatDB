from pymongo import MongoClient
import time

# Connection details
MONGO_URI = "mongodb://localhost:27017"  # Adjust if using Docker
DB_NAME = "flights"

def check_mongodb_connection():
    print(f"Attempting to connect to MongoDB at: {MONGO_URI}")
    
    try:
        # Connect with timeout
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # Verify connection works
        print("Pinging MongoDB server...")
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB")
        
        # List databases
        print("\nAvailable databases:")
        databases = client.list_database_names()
        for db in databases:
            print(f"- {db}")
        
        # Check the target database
        if DB_NAME in databases:
            print(f"\nDatabase '{DB_NAME}' exists")
            db = client[DB_NAME]
            
            # List collections
            print("\nCollections in database:")
            collections = db.list_collection_names()
            for coll in collections:
                print(f"- {coll}")
                # Show count of documents
                count = db[coll].count_documents({})
                print(f"  • Contains {count} documents")
                
                # Show a sample document if available
                if count > 0:
                    print("  • Sample document:")
                    sample = db[coll].find_one()
                    print(f"    {sample}")
        else:
            print(f"\n❌ Database '{DB_NAME}' does not exist")
            print("Available databases:", databases)
        
        return True
    
    except Exception as e:
        print(f"❌ MongoDB connection error: {e}")
        return False

if __name__ == "__main__":
    check_mongodb_connection()