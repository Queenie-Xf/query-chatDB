from pymongo import MongoClient
import time

def create_indexes():
    """Create indexes on MongoDB collections to improve performance"""
    print("Creating MongoDB indexes...")
    
    try:
        # Connect to MongoDB with timeout
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
        db = client.flights
        
        # Create index on originalId field in both collections
        db.flights_basic.create_index("originalId")
        print("✅ Created index on flights_basic.originalId")
        
        db.flights_segments.create_index("originalId")
        print("✅ Created index on flights_segments.originalId")
        
        # Create indexes on other commonly queried fields
        db.flights_basic.create_index("startingAirport")
        db.flights_basic.create_index("destinationAirport")
        print("✅ Created indexes on airport fields")
        
        print("All indexes created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating indexes: {e}")
        return False

if __name__ == "__main__":
    create_indexes()