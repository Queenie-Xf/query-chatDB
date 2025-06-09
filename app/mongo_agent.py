from pymongo import MongoClient
import os
import json
from bson import json_util, ObjectId
from .db_config import MONGO_URI, MONGO_DB, MONGO_COLLECTIONS, LOCAL_MONGO_URI

# Use environment variables to get URI, default to hardcoded URI
MONGO_URI = os.environ.get("MONGO_URI",
                           "mongodb+srv://flightsdata:dsci551@flightsdata.y57hp.mongodb.net/?retryWrites=true&w=majority")

# Local MongoDB connection string (as backup)
MONGO_HOST = os.environ.get("MONGO_HOST", "mongodb")
MONGO_PORT = os.environ.get("MONGO_PORT", "27017")
LOCAL_MONGO_URI = f"mongodb://{MONGO_HOST}:{MONGO_PORT}"


def get_client():
    """
    Get MongoDB client connection, prioritizing cloud connection
    """
    try:
        # Try to connect to cloud MongoDB first
        client = MongoClient(MONGO_URI)
        # Test connection
        client.server_info()
        print("Successfully connected to cloud MongoDB")
        return client
    except Exception as e:
        print(f"Failed to connect to cloud MongoDB: {e}")
        try:
            # Try using local MongoDB as fallback
            client = MongoClient(LOCAL_MONGO_URI)
            client.server_info()
            print("Successfully connected to local MongoDB")
            return client
        except Exception as e:
            print(f"Failed to connect to local MongoDB: {e}")
            # Re-raise exception
            raise


# Get client connection
client = get_client()
db = client["flights"]  # Connect to flights database


# Function to handle ObjectId conversion
def convert_objectid_to_str(document):
    """
    Convert ObjectId in documents to strings
    """
    if isinstance(document, list):
        return [convert_objectid_to_str(item) for item in document]
    elif isinstance(document, dict):
        result = {}
        for key, value in document.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, (dict, list)):
                result[key] = convert_objectid_to_str(value)
            else:
                result[key] = value
        return result
    else:
        return document


# Schema Exploration functions
def get_collections():
    """
    Get all collections in the database
    """
    return db.list_collection_names()


def get_sample_documents(collection_name, limit=5):
    """
    Get sample documents from specified collection
    """
    # Use the correct collection name from config if it's a known collection
    if collection_name in MONGO_COLLECTIONS:
        actual_collection = MONGO_COLLECTIONS[collection_name]
    else:
        actual_collection = collection_name
        
    collection = db[actual_collection]
    docs = list(collection.find().limit(limit))
    return convert_objectid_to_str(docs)


# Basic query functions
def find_with_projection(collection_name, query={}, projection=None, limit=100):
    """
    Execute find query with projection
    """
    # Use the correct collection name from config if it's a known collection
    if collection_name in MONGO_COLLECTIONS:
        actual_collection = MONGO_COLLECTIONS[collection_name]
    else:
        actual_collection = collection_name
        
    collection = db[actual_collection]
    docs = list(collection.find(query, projection).limit(limit))
    return convert_objectid_to_str(docs)


def aggregate(collection_name, pipeline):
    """
    Execute aggregation pipeline query
    """
    # Use the correct collection name from config if it's a known collection
    if collection_name in MONGO_COLLECTIONS:
        actual_collection = MONGO_COLLECTIONS[collection_name]
    else:
        actual_collection = collection_name
        
    collection = db[actual_collection]
    docs = list(collection.aggregate(pipeline))
    return convert_objectid_to_str(docs)


# Flight query functions
def get_all_flights(limit=100):
    """
    Get all flights (with limit) and include airline information
    """
    # Use the standardized collection names from config
    flights_collection_name = MONGO_COLLECTIONS["flights"]
    segments_collection_name = MONGO_COLLECTIONS["segments"]

    print(f"Getting all flights, using collection: {flights_collection_name}")
    flights_collection = db[flights_collection_name]

    # Use aggregation pipeline to join flights and segments collections
    pipeline = [
        {"$lookup": {
            "from": segments_collection_name,
            "localField": "originalId",
            "foreignField": "originalId",
            "as": "segmentDetails"
        }},
        {"$limit": limit}
    ]

    docs = list(flights_collection.aggregate(pipeline))
    return convert_objectid_to_str(docs)


def get_flights_by_airports(starting, destination):
    """
    Query flights by starting and destination airports
    """
    # Use the standardized collection name from config
    collection_name = MONGO_COLLECTIONS["flights"]
    
    print(f"Querying flights: from {starting} to {destination}, using collection: {collection_name}")
    collection = db[collection_name]

    # Check if there's data in the database
    total_flights = collection.count_documents({})
    print(f"Total flights in database: {total_flights}")

    # Get some sample records to understand the data structure
    sample = list(collection.find({}).limit(2))
    for doc in sample:
        print(f"Sample record fields: {list(doc.keys())}")

    # Execute exact query
    results = list(collection.find({
        "startingAirport": starting,
        "destinationAirport": destination
    }))

    # If no results, try fuzzy query
    if not results:
        print("No exact matches, trying fuzzy query...")
        results = list(collection.find({
            "startingAirport": {"$regex": starting, "$options": "i"},
            "destinationAirport": {"$regex": destination, "$options": "i"}
        }))

    print(f"Number of query results: {len(results)}")
    return convert_objectid_to_str(results)


def get_flights_by_airline(airline_name):
    """
    Query flights by airline name
    """
    # Use the standardized collection names from config
    segments_collection_name = MONGO_COLLECTIONS["segments"]
    flights_collection_name = MONGO_COLLECTIONS["flights"]

    print(f"Querying by airline: {airline_name}, using collection: {segments_collection_name}")
    collection = db[segments_collection_name]

    # Use segmentsAirlineName field for query
    # Since segmentsAirlineName may contain multiple airlines (format: "airline1||airline2"), use regex matching
    query = {"segmentsAirlineName": {"$regex": airline_name, "$options": "i"}}
    segments = list(collection.find(query, {"originalId": 1}))

    if not segments:
        print("No matching flight segments found")
        return []

    # Get list of originalIds from matching segments
    original_ids = [segment["originalId"] for segment in segments]

    # Find corresponding flight details in flights collection
    flights_collection = db[flights_collection_name]
    results = list(flights_collection.find({"originalId": {"$in": original_ids}}))

    print(f"Number of query results: {len(results)}")
    return convert_objectid_to_str(results)


# Advanced query functions
def search_flights(query_params, limit=100):
    """
    Search flights with multiple conditions

    query_params can include:
    - starting: departure airport
    - destination: arrival airport
    - airline: airline name
    - max_price: maximum price
    - min_price: minimum price
    - sort_by: field to sort by
    - sort_order: sort order (1 ascending, -1 descending)
    - skip: number of results to skip
    - limit: maximum number of results to return
    """
    # Use the standardized collection names from config
    flights_collection_name = MONGO_COLLECTIONS["flights"]
    segments_collection_name = MONGO_COLLECTIONS["segments"]

    print(f"Advanced flight search, parameters: {query_params}")

    # Basic query conditions (for flights collection)
    basic_query = {}
    if "starting" in query_params and query_params["starting"]:
        basic_query["startingAirport"] = {"$regex": query_params["starting"], "$options": "i"}
    if "destination" in query_params and query_params["destination"]:
        basic_query["destinationAirport"] = {"$regex": query_params["destination"], "$options": "i"}

    # Price range conditions
    price_condition = {}
    if "max_price" in query_params and query_params["max_price"]:
        price_condition["$lte"] = float(query_params["max_price"])
    if "min_price" in query_params and query_params["min_price"]:
        price_condition["$gte"] = float(query_params["min_price"])
    if price_condition:
        basic_query["totalFare"] = price_condition

    # Airline filter (requires query in flights_segments collection)
    has_airline_filter = "airline" in query_params and query_params["airline"]

    # Get sort parameters
    sort_field = query_params.get("sort_by", "totalFare")
    sort_order = int(query_params.get("sort_order", 1))  # 1 ascending, -1 descending

    # Get pagination parameters
    skip = int(query_params.get("skip", 0))
    limit = int(query_params.get("limit", limit))

    # If no airline filter, query directly in flights collection
    if not has_airline_filter:
        flights_collection = db[flights_collection_name]
        print(f"Built query conditions ({flights_collection_name}): {basic_query}")
        results = list(flights_collection.find(basic_query)
                       .sort(sort_field, sort_order)
                       .skip(skip)
                       .limit(limit))
        return convert_objectid_to_str(results)
    else:
        # If airline filter exists, first query flights_segments for matching originalIds
        segments_collection = db[segments_collection_name]
        segments_query = {"segmentsAirlineName": {"$regex": query_params["airline"], "$options": "i"}}
        print(f"Built query conditions ({segments_collection_name}): {segments_query}")

        matching_segments = list(segments_collection.find(segments_query, {"originalId": 1}))
        original_ids = [segment["originalId"] for segment in matching_segments]

        if not original_ids:
            print("No matching airlines for flights found")
            return []

        # Query flights collection with these originalIds and other filter conditions
        basic_query["originalId"] = {"$in": original_ids}
        flights_collection = db[flights_collection_name]
        print(f"Built query conditions ({flights_collection_name} with originalIds): {basic_query}")

        results = list(flights_collection.find(basic_query)
                       .sort(sort_field, sort_order)
                       .skip(skip)
                       .limit(limit))
        return convert_objectid_to_str(results)


# Aggregation query examples
def get_average_fare_by_airline():
    """
    Get average fare by airline
    """
    # Use the standardized collection names from config
    flights_collection_name = MONGO_COLLECTIONS["flights"]
    segments_collection_name = MONGO_COLLECTIONS["segments"]

    # Step 1: Get all airlines and corresponding originalIds from flights_segments
    segments_collection = db[segments_collection_name]

    # Process segmentsAirlineName field (split multiple airlines)
    pipeline = [
        # Expand segmentsAirlineName field (handle multiple airlines, like "UA||DL")
        {"$addFields": {
            "airlines": {"$split": ["$segmentsAirlineName", "||"]}
        }},
        # Expand airlines array, creating one document per airline
        {"$unwind": "$airlines"},
        # Group by airline and originalId
        {"$group": {
            "_id": {
                "airline": "$airlines",
                "originalId": "$originalId"
            }
        }},
        # Keep only necessary fields
        {"$project": {
            "_id": 0,
            "airline": "$_id.airline",
            "originalId": "$_id.originalId"
        }}
    ]

    airline_flights = list(segments_collection.aggregate(pipeline))

    # Step 2: Get corresponding flight prices from flights collection
    flights_collection = db[flights_collection_name]

    # Calculate average price by airline
    result = {}

    # Calculate separately for each airline
    airlines = set(item["airline"] for item in airline_flights)

    for airline in airlines:
        # Get all originalIds for this airline
        original_ids = [item["originalId"] for item in airline_flights if item["airline"] == airline]

        # Query these originalIds for flight prices
        flights = list(flights_collection.find({"originalId": {"$in": original_ids}}, {"totalFare": 1}))

        if flights:
            # Calculate average price
            total_fare = sum(flight["totalFare"] for flight in flights)
            avg_fare = total_fare / len(flights)

            result[airline] = {
                "averageFare": avg_fare,
                "flightCount": len(flights)
            }

    # Convert to list format for return
    formatted_result = []
    for airline, data in result.items():
        formatted_result.append({
            "airline": airline,
            "averageFare": data["averageFare"],
            "flightCount": data["flightCount"]
        })

    # Sort by average price
    formatted_result.sort(key=lambda x: x["averageFare"])

    return formatted_result


def get_popular_routes(limit=10):
    """
    Get most popular routes (by flight count)
    """
    # Use the standardized collection name from config
    flights_collection_name = MONGO_COLLECTIONS["flights"]
    flights_collection = db[flights_collection_name]

    pipeline = [
        {"$group": {
            "_id": {
                "from": "$startingAirport",
                "to": "$destinationAirport"
            },
            "count": {"$sum": 1},
            "avgFare": {"$avg": "$totalFare"}
        }},
        {"$sort": {"count": -1}},
        {"$limit": limit},
        {"$project": {
            "route": {
                "from": "$_id.from",
                "to": "$_id.to"
            },
            "flightCount": "$count",
            "averageFare": "$avgFare",
            "_id": 0
        }}
    ]

    results = list(flights_collection.aggregate(pipeline))
    return convert_objectid_to_str(results)


# Cross-collection queries
def join_flight_data(limit=100):
    """
    Join flights and flights_segments collections data
    """
    # Use the standardized collection names from config
    flights_collection_name = MONGO_COLLECTIONS["flights"]
    segments_collection_name = MONGO_COLLECTIONS["segments"]

    print(f"Joining {flights_collection_name} and {segments_collection_name} collections data")
    flights_collection = db[flights_collection_name]

    # Use originalId field for joining
    pipeline = [
        {"$lookup": {
            "from": segments_collection_name,
            "localField": "originalId",
            "foreignField": "originalId",
            "as": "segmentDetails"
        }},
        {"$limit": limit}
    ]

    results = list(flights_collection.aggregate(pipeline))
    return convert_objectid_to_str(results)


# Data modification operations
def insert_one(collection_name, document):
    """
    Insert a single document
    """
    # Ensure using correct collection name
    if collection_name in MONGO_COLLECTIONS:
        actual_collection = MONGO_COLLECTIONS[collection_name]
    else:
        actual_collection = collection_name

    collection = db[actual_collection]
    result = collection.insert_one(document)
    return {
        "acknowledged": result.acknowledged,
        "inserted_id": str(result.inserted_id)
    }


def insert_many(collection_name, documents):
    """
    Insert multiple documents
    """
    # Ensure using correct collection name
    if collection_name in MONGO_COLLECTIONS:
        actual_collection = MONGO_COLLECTIONS[collection_name]
    else:
        actual_collection = collection_name

    collection = db[actual_collection]
    result = collection.insert_many(documents)
    return {
        "acknowledged": result.acknowledged,
        "inserted_ids": [str(id) for id in result.inserted_ids],
        "inserted_count": len(result.inserted_ids)
    }


def update_one(collection_name, filter_query, update_query):
    """
    Update a single document
    """
    # Ensure using correct collection name
    if collection_name in MONGO_COLLECTIONS:
        actual_collection = MONGO_COLLECTIONS[collection_name]
    else:
        actual_collection = collection_name

    collection = db[actual_collection]
    result = collection.update_one(filter_query, update_query)
    return {
        "acknowledged": result.acknowledged,
        "matched_count": result.matched_count,
        "modified_count": result.modified_count
    }


def delete_one(collection_name, filter_query):
    """
    Delete a single document
    """
    # Ensure using correct collection name
    if collection_name in MONGO_COLLECTIONS:
        actual_collection = MONGO_COLLECTIONS[collection_name]
    else:
        actual_collection = collection_name

    collection = db[actual_collection]
    result = collection.delete_one(filter_query)
    return {
        "acknowledged": result.acknowledged,
        "deleted_count": result.deleted_count
    }