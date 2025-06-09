import pandas as pd
import sqlite3
import os
import requests
import json
import re
import logging
import socket

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment-aware API URL
def get_api_url():
    backend_host = "backend"
    backend_port = "8000"
    api_url = os.environ.get("API_URL")
    if api_url:
        return api_url
    try:
        socket.gethostbyname(backend_host)
        return f"http://{backend_host}:{backend_port}"
    except socket.gaierror:
        return f"http://localhost:{backend_port}"

API_URL = get_api_url()
logger.info(f"Using API URL: {API_URL}")

SQLITE_DB_DIR = "./data"
LOCATION_DB_PATH = os.path.join(SQLITE_DB_DIR, "hotel_location.db")
RATE_DB_PATH = os.path.join(SQLITE_DB_DIR, "hotel_rate.db")

PROMPT_NOSQL = """You are an expert MongoDB assistant.
Convert the following natural language request into a valid MongoDB query using the appropriate methods like find, aggregate, insertOne, updateOne, deleteOne.

Only return the MongoDB query starting with db... No explanations.

The database has the following structure:

1. Collection: db.flights
   - Fields:
     - startingAirport (string)
     - destinationAirport (string)
     - totalFare (number)
     - isNonStop (boolean)
     - originalId (string)

2. Collection: db.flights_segments
   - Fields:
     - originalId (string)  <-- joins with db.flights.originalId
     - segmentsAirlineName (string)
     - segmentsCabinCode (string)

To get the cheapest flights to a destination:
- Use destinationAirport = \"XXX\"
- Sort by totalFare ascending
- Limit results

To include airline info:
- Join flights with flights_segments using $lookup on originalId

Map city names to airport codes:
- \"Chicago\" → \"ORD\"
- \"New York\" → \"JFK\" or \"LGA\"
- \"Los Angeles\" → \"LAX\"
- \"San Francisco\" → \"SFO\"
- \"Dallas\" → \"DFW\"
- \"Miami\" → \"MIA\"
- \"Denver\" → \"DEN\"

Convert the request carefully using this structure.

Natural language request: """

def build_mongo_prompt(user_query: str) -> str:
   return PROMPT_NOSQL + f"\nNatural language request: {user_query}\nMongoDB query:"

def call_llm_for_query(prompt: str) -> str:
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        })
        return response.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return "// Error generating query"

def generate_mongo_query(query):
    """
    Generate MongoDB query using LLM prompt approach
    """
    prompt = build_mongo_prompt(query)
    return call_llm_for_query(prompt)

# fallback dummy function to avoid backend crash if not implemented
def search_flights(params):
    """
    Search flights in MongoDB with filtering logic.
    """
    from pymongo import MongoClient
    try:
        client = MongoClient("mongodb://mongodb:27017/")
        db = client["travel"]
        query = {}

        if "destination" in params:
            query["destinationAirport"] = params["destination"]
        if "max_price" in params:
            query["totalFare"] = {"$lte": float(params["max_price"])}
        if "isNonStop" in params:
            query["isNonStop"] = params["isNonStop"] in ["true", "True", True]

        projection = {"_id": 0}
        results = list(db.flights.find(query, projection).sort("totalFare", 1).limit(int(params.get("limit", 10))))
        return results
    except Exception as e:
        logger.error(f"Mongo search failed: {e}")
        return []

# Basic natural language processor for destination queries (used by frontend)
def process_flight_nl_query(nl_query):
    query_lower = nl_query.lower()
    airport_map = {
        "chicago": "ORD",
        "new york": "JFK",
        "los angeles": "LAX",
        "san francisco": "SFO",
        "dallas": "DFW",
        "miami": "MIA",
        "denver": "DEN"
    }
    for city, code in airport_map.items():
        if city in query_lower:
            return f'db.flights.find({{"destinationAirport": "{code}"}}).sort({{"totalFare": 1}}).limit(10)', "by_city", {"destination": code}

    return "db.flights.find({}).limit(10)", "default", {}

def format_flights_as_df(flights):
    """Format flight data as pandas DataFrame for display"""
    if not flights:
        return pd.DataFrame()

    formatted_flights = []
    for flight in flights:
        airline_name = "N/A"
        if "segmentDetails" in flight and flight["segmentDetails"]:
            for seg in flight["segmentDetails"]:
                if isinstance(seg, dict) and "segmentsAirlineName" in seg:
                    airline_name = seg["segmentsAirlineName"]
                    break
        elif "segmentsAirlineName" in flight:
            airline_name = flight["segmentsAirlineName"]
        elif "airline_name" in flight:
            airline_name = flight["airline_name"]

        formatted_flights.append({
            "Departure Airport": flight.get("startingAirport", "N/A"),
            "Destination Airport": flight.get("destinationAirport", "N/A"),
            "Airline": airline_name,
            "Price": f"${flight.get('totalFare', 'N/A')}",
            "Duration (min)": flight.get("totalTripDuration", "N/A")
        })

    return pd.DataFrame(formatted_flights)

# Helper functions for entity extraction
def extract_entity(query_lower, entity_type):
    """
    Extract various entities from natural language queries
    """
    if entity_type == "airport_from":
        regex = r'from\s+(\w+)'
        match = re.search(regex, query_lower)
        if match:
            return match.group(1).upper()
        if "lax" in query_lower:
            return "LAX"
        return None  # Return None instead of default if not found
        
    elif entity_type == "airport_to":
        regex = r'to\s+(\w+)'
        match = re.search(regex, query_lower)
        if match:
            return match.group(1).upper()
        if "jfk" in query_lower:
            return "JFK"
        return None  # Return None instead of default if not found
        
    elif entity_type == "airline":
        # More precise airline detection
        airlines = {
            "delta": "Delta", 
            "united": "United", 
            "american": "American",
            "jetblue": "JetBlue",
            "southwest": "Southwest"
        }
        
        # First try to match with "operated by" or similar phrases
        operated_regex = r'(?:operated|run|flown)\s+by\s+([A-Za-z\s]+?)(?:\s+airlines?|\s+air|\s+airways)?\s*(?:$|[,.])'
        match = re.search(operated_regex, query_lower)
        if match:
            airline_name = match.group(1).strip().lower()
            # Check if it matches one of our known airlines
            for key, airline in airlines.items():
                if key in airline_name:
                    return airline
            # Return the extracted name with first letter capitalized if not in our list
            return match.group(1).strip().title()
        
        # If no operation phrase found, try regular matching
        for key, airline in airlines.items():
            if key in query_lower:
                return airline
                
        # Check for the word "airline" preceded by a capitalized word
        airline_regex = r'([A-Z][a-z]+)\s+airlines?'
        match = re.search(airline_regex, query_lower)
        if match:
            return match.group(1)
            
        return None
        
    elif entity_type == "price":
        regex = r'\$?(\d+(?:\.\d+)?)'
        match = re.search(regex, query_lower)
        if match:
            return float(match.group(1))
        return None
        
    elif entity_type == "rating":
        regex = r'(\d+\.?\d*)\s+(?:stars?|rating)'
        match = re.search(regex, query_lower)
        if match:
            return float(match.group(1))
            
        if "above" in query_lower:
            regex = r'above\s+(\d+\.?\d*)'
            match = re.search(regex, query_lower)
            if match:
                return float(match.group(1))
        return None
        
    elif entity_type == "county":
        counties = {
            "orange": "Orange",
            "los angeles": "Los Angeles",
            "san diego": "San Diego",
            "san francisco": "San Francisco"
        }
        
        for key, county in counties.items():
            if key in query_lower:
                return county
                
        regex = r'in\s+([a-zA-Z\s]+)\s+county'
        match = re.search(regex, query_lower)
        if match:
            return match.group(1).strip()
        return None
        
    elif entity_type == "state":
        states = {
            "california": "California",
            "new york": "New York",
            "florida": "Florida",
            "texas": "Texas"
        }
        
        for key, state in states.items():
            if key in query_lower:
                return state
        return None
        
    elif entity_type == "hotel_name":
        regex = r'(?:hotel|called|named)\s+[\'"]?([^\'",]+)[\'"]?'
        match = re.search(regex, query_lower)
        if match:
            return match.group(1).strip()
        return None
        
    elif entity_type == "id":
        regex = r'id\s+(\d+)'
        match = re.search(regex, query_lower)
        if match:
            return match.group(1)
        return None
        
    return None
def detect_query_type(query_text):
    """
    Detect the type of query based on natural language
    Returns a tuple of (query_type, params)
    """
    query_lower = query_text.lower()

    if any(term in query_lower for term in ["flight", "airline", "airport", "fare", "route"]):
        from_airport = extract_entity(query_lower, "airport_from")
        to_airport = extract_entity(query_lower, "airport_to")
        airline = extract_entity(query_lower, "airline")
        price = extract_entity(query_lower, "price")

        if from_airport and to_airport:
            return ("flight_route", {"starting": from_airport, "destination": to_airport})
        elif airline and any(term in query_lower for term in ["operated by", "run by", "operated", "airline"]):
            return ("airline", {"airline": airline})
        elif price and any(term in query_lower for term in ["under", "below", "less than", "<", "cheaper than"]):
            params = {"max_price": price}
            if "non-stop" in query_lower or "nonstop" in query_lower:
                params["isNonStop"] = True
            return ("flight_price", params)
        elif "average price" in query_lower and "airline" in query_lower:
            return ("airline_avg_price", {})
        return ("all_flights", {})

    county = extract_entity(query_lower, "county")
    state = extract_entity(query_lower, "state")
    rating = extract_entity(query_lower, "rating")

    params = {}
    if county:
        params["county"] = county
    if state:
        params["state"] = state
    if rating and "above" in query_lower:
        params["min_rating"] = rating

    if params:
        return ("hotel_filtered", params)
    return ("all_hotels", {})

# Function to generate SQL queries from natural language
def generate_sql_query(query):
    """
    Generate SQL query from natural language query
    """
    query_lower = query.lower()
    DEFAULT_LIMIT = 20
    
    # Schema exploration queries
    if any(term in query_lower for term in ["what tables", "schema", "structure", "columns"]):
        if "hotel" in query_lower or "location" in query_lower:
            return "SELECT name FROM sqlite_master WHERE type='table' OR type='view'", LOCATION_DB_PATH
        elif "rate" in query_lower:
            return "SELECT name FROM sqlite_master WHERE type='table' OR type='view'", RATE_DB_PATH
        else:
            return "SELECT name FROM sqlite_master WHERE type='table' OR type='view'", LOCATION_DB_PATH
    
    # Sample rows exploration
    if "sample" in query_lower:
        table_match = re.search(r'from\s+(\w+)', query_lower)
        if table_match:
            table_name = table_match.group(1)
            return f"SELECT * FROM {table_name} LIMIT {DEFAULT_LIMIT}", LOCATION_DB_PATH if "location" in table_name.lower() or "hotel" in table_name.lower() else RATE_DB_PATH
    
    # Basic hotel queries
    if any(term in query_lower for term in ["hotel", "rating", "county", "state"]) and not any(term in query_lower for term in ["flight", "airline"]):
        # Default query setup
        select_clause = "SELECT h.hotel_name, h.county, h.state, r.rating"
        from_clause = "FROM hotel_complete_view h JOIN rate_complete_view r ON h.ID = r.ID"
        where_clauses = []
        order_clause = "ORDER BY r.rating DESC"
        limit_clause = f"LIMIT {DEFAULT_LIMIT}"
        
        # Add fields
        if "cleanliness" in query_lower:
            select_clause += ", r.cleanliness"
        if "service" in query_lower:
            select_clause += ", r.service"
            
        # Add conditions
        county = extract_entity(query_lower, "county")
        if county:
            where_clauses.append(f"h.county = '{county}'")
            
        state = extract_entity(query_lower, "state")
        if state:
            where_clauses.append(f"h.state = '{state}'")
            
        rating = extract_entity(query_lower, "rating")
        if rating and "above" in query_lower:
            where_clauses.append(f"r.rating >= {rating}")
            
        # Add group by for aggregation
        if "average" in query_lower and "state" in query_lower:
            select_clause = "SELECT h.state, AVG(r.rating) as avg_rating, COUNT(*) as count"
            order_clause = "ORDER BY avg_rating DESC"
            from_clause += " GROUP BY h.state"
            
        if "count" in query_lower and "state" in query_lower:
            select_clause = "SELECT h.state, COUNT(*) as hotel_count"
            order_clause = "ORDER BY hotel_count DESC"
            from_clause += " GROUP BY h.state"
            
        # Adjust limit
        if "top" in query_lower:
            match = re.search(r'top\s+(\d+)', query_lower)
            if match:
                limit = int(match.group(1))
                limit_clause = f"LIMIT {limit}"
        
        # Construct where clause
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        # Build full query
        query = f"{select_clause} {from_clause} {where_clause} {order_clause} {limit_clause}"
        return query, LOCATION_DB_PATH
    
    # Data modification queries
    if any(term in query_lower for term in ["add", "insert", "update", "delete", "remove"]):
        if "add" in query_lower or "insert" in query_lower:
            hotel_name = extract_entity(query_lower, "hotel_name")
            county = extract_entity(query_lower, "county")
            state = extract_entity(query_lower, "state")
            hotel_id = extract_entity(query_lower, "id") or "9999"
            
            if hotel_name and county and state:
                return (
                    f"INSERT INTO hotel_name1 (ID, hotel_name) VALUES ({hotel_id}, '{hotel_name}'); "
                    f"INSERT INTO location (ID, county, state) VALUES ({hotel_id}, '{county}', '{state}');",
                    LOCATION_DB_PATH
                )
        
        elif "update" in query_lower or "change" in query_lower:
            hotel_id = extract_entity(query_lower, "id")
            rating = extract_entity(query_lower, "rating")
            
            if hotel_id and rating and "rating" in query_lower:
                return f"UPDATE rate SET rating = {rating} WHERE ID = {hotel_id}", RATE_DB_PATH
                
        elif "delete" in query_lower or "remove" in query_lower:
            hotel_id = extract_entity(query_lower, "id")
            
            if hotel_id:
                return (
                    f"DELETE FROM hotel_name1 WHERE ID = {hotel_id}; "
                    f"DELETE FROM location WHERE ID = {hotel_id}; "
                    f"DELETE FROM rate WHERE ID = {hotel_id};",
                    LOCATION_DB_PATH
                )
    
    # Default query if nothing else matched
    return "SELECT h.hotel_name, h.county, h.state, r.rating FROM hotel_complete_view h JOIN rate_complete_view r ON h.ID = r.ID ORDER BY r.rating DESC LIMIT 10", LOCATION_DB_PATH

# Functions for MongoDB queries
def generate_mongo_query(query):
    """
    Generate MongoDB query from natural language query
    """
    query_lower = query.lower()
    
    # First detect the query type and get parameters
    query_type, params = detect_query_type(query)
    
    # Schema exploration
    if any(term in query_lower for term in ["what collections", "schema", "structure"]):
        return "db.listCollections()"
        
    # Sample documents
    if "sample" in query_lower:
        if "segment" in query_lower:
            return "db.flights_segments.find().limit(3)"
        else:
            return "db.flights.find().limit(3)"
    
    # Generate MongoDB query based on detected type
    if query_type == "flight_route":
        from_airport = params.get("starting")
        to_airport = params.get("destination")
        
        query = {
            "startingAirport": from_airport,
            "destinationAirport": to_airport
        }
        
        # Add non-stop filter
        if "non-stop" in query_lower or "nonstop" in query_lower:
            query["isNonStop"] = True
            
        # Add price filter
        if "under" in query_lower:
            price = extract_entity(query_lower, "price")
            if price:
                query["totalFare"] = {"$lte": price}
                
        return f"db.flights.find({json.dumps(query)}).sort({{\"totalFare\": 1}}).limit(10)"
    
    elif query_type == "airline":
        airline = params.get("airline")
        return f"db.flights_segments.find({{\"segmentsAirlineName\": {{\"$regex\": \"{airline}\", \"$options\": \"i\"}}}}).limit(10)"
    
    elif query_type == "flight_price":
        max_price = params.get("max_price")
        query = {"totalFare": {"$lte": max_price}}
        return f"db.flights.find({json.dumps(query)}).sort({{\"totalFare\": 1}}).limit(10)"
    
    # Aggregation queries
    elif "average" in query_lower and "airline" in query_lower:
        return """
        db.flights.aggregate([
          {
            $lookup: {
              from: "flights_segments",
              localField: "originalId",
              foreignField: "originalId",
              as: "segments"
            }
          },
          {$unwind: "$segments"},
          {
            $group: {
              _id: "$segments.segmentsAirlineName",
              averagePrice: {$avg: "$totalFare"},
              count: {$sum: 1}
            }
          },
          {$sort: {averagePrice: 1}}
        ])
        """
    
    elif "popular" in query_lower and "route" in query_lower:
        return """
        db.flights.aggregate([
          {
            $group: {
              _id: {from: "$startingAirport", to: "$destinationAirport"},
              count: {$sum: 1},
              avgPrice: {$avg: "$totalFare"}
            }
          },
          {$sort: {count: -1}},
          {$limit: 10}
        ])
        """
    
    # Data modification
    if any(term in query_lower for term in ["add", "insert", "update", "delete", "remove"]) and any(term in query_lower for term in ["flight", "airline"]):
        if "add" in query_lower or "insert" in query_lower:
            from_airport = extract_entity(query_lower, "airport_from")
            to_airport = extract_entity(query_lower, "airport_to")
            price = extract_entity(query_lower, "price") or 199
            
            doc = {
                "startingAirport": from_airport,
                "destinationAirport": to_airport,
                "totalFare": price,
                "isNonStop": True
            }
            
            return f"db.flights.insertOne({json.dumps(doc)})"
            
        elif "update" in query_lower:
            flight_id = extract_entity(query_lower, "id") or "12345"
            price = extract_entity(query_lower, "price") or 299
            
            return f"""
            db.flights.updateOne(
              {{ "originalId": "{flight_id}" }}, 
              {{ "$set": {{ "totalFare": {price} }} }}
            )
            """
            
        elif "delete" in query_lower:
            flight_id = extract_entity(query_lower, "id") or "12345"
            
            return f'db.flights.deleteOne({{ "originalId": "{flight_id}" }})'
            
    # Default MongoDB query
    return "db.flights.find({}).sort({\"totalFare\": 1}).limit(10)"

def execute_query(query_text):
    detected = detect_query_type(query_text)
    if not detected:
        return {"error": "Could not understand your query."}
    query_type, params = detected

    logger.info(f"Executing query: {query_text}")
    logger.info(f"Using API URL: {API_URL}")


    # For flight database queries
    if query_type in ["flight_route", "airline", "flight_price", "all_flights"]:
        mongo_query = generate_mongo_query(query_text)
        # Add default limit if not present
        limit_match = re.search(r'\.limit\((\d+)\)', mongo_query)
        limit = int(limit_match.group(1)) if limit_match else 20
        
        # Use the extracted limit in params
        params["limit"] = limit
        
        # Determine the correct API endpoint based on query type
        if query_type == "flight_route":
            endpoint = f"{API_URL}/flights/airports"
        elif query_type == "airline":
            endpoint = f"{API_URL}/flights/airline"
        else:
            endpoint = f"{API_URL}/flights"
        
        # Log the API call for debugging
        logger.info(f"Making API call to: {endpoint} with params: {params}")
        
        # Execute API call
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                results = response.json()
                logger.info(f"Received {len(results) if isinstance(results, list) else 'non-list'} results")
            else:
                results = {"error": f"API returned status code {response.status_code}"}
                logger.error(f"API error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error calling API: {str(e)}")
            results = {"error": str(e)}
            
        return {"query": mongo_query, "results": results, "type": "mongodb", "params": params, "query_type": query_type}
    
    # For hotel database queries
    else:
        # Generate SQL query for display purposes
        query, db_path = generate_sql_query(query_text)
        
        # Add default limit if not present
        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
        limit = int(limit_match.group(1)) if limit_match else 20
        
        # Use the extracted limit in params
        params["limit"] = limit
            
        # Execute API call
        try:
            endpoint = f"{API_URL}/hotels"
            
            # Log the API call for debugging
            logger.info(f"Making API call to: {endpoint} with params: {params}")
                
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                results = response.json()
                # Convert to DataFrame for consistent handling
                results_df = pd.DataFrame(results) if results else pd.DataFrame()
                logger.info(f"Received {len(results)} results")
            else:
                results = {"error": f"API returned status code {response.status_code}"}
                results_df = pd.DataFrame()
                logger.error(f"API error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error calling API: {str(e)}")
            results = {"error": str(e)}
            results_df = pd.DataFrame()
            
        return {"query": query, "results": results_df, "type": "sql", "db_path": db_path, "params": params, "query_type": query_type}

