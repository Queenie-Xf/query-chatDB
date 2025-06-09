import streamlit as st
st.set_page_config(page_title="Travel Database Query", page_icon="✈️", layout="wide")

import pandas as pd
import os
import sqlite3
import re
import requests
import json
import logging
from schema_display import display_schema_in_streamlit
from query_interface import display_natural_language_query

# Define a consistent default limit across the application
DEFAULT_LIMIT = 20

SQLITE_DB_DIR = "./data"
LOCATION_DB_PATH = os.path.join(SQLITE_DB_DIR, "hotel_location.db")
RATE_DB_PATH = os.path.join(SQLITE_DB_DIR, "hotel_rate.db")
# Setup logging
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
        # Add your code here
        pass
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import socket
        socket.gethostbyname(backend_host)
        return f"http://{backend_host}:{backend_port}"
    except:
        return f"http://localhost:{backend_port}"

API_URL = get_api_url()
logger.info(f"Using API URL: {API_URL}")

# Improved function to display dataframes with proper limit and no duplicate messages
def display_dataframe_with_limit(dataframe, limit=DEFAULT_LIMIT):
    """
    Display a dataframe with a proper limit and height.
    Fixed version to prevent duplicate "Found X results" messages.
    """
    # Check if dataframe is None or empty
    if dataframe is None or (isinstance(dataframe, (list, pd.DataFrame)) and (len(dataframe) == 0)):
        st.info("No results to display")
        return
    
    # Convert to DataFrame if it's a list
    if isinstance(dataframe, list):
        if len(dataframe) > 0:
            df = pd.DataFrame(dataframe)
        else:
            pass  # No action needed
            st.info("No results to display")
            return
    else:
        df = dataframe
    
    # Limit rows to display - show message ONLY ONCE
    if len(df) > limit:
        display_df = df.head(limit)
        st.markdown(f"**Found {len(df)} results (showing first {limit})**")
    else:
        display_df = df
        st.markdown(f"**Found {len(df)} results**")
    
    # Display with appropriate height - using st.dataframe ONLY ONCE
    row_height = 35  # approximate height per row in pixels
    height = min(500, max(150, (len(display_df) + 1) * row_height))  # +1 for header
    
    # Use st.dataframe directly (not st.write) with appropriate parameters
    st.dataframe(display_df, height=height, use_container_width=True)

# Function to process natural language queries for flights
def process_flight_nl_query(nl_query):
    """Process natural language query for flights and return MongoDB query, query type, and parameters"""
    query_lower = nl_query.lower()
    
    # Default values
    mongo_query = f"db.flights.find({{}}).limit({DEFAULT_LIMIT})"
    query_type = "all_flights"
    params = {"limit": DEFAULT_LIMIT}
    
    # Check for origin-destination pattern
    if ("from" in query_lower and "to" in query_lower) or ("between" in query_lower and "and" in query_lower):
        # Extract airport codes - this is a simplified implementation
        # In production, you'd want more sophisticated NLP
        words = query_lower.split()
        for i, word in enumerate(words):
            if word == "from" and i+1 < len(words):
                params["starting"] = words[i+1].upper()
            if word == "to" and i+1 < len(words):
                params["destination"] = words[i+1].upper()
        
        if "starting" in params and "destination" in params:
            mongo_query = f"""db.flights.find({{
                "startingAirport": "{params['starting']}", 
                "destinationAirport": "{params['destination']}"
            }}).sort({{ "totalFare": 1 }}).limit({DEFAULT_LIMIT})"""
            query_type = "by_airports"
    # Check for airline pattern
    elif any(airline in query_lower for airline in ["delta", "american", "united", "southwest", "airlines", "airways"]):
        # Extract airline name - simplified implementation
        airline_keywords = ["delta", "american", "united", "southwest", "jetblue", "frontier"]
        for keyword in airline_keywords:
            if keyword in query_lower:
                params["airline"] = keyword
                break
        
        if "airline" in params:
            mongo_query = f"""db.flights_segments.find({{
                "segmentsAirlineName": {{ "$regex": "{params['airline']}", "$options": "i" }}
            }}).limit({DEFAULT_LIMIT})"""
            query_type = "by_airline"
    
    return mongo_query, query_type, params

# Function to process natural language queries for hotels
def process_hotel_nl_query(nl_query):
    """Process natural language query for hotels and return SQL query and parameters"""
    query_lower = nl_query.lower()
    params = {"limit": DEFAULT_LIMIT}
    
    # Extract county information
    county_match = re.search(r'(in|from) ([a-z]+) county', query_lower)
    if county_match:
        params["county"] = county_match.group(2).title()
    
    # Extract state information
    state_match = re.search(r'(in|from) ([a-z]+)(,| state)', query_lower)
    if state_match:
        params["state"] = state_match.group(2).title()
    
    # Extract rating information
    rating_match = re.search(r'(rating|rated) (above|over|higher than) (\d+\.?\d*)', query_lower)
    if rating_match:
        params["min_rating"] = float(rating_match.group(3))
        
    # Generate appropriate SQL query based on extracted parameters
    if "county" in params and "state" in params and "min_rating" in params:
        sql_query = f"""
        SELECT h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
        FROM hotel_complete_view h
        JOIN rate_complete_view r ON h.ID = r.ID
        WHERE h.county = '{params["county"]}' AND h.state = '{params["state"]}' AND r.rating >= {params["min_rating"]}
        ORDER BY r.rating DESC
        LIMIT {DEFAULT_LIMIT}
        """
    elif "county" in params and "state" in params:
        sql_query = f"""
        SELECT h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
        FROM hotel_complete_view h
        JOIN rate_complete_view r ON h.ID = r.ID
        WHERE h.county = '{params["county"]}' AND h.state = '{params["state"]}'
        ORDER BY r.rating DESC
        LIMIT {DEFAULT_LIMIT}
        """
    elif "county" in params:
        sql_query = f"""
        SELECT h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
        FROM hotel_complete_view h
        JOIN rate_complete_view r ON h.ID = r.ID
        WHERE h.county = '{params["county"]}'
        ORDER BY r.rating DESC
        LIMIT {DEFAULT_LIMIT}
        """
    elif "state" in params:
        sql_query = f"""
        SELECT h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
        FROM hotel_complete_view h
        JOIN rate_complete_view r ON h.ID = r.ID
        WHERE h.state = '{params["state"]}'
        ORDER BY r.rating DESC
        LIMIT {DEFAULT_LIMIT}
        """
    elif "min_rating" in params:
        sql_query = f"""
        SELECT h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
        FROM hotel_complete_view h
        JOIN rate_complete_view r ON h.ID = r.ID
        WHERE r.rating >= {params["min_rating"]}
        ORDER BY r.rating DESC
        LIMIT {DEFAULT_LIMIT}
        """
    else:
        sql_query = f"""
        SELECT h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
        FROM hotel_complete_view h
        JOIN rate_complete_view r ON h.ID = r.ID
        ORDER BY r.rating DESC
        LIMIT {DEFAULT_LIMIT}
        """
    
    return sql_query, params

# Generate SQL query function
def generate_sql_query(natural_query):
    query_lower = natural_query.lower()
    
    # Check for schema requests
    if "select * from" in query_lower or "schema" in query_lower or "show tables" in query_lower:
        if "hotel_location" in query_lower or "location" in query_lower:
            return f"""
            SELECT h1.ID, h1.hotel_name, l.county, l.state
            FROM hotel_name1 h1 JOIN location l ON h1.ID = l.ID
            UNION ALL
            SELECT h2.ID, h2.hotel_name, l.county, l.state
            FROM hotel_name2 h2 JOIN location l ON h2.ID = l.ID
            UNION ALL
            SELECT h3.ID, h3.hotel_name, l.county, l.state
            FROM hotel_name3 h3 JOIN location l ON h3.ID = l.ID
            LIMIT {DEFAULT_LIMIT}
            """, LOCATION_DB_PATH
        elif "hotel_rate" in query_lower or "rate" in query_lower:
            return f"""
            SELECT r.ID, h.hotel_name, r.rating, r.service, r.rooms, r.cleanliness
            FROM rate r
            LEFT JOIN (
                SELECT ID, hotel_name FROM hotel_name1
                UNION ALL 
                SELECT ID, hotel_name FROM hotel_name2
                UNION ALL
                SELECT ID, hotel_name FROM hotel_name3
            ) h ON r.ID = h.ID
            LIMIT {DEFAULT_LIMIT}
            """, RATE_DB_PATH
        else:
            return f"""
            SELECT h.ID, h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
            FROM (
                SELECT h1.ID, h1.hotel_name, l.county, l.state
                FROM hotel_name1 h1 JOIN location l ON h1.ID = l.ID
                UNION ALL
                SELECT h2.ID, h2.hotel_name, l.county, l.state
                FROM hotel_name2 h2 JOIN location l ON h2.ID = l.ID
                UNION ALL
                SELECT h3.ID, h3.hotel_name, l.county, l.state
                FROM hotel_name3 h3 JOIN location l ON h3.ID = l.ID
            ) h
            JOIN rate r ON h.ID = r.ID
            LIMIT {DEFAULT_LIMIT}
            """, LOCATION_DB_PATH
    
    # Sample rows exploration
    if "sample" in query_lower:
        table_match = re.search(r'from\s+(\w+)', query_lower)
        if table_match:
            table_name = table_match.group(1)
            return f"SELECT * FROM {table_name} LIMIT {DEFAULT_LIMIT}", LOCATION_DB_PATH if "location" in table_name.lower() or "hotel" in table_name.lower() else RATE_DB_PATH
    
    # Check for specific queries
    if "orange county" in query_lower:
        return f"""
        SELECT h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
        FROM (
            SELECT h1.ID, h1.hotel_name, l.county, l.state
            FROM hotel_name1 h1 JOIN location l ON h1.ID = l.ID
            UNION ALL
            SELECT h2.ID, h2.hotel_name, l.county, l.state
            FROM hotel_name2 h2 JOIN location l ON h2.ID = l.ID
            UNION ALL
            SELECT h3.ID, h3.hotel_name, l.county, l.state
            FROM hotel_name3 h3 JOIN location l ON h3.ID = l.ID
        ) h
        JOIN rate r ON h.ID = r.ID
        WHERE h.county = 'Orange'
        ORDER BY r.rating DESC
        LIMIT {DEFAULT_LIMIT}
        """, LOCATION_DB_PATH
    elif "california" in query_lower:
        return f"""
        SELECT h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
        FROM (
            SELECT h1.ID, h1.hotel_name, l.county, l.state
            FROM hotel_name1 h1 JOIN location l ON h1.ID = l.ID
            UNION ALL
            SELECT h2.ID, h2.hotel_name, l.county, l.state
            FROM hotel_name2 h2 JOIN location l ON h2.ID = l.ID
            UNION ALL
            SELECT h3.ID, h3.hotel_name, l.county, l.state
            FROM hotel_name3 h3 JOIN location l ON h3.ID = l.ID
        ) h
        JOIN rate r ON h.ID = r.ID
        WHERE h.state = 'CA'
        ORDER BY r.rating DESC
        LIMIT {DEFAULT_LIMIT}
        """, LOCATION_DB_PATH
    elif "best rating" in query_lower or "highest rating" in query_lower:
        return f"""
        SELECT h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
        FROM (
            SELECT h1.ID, h1.hotel_name, l.county, l.state
            FROM hotel_name1 h1 JOIN location l ON h1.ID = l.ID
            UNION ALL
            SELECT h2.ID, h2.hotel_name, l.county, l.state
            FROM hotel_name2 h2 JOIN location l ON h2.ID = l.ID
            UNION ALL
            SELECT h3.ID, h3.hotel_name, l.county, l.state
            FROM hotel_name3 h3 JOIN location l ON h3.ID = l.ID
        ) h
        JOIN rate r ON h.ID = r.ID
        ORDER BY r.rating DESC
        LIMIT {DEFAULT_LIMIT}
        """, LOCATION_DB_PATH
    elif "cleanliness" in query_lower:
        return f"""
        SELECT h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
        FROM (
            SELECT h1.ID, h1.hotel_name, l.county, l.state
            FROM hotel_name1 h1 JOIN location l ON h1.ID = l.ID
            UNION ALL
            SELECT h2.ID, h2.hotel_name, l.county, l.state
            FROM hotel_name2 h2 JOIN location l ON h2.ID = l.ID
            UNION ALL
            SELECT h3.ID, h3.hotel_name, l.county, l.state
            FROM hotel_name3 h3 JOIN location l ON h3.ID = l.ID
        ) h
        JOIN rate r ON h.ID = r.ID
        ORDER BY r.cleanliness DESC
        LIMIT {DEFAULT_LIMIT}
        """, LOCATION_DB_PATH
    
    # Default query
    return f"""
    SELECT h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
    FROM (
        SELECT h1.ID, h1.hotel_name, l.county, l.state
        FROM hotel_name1 h1 JOIN location l ON h1.ID = l.ID
        UNION ALL
        SELECT h2.ID, h2.hotel_name, l.county, l.state
        FROM hotel_name2 h2 JOIN location l ON h2.ID = l.ID
        UNION ALL
        SELECT h3.ID, h3.hotel_name, l.county, l.state
        FROM hotel_name3 h3 JOIN location l ON h3.ID = l.ID
    ) h
    JOIN rate r ON h.ID = r.ID
    LIMIT {DEFAULT_LIMIT}
    """, LOCATION_DB_PATH

# Function to generate MongoDB query from natural language
def generate_mongo_query(natural_language_query):
    """
    Generate MongoDB query from natural language using Ollama API
    """
    # Get Ollama API URL from environment or use default
    OLLAMA_API = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
    
    try:
        # Create a more effective prompt with examples
        prompt = f"""
        You are a MongoDB query generator. Your task is to convert natural language queries into MongoDB queries for a flight database.

        Natural Language Query: "{natural_language_query}"

        Database Structure:
        - flights_basic collection: contains fields startingAirport, destinationAirport, totalFare, travelDuration
        - flights_segments collection: contains fields originalId, segmentsAirlineName

        Example Conversions:
        - "Find flights from SFO" → db.flights_basic.find({{"startingAirport": "SFO"}}).limit({DEFAULT_LIMIT})
        - "Show Delta Airlines flights" → db.flights_segments.find({{"segmentsAirlineName": {{"$regex": "Delta", "$options": "i"}}}}).limit({DEFAULT_LIMIT})
        - "Find flights from LAX to JFK" → db.flights_basic.find({{"startingAirport": "LAX", "destinationAirport": "JFK"}}).limit({DEFAULT_LIMIT})
        - "What are the cheapest flights?" → db.flights_basic.find({{}}).sort({{"totalFare": 1}}).limit({DEFAULT_LIMIT})

        Return ONLY the valid MongoDB query, no explanation.
        """

        # Call Ollama API with optimized parameters
        response = requests.post(
            f"{OLLAMA_API}/api/generate",
            json={
                "model": "llama3",  # Use llama3 as shown in the logs
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,  # Low temperature for more predictable results
                "top_p": 0.9,  # Focus on more likely tokens
                "stop": ["\n", ";"]  # Stop generation at newlines or semicolons
            }
        )

        if response.status_code == 200:
            result = response.json()
            generated_query = result.get("response", "").strip()

            # Clean up the generated query
            # Remove any markdown code block formatting
            generated_query = generated_query.replace("```javascript", "").replace("```", "").strip()

            # Basic validation of the generated query
            if not generated_query.startswith("db."):
                # If no valid query was generated, create a basic query
                st.warning("The model didn't generate a valid MongoDB query. Using a default query.")
                generated_query = f"db.flights_basic.find({{}}).limit({DEFAULT_LIMIT})"

            # Make sure it has a limit
            if "limit" not in generated_query:
                # Add limit before the closing parenthesis if not present
                if generated_query.endswith(")"):
                    generated_query = generated_query[:-1] + f".limit({DEFAULT_LIMIT}))"
                else:
                    generated_query = generated_query + f".limit({DEFAULT_LIMIT})"

            return generated_query
        else:
            st.error(f"Error calling Ollama API: {response.status_code} - {response.text}")
            return f"db.flights_basic.find({{}}).limit({DEFAULT_LIMIT})"  # Default fallback query
    except Exception as e:
        st.error(f"Error generating MongoDB query: {str(e)}")
        return f"db.flights_basic.find({{}}).limit({DEFAULT_LIMIT})"  # Default fallback query
    # Function to parse the MongoDB query into components for API call
def parse_mongo_query(query_string):
    """
    Parse a MongoDB query string into components needed for API call
    """
    # Default values
    query_type = "mongo_query"  # Use mongo_query as type for direct execution
    params = {"limit": DEFAULT_LIMIT, "mongo_query": query_string}  # Include original query string

    try:
        # Extract collection name
        collection_match = re.search(r'db\.(\w+)\.', query_string)
        collection = collection_match.group(1) if collection_match else "flights_basic"

        # Map collection names to match backend expectations
        if collection == "flights_basic":
            # Update the query to use "flights" instead
            query_string = query_string.replace("db.flights_basic", "db.flights")
            params["mongo_query"] = query_string
            collection = "flights"
        elif collection == "flights_segments":
            # Update the query to use "segments" instead
            query_string = query_string.replace("db.flights_segments", "db.segments")
            params["mongo_query"] = query_string
            collection = "segments"
        
        # Extract query parameters
        query_params_match = re.search(r'find\(\s*(\{.*?\})\s*\)', query_string)
        query_params = {}

        if query_params_match:
            # Try to parse the query parameters
            params_str = query_params_match.group(1)

            # Handle advanced regex patterns in the query
            # Replace any single quotes with double quotes for valid JSON
            params_str = params_str.replace("'", '"')

            try:
                query_params = json.loads(params_str)
            except json.JSONDecodeError:
                # If we can't parse the JSON, try to extract keys and values manually
                st.warning(f"Could not parse query parameters as JSON: {params_str}")
                query_params = {}

        # Extract limit
        limit_match = re.search(r'\.limit\((\d+)\)', query_string)
        limit = int(limit_match.group(1)) if limit_match else DEFAULT_LIMIT
        params["limit"] = limit

        # Determine query type based on parameters - this helps our API routing
        if "startingAirport" in query_params and "destinationAirport" in query_params:
            query_type = "by_airports"
            params["starting"] = query_params["startingAirport"]
            params["destination"] = query_params["destinationAirport"]
        elif "segmentsAirlineName" in query_params or collection == "segments":
            query_type = "by_airline"
            # Try to extract airline name from regex pattern if present
            if isinstance(query_params.get("segmentsAirlineName"), dict):
                regex = query_params["segmentsAirlineName"].get("$regex", "")
                params["airline"] = regex
            else:
                params["airline"] = query_params.get("segmentsAirlineName", "")

        # Add collection name to params
        params["collection"] = collection

        return query_type, params, query_string

    except Exception as e:
        st.error(f"Error parsing MongoDB query: {str(e)}")
        return "mongo_query", {"limit": DEFAULT_LIMIT, "mongo_query": query_string, "collection": "flights"}, query_string

# Function to extract entities from natural language queries
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

# Function to get flights by direct API
def get_flights_by_direct_api(query_type, params=None):
    """
    Directly query backend API endpoints instead of using MongoDB query execution
    """
    if params is None:
        params = {}
    
    # Make sure params has a limit value
    if "limit" not in params:
        params["limit"] = DEFAULT_LIMIT

    try:
        # Handle direct MongoDB query execution
        if query_type == "mongo_query" and "mongo_query" in params:
            # Execute the MongoDB query directly through our new endpoint
            payload = {
                "collection": params.get("collection", "flights"),
                "query": params["mongo_query"],
                "limit": params.get("limit", DEFAULT_LIMIT)
            }
            response = requests.post(
                f"{API_URL}/execute_mongo_query",
                json=payload
            )
            if st.sidebar.checkbox("Show Debug Info", key="debug_info_mongo"):
                st.write(f"Debug: Using /execute_mongo_query endpoint with query: {params['mongo_query']}")

            if response.status_code == 200:
                flights = response.json()
                return flights
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return []

        # Select the appropriate endpoint based on query type
        if query_type == "by_airports" and "starting" in params and "destination" in params:
            # Add a limit parameter to avoid performance issues
            limit = params.get("limit", DEFAULT_LIMIT)

            # Query for flights between specific airports
            response = requests.get(
                f"{API_URL}/flights/airports",
                params={
                    "starting": params["starting"],
                    "destination": params["destination"],
                    "limit": limit  # Add limit to query parameters
                }
            )

        elif query_type == "by_airline" and "airline" in params:
            # Add a limit parameter to avoid performance issues
            limit = params.get("limit", DEFAULT_LIMIT)

            # Query for flights by airline
            response = requests.get(
                f"{API_URL}/flights/airline",
                params={"airline": params["airline"], "limit": limit}
            )

        else:
            # Default to getting all flights with limit
            limit = params.get("limit", DEFAULT_LIMIT)
            response = requests.get(f"{API_URL}/flights", params={"limit": limit})

        if response.status_code == 200:
            flights = response.json()
            result_count = len(flights)

            # If we got too many results, limit them to avoid performance issues
            if result_count > 100:
                st.warning(f"Found {result_count} matching flights. Showing first 100 results.")
                flights = flights[:100]  # Limit to first 100 flights

            return flights
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []

    except Exception as e:
        st.error(f"Error querying API: {str(e)}")
        return []

# Execute Query function
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
        limit = int(limit_match.group(1)) if limit_match else DEFAULT_LIMIT
        
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
        try:
            query, db_path = generate_sql_query(query_text)
        except:
            # If generate_sql_query fails, we'll use a default query
            query = f"""
            SELECT h.hotel_name, h.county, h.state, r.rating, r.cleanliness, r.service, r.rooms
            FROM hotel_complete_view h
            JOIN rate_complete_view r ON h.ID = r.ID
            LIMIT {DEFAULT_LIMIT}
            """
            db_path = "default"
        
        # Add default limit if not present
        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
        limit = int(limit_match.group(1)) if limit_match else DEFAULT_LIMIT
        
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

# Utility function to check backend and database connections
def check_connections():
    """Check connections to backend and databases"""
    # Check backend connection
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        if response.status_code == 200:
            st.sidebar.success("✅ Backend API Connected")
        else:
            st.sidebar.error(f"❌ Backend API Error (Status {response.status_code})")
    except Exception as e:
        st.sidebar.error(f"❌ Backend API Not Connected: {str(e)[:50]}...")
    
    # Check Ollama connection
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            st.sidebar.success(f"✅ Ollama LLM Connected ({', '.join(model_names[:3]) if model_names else 'No models found'})")
        else:
            st.sidebar.error(f"❌ Ollama LLM Error (Status {response.status_code})")
    except Exception as e:
        st.sidebar.error(f"❌ Ollama LLM Not Connected: {str(e)[:50]}...")
    
    # Check MongoDB connection (via backend API)
    try:
        response = requests.get(f"{API_URL}/flights?limit=1", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                st.sidebar.success(f"✅ MongoDB Connected")
            else:
                st.sidebar.warning("⚠️ MongoDB Connected but no data found")
        else:
            st.sidebar.error(f"❌ MongoDB Error (Status {response.status_code})")
    except Exception as e:
        st.sidebar.error(f"❌ MongoDB Not Connected: {str(e)[:50]}...")
    
    # Check SQLite connection (via backend API)
    try:
        response = requests.get(f"{API_URL}/hotels?limit=1", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                st.sidebar.success(f"✅ SQLite Connected")
            else:
                st.sidebar.warning("⚠️ SQLite Connected but no data found")
        else:
            st.sidebar.error(f"❌ SQLite Error (Status {response.status_code})")
    except Exception as e:
        st.sidebar.error(f"❌ SQLite Not Connected: {str(e)[:50]}...")

# Example queries section
def add_example_queries_section():
    """
    Display example queries with a clean, organized interface
    """
    with st.expander("📚 Example Queries", expanded=True):
        st.markdown("### Click on any example to try it")
        
        # Create tabs for SQL and MongoDB examples
        sql_tab, mongo_tab = st.tabs(["SQL (Hotel Database)", "MongoDB (Flight Database)"])
        
        with sql_tab:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Basic Queries")
                hotel_examples = [
                    "Find hotels in Orange County",
                    "Show hotels with ratings above 4.5",
                    "List hotels in California with good service",
                    "What tables exist in the hotel database?"
                ]
                
                for i, example in enumerate(hotel_examples):
                    if st.button(example, key=f"hotel_ex_{i}"):
                        st.session_state.query = example
                        st.rerun()  # Changed from st.experimental_rerun()
            
            with col2:
                st.markdown("#### Advanced Queries")
                advanced_hotel_examples = [
                    "Show average ratings by state",
                    "Find hotels with excellent cleanliness",
                    "Show hotels where service rating > cleanliness",
                    "Count hotels in each county in California"
                ]
                
                for i, example in enumerate(advanced_hotel_examples):
                    if st.button(example, key=f"adv_hotel_ex_{i}"):
                        st.session_state.query = example
                        st.rerun()  # Changed from st.experimental_rerun()
        
        with mongo_tab:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Basic Queries")
                flight_examples = [
                    "Find flights from LAX to JFK",
                    "Show flights operated by Delta Airlines",
                    "Find the cheapest flights to Chicago",
                    "What collections exist in the flight database?"
                ]
                
                for i, example in enumerate(flight_examples):
                    if st.button(example, key=f"flight_ex_{i}"):
                        st.session_state.query = example
                        st.rerun()  # Changed from st.experimental_rerun()
            
            with col2:
                st.markdown("#### Advanced Queries")
                advanced_flight_examples = [
                    "Show average price by airline",
                    "Find non-stop flights under $300",
                    "Find flights with Delta that arrive after 3pm",
                    "Show the most popular flight routes"
                ]
                
                for i, example in enumerate(advanced_flight_examples):
                    if st.button(example, key=f"adv_flight_ex_{i}"):
                        st.session_state.query = example
                        st.rerun()  # Changed from st.experimental_rerun()

# Display natural language query interface
def display_nl_query_interface():
    """
    Display a natural language query interface with support for both SQL and MongoDB
    """
    # Initialize session state if not already set
    if 'query_type' not in st.session_state:
        st.session_state.query_type = 'sql'
    if 'query' not in st.session_state:
        st.session_state.query = ''
    
    # Natural language query input
    nl_query = st.text_area(
        "Enter your query in natural language:",
        height=100,
        key="nl_query",
        value=st.session_state.get('query', '')
    )

    col1, col2 = st.columns(2)
    with col1:
        query_type = st.radio(
            "Select Database Type:",
            ["SQL (Hotel Database)", "MongoDB (Flight Database)"],
            index=0 if st.session_state.get('query_type', 'sql') == 'sql' else 1,
            key="query_type_radio"
        )
        # Update session state when radio button changes
        st.session_state.query_type = "sql" if query_type == "SQL (Hotel Database)" else "mongo"
    
    with col2:
        if st.button("Execute Query", key="execute_nl_query"):
            if not nl_query:
                st.warning("Please enter a query")
                return
            
            # Process the query based on selected type
            if st.session_state.query_type == "sql":
                # Process SQL (Hotel) query
                sql_query, params = process_hotel_nl_query(nl_query)
                st.session_state.generated_query = sql_query
                st.session_state.query_params = params
                
                # Display the generated SQL query
                st.subheader("Generated SQL Query:")
                st.code(sql_query, language="sql")
                
                # Execute the query
                try:
                    result = execute_query(nl_query)
                    
                    if result:
                        if "query" in result:
                            st.subheader("Executed SQL Query:")
                            st.code(result["query"], language="sql")
                        
                        if "results" in result and isinstance(result["results"], pd.DataFrame):
                            st.subheader("Query Results:")
                            # Use our custom display function with limit from params
                            limit = params.get("limit", DEFAULT_LIMIT)
                            display_dataframe_with_limit(result["results"], limit)
                        elif "results" in result and isinstance(result["results"], list):
                            # Convert list of dicts to DataFrame
                            df = pd.DataFrame(result["results"])
                            st.subheader("Query Results:")
                            # Use our custom display function with limit from params
                            limit = params.get("limit", DEFAULT_LIMIT)
                            display_dataframe_with_limit(df, limit)
                        elif "results" in result and isinstance(result["results"], dict) and "error" in result["results"]:
                            st.error(f"Error: {result['results']['error']}")
                        else:
                            st.info("No results returned or format not recognized")
                    else:
                        st.warning("No results returned from the backend")
                    
                except Exception as e:
                    st.error(f"Error executing SQL query: {str(e)}")
                    
            else:
                # Process MongoDB (Flight) query
                mongo_query, query_type, params = process_flight_nl_query(nl_query)
                st.session_state.generated_query = mongo_query
                st.session_state.query_params = params
                
                # Display the generated MongoDB query
                st.subheader("Generated MongoDB Query:")
                st.code(mongo_query, language="javascript")
                
                # Execute the query
                try:
                    # Get flights using the API
                    flights = get_flights_by_direct_api(query_type, params)
                    if flights:
                        st.subheader(f"Results:")
                        # Convert to DataFrame for better display
                        df = pd.DataFrame(flights)
                        
                        # Simplify columns for better display if too many columns
                        if len(df.columns) > 10:
                            # Priority columns to show
                            priority_cols = ["startingAirport", "destinationAirport", "totalFare", 
                                            "segmentsAirlineName", "totalTripDuration", "isNonStop"]
                            display_cols = [col for col in priority_cols if col in df.columns]
                            
                            # Add a few more columns if available
                            remaining_cols = [col for col in df.columns if col not in display_cols]
                            display_cols.extend(remaining_cols[:10 - len(display_cols)])
                            
                            # Use our custom display function with the limit from params
                            limit = params.get("limit", DEFAULT_LIMIT)
                            display_dataframe_with_limit(df[display_cols], limit)
                            st.info(f"Showing {len(display_cols)} of {len(df.columns)} available columns")
                        else:
                            # Use our custom display function with the limit from params
                            limit = params.get("limit", DEFAULT_LIMIT)
                            display_dataframe_with_limit(df, limit)
                    else:
                        st.warning("No flights found matching your criteria")
                except Exception as e:
                    st.error(f"Error executing MongoDB query: {str(e)}")
    
    # Add alternative query generation using LLM for more complex queries
    with st.expander("Advanced Query Generation (Using LLM)"):
        st.info("For more complex queries, you can use our LLM-powered query generator.")
        
        if st.button("Generate Query with LLM", key="llm_generate"):
            if not nl_query:
                st.warning("Please enter a query above")
            else:
                with st.spinner("Generating query using LLM..."):
                    if st.session_state.query_type == "mongo":
                        # Use LLM to generate MongoDB query
                        try:
                            mongo_query = generate_mongo_query(nl_query)
                            st.session_state.generated_query = mongo_query
                            
                            # Display the generated query
                            st.subheader("LLM-Generated MongoDB Query:")
                            st.code(mongo_query, language="javascript")
                            
                            # Parse the query for execution
                            query_type, params, _ = parse_mongo_query(mongo_query)
                            
                            # Execute the query
                            flights = get_flights_by_direct_api(query_type, params)
                            if flights:
                                st.subheader(f"Results:")
                                # Convert to DataFrame for better display
                                df = pd.DataFrame(flights)
                                
                                # Simplify columns for better display if too many columns
                                if len(df.columns) > 10:
                                    # Priority columns to show
                                    priority_cols = ["startingAirport", "destinationAirport", "totalFare", 
                                                    "segmentsAirlineName", "totalTripDuration", "isNonStop"]
                                    display_cols = [col for col in priority_cols if col in df.columns]
                                    
                                    # Add a few more columns if available
                                    remaining_cols = [col for col in df.columns if col not in display_cols]
                                    display_cols.extend(remaining_cols[:10 - len(display_cols)])
                                    
                                    # Use our custom display function
                                    limit = params.get("limit", DEFAULT_LIMIT)
                                    display_dataframe_with_limit(df[display_cols], limit)
                                    st.info(f"Showing {len(display_cols)} of {len(df.columns)} available columns")
                                else:
                                    # Use our custom display function
                                    limit = params.get("limit", DEFAULT_LIMIT)
                                    display_dataframe_with_limit(df, limit)
                            else:
                                st.warning("No flights found matching your criteria")
                        except Exception as e:
                            st.error(f"Error generating or executing MongoDB query: {str(e)}")
                    else:
                        st.warning("LLM query generation is currently only supported for MongoDB (Flight) queries")
# Main application code
def main():
    st.title("Travel Information System")

    # Display connection status in sidebar
    st.sidebar.title("System Status")
    check_connections()

    # Add information about the current API URL
    st.sidebar.markdown(f"**API URL:** {API_URL}")

    # Display API URL and allow manual override
    with st.sidebar.expander("API Configuration"):
        new_api_url = st.text_input("Override API URL", value=API_URL)
        if st.button("Update API URL"):
            os.environ["API_URL"] = new_api_url
            st.success(f"API URL updated to {new_api_url}")
            st.rerun()  # Changed from st.experimental_rerun()

    # Create main tabs - keeping all three tabs
    tab1, tab2, tab3 = st.tabs(["Natural Language Query", "Database Schema", "Data Modification"])

    # Natural Language Query tab
    with tab1:
        # Initialize session state if needed
        if 'query' not in st.session_state:
            st.session_state.query = ''

        # Add example queries section
        add_example_queries_section()
        
        # Create query input area
        query_input = st.text_area(
            "Enter your query in natural language:",
            value=st.session_state.get('query', ''),
            height=100,
            key="nl_query"
        )
        
        # Execute button
        if st.button("Execute Query", key="execute_btn"):
            if not query_input:
                st.warning("Please enter a query")
            else:
                # Update session state
                st.session_state.query = query_input
                
                # Execute query
                with st.spinner("Processing query..."):
                    result = execute_query(query_input)
                
                # Display query info
                st.markdown("---")
                st.markdown("## Query Results")
                
                # Display database type
                if result["type"] == "sql":
                    st.markdown("### Database: SQL (Hotel Database)")
                else:
                    st.markdown("### Database: MongoDB (Flight Database)")
                
                # Show the generated query
                with st.expander("View Generated Query", expanded=False):
                    if result["type"] == "sql":
                        st.code(result["query"], language="sql")
                    else:
                        st.code(result["query"], language="javascript")
                    
                    st.markdown("### API Parameters:")
                    st.json(result["params"])
                
                # Display results
                st.markdown("### Results:")
                
                # Handle SQL (Hotel) results - Using the improved display function
                if result["type"] == "sql" and isinstance(result["results"], pd.DataFrame):
                    if not result["results"].empty:
                        display_dataframe_with_limit(result["results"], result["params"].get("limit", DEFAULT_LIMIT))
                    else:
                        st.info("No results found matching your criteria.")
                
                # Handle MongoDB (Flight) results - Using the improved display function
                elif result["type"] == "mongodb" and isinstance(result["results"], list):
                    if result["results"]:
                        df = pd.DataFrame(result["results"])
                        
                        # Show relevant columns for flights
                        if "startingAirport" in df.columns:
                            priority_cols = ["startingAirport", "destinationAirport", "totalFare", "isNonStop"]
                            display_cols = [col for col in priority_cols if col in df.columns]
                            
                            # If no priority columns found, use all columns
                            if not display_cols:
                                display_cols = df.columns
                            
                            display_dataframe_with_limit(df[display_cols], result["params"].get("limit", DEFAULT_LIMIT))
                        else:
                            display_dataframe_with_limit(df, result["params"].get("limit", DEFAULT_LIMIT))
                    else:
                        st.info("No results found matching your criteria.")
    
    # Database Schema tab
    with tab2:
        display_schema_in_streamlit()

    # Data Modification tab
    # Data Modification tab
    with tab3:
        st.header("Data Modification")

        # Create subtabs for different databases
        mod_tab1, mod_tab2 = st.tabs(["Hotel Database Modification", "Flight Database Modification"])

        # Hotel Database Modification
        with mod_tab1:
            st.subheader("Modify Hotel Data")

            hotel_operation = st.radio(
                "Select Operation",
                ["Add New Hotel", "Update Hotel", "Delete Hotel"],
                horizontal=True
            )

            if hotel_operation == "Add New Hotel":
                with st.form("add_hotel_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        hotel_name = st.text_input("Hotel Name", key="add_hotel_name")
                        county = st.text_input("County", key="add_county")
                        state = st.text_input("State (2-letter code)", key="add_state", max_chars=2)
                    with col2:
                        rating = st.slider("Rating", min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                                           key="add_rating")
                        cleanliness = st.slider("Cleanliness", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
                        service = st.slider("Service", min_value=1.0, max_value=5.0, value=3.0, step=0.5)

                    rooms = st.slider("Rooms", min_value=1.0, max_value=5.0, value=3.0, step=0.5)

                    submit_button = st.form_submit_button("Add Hotel")

                    if submit_button:
                        if not hotel_name or not county or not state:
                            st.error("Please fill in all required fields (Hotel Name, County, State)")
                        else:
                            hotel_data = {
                                "hotel_name": hotel_name,
                                "county": county,
                                "state": state.upper(),
                                "rating": rating,
                                "cleanliness": cleanliness,
                                "service": service,
                                "rooms": rooms
                            }

                            try:
                                response = requests.post(f"{API_URL}/hotels", json=hotel_data)
                                if response.status_code == 200 or response.status_code == 201:
                                    st.success(f"Successfully added hotel: {hotel_name}")

                                    # Display the SQL queries that would be executed
                                    st.subheader("Equivalent SQL Queries:")

                                    # Get the hotel_id from the response if available
                                    hotel_id = "new_id"
                                    try:
                                        response_data = response.json()
                                        if "id" in response_data:
                                            hotel_id = response_data["id"]
                                    except:
                                        pass

                                    # Display the insert queries
                                    hotel_name_insert = f"""
                                    -- Insert into hotel_name1
                                    INSERT INTO hotel_name1 (hotel_name) VALUES ('{hotel_name}');
                                    """
                                    st.code(hotel_name_insert, language="sql")

                                    location_insert = f"""
                                    -- Insert into location
                                    INSERT INTO location (ID, county, state) 
                                    VALUES ({hotel_id}, '{county}', '{state.upper()}');
                                    """
                                    st.code(location_insert, language="sql")

                                    rating_insert = f"""
                                    -- Insert into rate
                                    INSERT INTO rate (ID, rating, service, rooms, cleanliness)
                                    VALUES ({hotel_id}, {rating}, {service}, {rooms}, {cleanliness});
                                    """
                                    st.code(rating_insert, language="sql")

                                    st.balloons()
                            except Exception as e:
                                st.error(f"Error connecting to API: {str(e)}")


            elif hotel_operation == "Update Hotel":

                st.subheader("Update Hotel Information")

                # Let user enter a hotel ID directly

                hotel_id = st.number_input("Enter Hotel ID to update:", min_value=1, step=1)

                if hotel_id:

                    # Attempt to fetch the specific hotel by ID

                    try:

                        fetch_button = st.button("Fetch Hotel Details")

                        if fetch_button:

                            with st.spinner("Fetching hotel details..."):

                                response = requests.get(f"{API_URL}/hotels/{hotel_id}")

                                if response.status_code == 200:

                                    hotel_data = response.json()

                                    if isinstance(hotel_data, list) and len(hotel_data) > 0:
                                        hotel_data = hotel_data[0]  # Take the first item if it's a list

                                    st.success(f"Found hotel: {hotel_data.get('hotel_name', 'Unknown')}")

                                    st.session_state.current_hotel = hotel_data


                                else:

                                    st.error(f"Error fetching hotel: {response.text}")

                        # If we have hotel data stored in session state, show the update form

                        if hasattr(st.session_state, 'current_hotel') and st.session_state.current_hotel:

                            hotel_data = st.session_state.current_hotel

                            with st.form("update_hotel_form"):

                                col1, col2 = st.columns(2)

                                with col1:

                                    hotel_name = st.text_input("Hotel Name",

                                                               value=hotel_data.get('hotel_name', ''))

                                    county = st.text_input("County",

                                                           value=hotel_data.get('county', ''))

                                    state = st.text_input("State (2-letter code)",

                                                          value=hotel_data.get('state', ''),

                                                          max_chars=2)

                                with col2:

                                    # Convert to float with safe defaults

                                    try:

                                        rating_val = float(hotel_data.get('rating', 3.0))


                                    except (ValueError, TypeError):

                                        rating_val = 3.0

                                    try:

                                        cleanliness_val = float(hotel_data.get('cleanliness', 3.0))


                                    except (ValueError, TypeError):

                                        cleanliness_val = 3.0

                                    try:

                                        service_val = float(hotel_data.get('service', 3.0))


                                    except (ValueError, TypeError):

                                        service_val = 3.0

                                    rating = st.slider("Rating", min_value=1.0, max_value=5.0,

                                                       value=rating_val, step=0.5)

                                    cleanliness = st.slider("Cleanliness", min_value=1.0, max_value=5.0,

                                                            value=cleanliness_val, step=0.5)

                                    service = st.slider("Service", min_value=1.0, max_value=5.0,

                                                        value=service_val, step=0.5)

                                try:

                                    rooms_val = float(hotel_data.get('rooms', 3.0))


                                except (ValueError, TypeError):

                                    rooms_val = 3.0

                                rooms = st.slider("Rooms", min_value=1.0, max_value=5.0,

                                                  value=rooms_val, step=0.5)

                                # Display the hotel ID (read-only)

                                st.info(f"Hotel ID: {hotel_id}")

                                update_button = st.form_submit_button("Update Hotel")

                                if update_button:

                                    updated_data = {

                                        "hotel_name": hotel_name,

                                        "county": county,

                                        "state": state.upper(),

                                        "rating": rating,

                                        "cleanliness": cleanliness,

                                        "service": service,

                                        "rooms": rooms

                                    }

                                    try:

                                        # Make the update request

                                        with st.spinner("Updating hotel information..."):

                                            update_response = requests.put(f"{API_URL}/hotels/{hotel_id}",

                                                                           json=updated_data)

                                            if update_response.status_code == 200:

                                                st.success(f"Successfully updated hotel: {hotel_name}")

                                                # Display the SQL queries that would be executed

                                                st.subheader("Equivalent SQL Queries:")

                                                hotel_name_update = f"""


                                                -- Update hotel_name1


                                                UPDATE hotel_name1 SET hotel_name = '{hotel_name}' 


                                                WHERE ID = {hotel_id};


                                                """

                                                st.code(hotel_name_update, language="sql")

                                                location_update = f"""


                                                -- Update location


                                                UPDATE location SET county = '{county}', state = '{state.upper()}' 


                                                WHERE ID = {hotel_id};


                                                """

                                                st.code(location_update, language="sql")

                                                rating_update = f"""


                                                -- Update rate


                                                UPDATE rate 


                                                SET rating = {rating}, 


                                                    service = {service}, 


                                                    rooms = {rooms}, 


                                                    cleanliness = {cleanliness}


                                                WHERE ID = {hotel_id};


                                                """

                                                st.code(rating_update, language="sql")

                                                st.balloons()

                                                # Clear the current hotel data after update

                                                if 'current_hotel' in st.session_state:
                                                    del st.session_state.current_hotel


                                            else:

                                                st.error(f"Error updating hotel: {update_response.text}")


                                    except Exception as e:

                                        st.error(f"Error connecting to API: {str(e)}")


                    except Exception as e:

                        st.error(f"Error: {str(e)}")

            elif hotel_operation == "Delete Hotel":

                st.subheader("Delete Hotel by ID")

                # Direct delete by ID input

                hotel_id = st.number_input("Enter Hotel ID to delete", min_value=1, step=1)

                if st.button("Delete Hotel", key="delete_hotel_btn"):

                    # Confirm deletion without additional search

                    if hotel_id:

                        try:

                            # Delete the hotel

                            response = requests.delete(f"{API_URL}/hotels/{hotel_id}")

                            if response.status_code == 200:

                                st.success(f"Successfully deleted hotel with ID: {hotel_id}")

                                # Display the SQL queries that would be executed

                                st.subheader("Equivalent SQL Queries:")

                                # Display the delete queries

                                hotel_name_delete = f"""

                                -- Delete from hotel_name1

                                DELETE FROM hotel_name1 WHERE ID = {hotel_id};

                                """

                                st.code(hotel_name_delete, language="sql")

                                location_delete = f"""

                                -- Delete from location

                                DELETE FROM location WHERE ID = {hotel_id};

                                """

                                st.code(location_delete, language="sql")

                                rating_delete = f"""

                                -- Delete from rate

                                DELETE FROM rate WHERE ID = {hotel_id};

                                """

                                st.code(rating_delete, language="sql")

                        except Exception as e:

                            st.error(f"Error connecting to API: {str(e)}")

                    else:

                        st.warning("Please enter a valid hotel ID")

            # Flight Database Modification
            with mod_tab2:
                st.subheader("Modify Flight Data")

                flight_operation = st.radio(
                    "Select Operation",
                    ["Add New Flight", "Update Flight", "Delete Flight"],
                    horizontal=True
                )

                if flight_operation == "Add New Flight":
                    # Choose between single and multiple flight addition
                    add_mode = st.radio(
                        "Select Add Mode",
                        ["Add Single Flight", "Add Multiple Flights"],
                        horizontal=True
                    )

                    if add_mode == "Add Single Flight":
                        # Keep the original single flight addition form
                        with st.form("add_flight_form"):
                            col1, col2 = st.columns(2)
                            with col1:
                                original_id = st.text_input("Original ID (unique identifier)", key="add_flight_id")
                                starting_airport = st.text_input("Starting Airport Code", key="add_starting")
                                destination_airport = st.text_input("Destination Airport Code", key="add_destination")
                            with col2:
                                airline_name = st.text_input("Airline Name", key="add_airline")
                                total_fare = st.number_input("Total Fare ($)", min_value=0.0, value=100.0,
                                                             format="%.2f",
                                                             key="add_fare")
                                trip_duration = st.number_input("Trip Duration (minutes)", min_value=0, value=180,
                                                                key="add_duration")

                            submit_button = st.form_submit_button("Add Flight")

                            if submit_button:
                                if not original_id or not starting_airport or not destination_airport:
                                    st.error(
                                        "Please fill in all required fields (Original ID, Starting Airport, Destination Airport)")
                                else:
                                    # Need to add both flight and segment
                                    flight_data = {
                                        "originalId": original_id,
                                        "startingAirport": starting_airport.upper(),
                                        "destinationAirport": destination_airport.upper(),
                                        "totalFare": total_fare,
                                        "totalTripDuration": trip_duration
                                    }

                                    segment_data = {
                                        "originalId": original_id,
                                        "segmentsAirlineName": airline_name
                                    }

                                    try:
                                        # First add flight
                                        flight_response = requests.post(f"{API_URL}/flights", json=flight_data)
                                        if flight_response.status_code == 200 or flight_response.status_code == 201:
                                            # Then add segment
                                            segment_response = requests.post(f"{API_URL}/segments", json=segment_data)
                                            if segment_response.status_code == 200 or segment_response.status_code == 201:
                                                st.success(
                                                    f"Successfully added flight from {starting_airport} to {destination_airport}")

                                                # Display the MongoDB queries instead of raw data
                                                st.subheader("MongoDB Insert Queries:")

                                                # Flight insert query
                                                flight_insert_query = f"""
                                                        db.flights.insertOne({{
                                                            originalId: "{original_id}",
                                                            startingAirport: "{starting_airport.upper()}",
                                                            destinationAirport: "{destination_airport.upper()}",
                                                            totalFare: {total_fare},
                                                            totalTripDuration: {trip_duration}
                                                        }})
                                                        """
                                                st.code(flight_insert_query, language="javascript")

                                                # Segment insert query
                                                segment_insert_query = f"""
                                                        db.flights_segments.insertOne({{
                                                            originalId: "{original_id}",
                                                            segmentsAirlineName: "{airline_name}"
                                                        }})
                                                        """
                                                st.code(segment_insert_query, language="javascript")

                                                st.balloons()
                                            else:
                                                st.error(f"Error adding segment: {segment_response.text}")
                                        else:
                                            st.error(f"Error adding flight: {flight_response.text}")
                                    except Exception as e:
                                        st.error(f"Error connecting to API: {str(e)}")

                    else:  # Add multiple flights
                        st.markdown("### Add Multiple Flights")

                        # Input the number of flights to add
                        num_flights = st.number_input("Number of flights to add", min_value=2, max_value=10, value=3)

                        # Create form
                        with st.form("add_multiple_flights_form"):
                            # List to store all flight data
                            all_flights_data = []

                            # Create input fields for each flight
                            for i in range(num_flights):
                                st.markdown(f"### Flight #{i + 1}")

                                col1, col2 = st.columns(2)
                                with col1:
                                    original_id = st.text_input(f"Original ID #{i + 1}", key=f"id_{i}")
                                    starting_airport = st.text_input(f"Starting Airport Code #{i + 1}",
                                                                     key=f"start_{i}")
                                    destination_airport = st.text_input(f"Destination Airport Code #{i + 1}",
                                                                        key=f"dest_{i}")
                                with col2:
                                    airline_name = st.text_input(f"Airline Name #{i + 1}", key=f"airline_{i}")
                                    total_fare = st.number_input(f"Total Fare ($) #{i + 1}", min_value=0.0, value=100.0,
                                                                 format="%.2f", key=f"fare_{i}")
                                    trip_duration = st.number_input(f"Trip Duration (minutes) #{i + 1}", min_value=0,
                                                                    value=180, key=f"duration_{i}")

                                # Add flight data to the list
                                all_flights_data.append({
                                    "original_id": original_id,
                                    "starting_airport": starting_airport,
                                    "destination_airport": destination_airport,
                                    "airline_name": airline_name,
                                    "total_fare": total_fare,
                                    "trip_duration": trip_duration
                                })

                                # Add separator (except after the last flight)
                                if i < num_flights - 1:
                                    st.markdown("---")

                            # Submit button
                            submit_button = st.form_submit_button("Add All Flights")

                            if submit_button:
                                # Validate data for each flight
                                valid_flights = []
                                invalid_indices = []

                                for i, flight in enumerate(all_flights_data):
                                    # Check required fields
                                    if flight["original_id"] and flight["starting_airport"] and flight[
                                        "destination_airport"]:
                                        valid_flights.append(flight)
                                    else:
                                        invalid_indices.append(i + 1)

                                if invalid_indices:
                                    # Show flights with missing required fields
                                    st.error(
                                        f"Flights #{', #'.join(map(str, invalid_indices))} are missing required fields")

                                if valid_flights:
                                    # Display MongoDB insertMany queries that would be executed
                                    st.subheader("Equivalent MongoDB Queries:")

                                    # Prepare the insertMany data arrays
                                    mongo_flights_data = []
                                    mongo_segments_data = []

                                    for flight in valid_flights:
                                        mongo_flights_data.append({
                                            "originalId": flight["original_id"],
                                            "startingAirport": flight["starting_airport"].upper(),
                                            "destinationAirport": flight["destination_airport"].upper(),
                                            "totalFare": flight["total_fare"],
                                            "totalTripDuration": flight["trip_duration"]
                                        })

                                        mongo_segments_data.append({
                                            "originalId": flight["original_id"],
                                            "segmentsAirlineName": flight["airline_name"]
                                        })

                                    # Format the insertMany queries
                                    flights_insert_query = "db.flights.insertMany(" + json.dumps(mongo_flights_data,
                                                                                                 indent=2) + ")"
                                    segments_insert_query = "db.flights_segments.insertMany(" + json.dumps(
                                        mongo_segments_data, indent=2) + ")"

                                    st.code(flights_insert_query, language="javascript")
                                    st.code(segments_insert_query, language="javascript")

                                    # Show progress bar
                                    progress_bar = st.progress(0)
                                    status_placeholder = st.empty()

                                    # Track addition results
                                    success_count = 0
                                    failed_flights = []

                                    # Add flights one by one
                                    for i, flight in enumerate(valid_flights):
                                        try:
                                            # Update progress bar and status text
                                            progress = (i + 1) / len(valid_flights)
                                            progress_bar.progress(progress)
                                            status_placeholder.text(
                                                f"Adding flight {i + 1} of {len(valid_flights)}: {flight['starting_airport']} to {flight['destination_airport']}")

                                            # Prepare flight data
                                            flight_data = {
                                                "originalId": flight["original_id"],
                                                "startingAirport": flight["starting_airport"].upper(),
                                                "destinationAirport": flight["destination_airport"].upper(),
                                                "totalFare": flight["total_fare"],
                                                "totalTripDuration": flight["trip_duration"]
                                            }

                                            segment_data = {
                                                "originalId": flight["original_id"],
                                                "segmentsAirlineName": flight["airline_name"]
                                            }

                                            # First add flight
                                            flight_response = requests.post(f"{API_URL}/flights", json=flight_data)

                                            if flight_response.status_code == 200 or flight_response.status_code == 201:
                                                # Then add segment
                                                segment_response = requests.post(f"{API_URL}/segments",
                                                                                 json=segment_data)

                                                if segment_response.status_code == 200 or segment_response.status_code == 201:
                                                    success_count += 1
                                                else:
                                                    failed_flights.append({
                                                        "route": f"{flight['starting_airport']} to {flight['destination_airport']}",
                                                        "error": f"Error adding segment: {segment_response.text}"
                                                    })
                                            else:
                                                failed_flights.append({
                                                    "route": f"{flight['starting_airport']} to {flight['destination_airport']}",
                                                    "error": f"Error adding flight: {flight_response.text}"
                                                })

                                        except Exception as e:
                                            failed_flights.append({
                                                "route": f"{flight['starting_airport']} to {flight['destination_airport']}",
                                                "error": str(e)
                                            })

                                    # Clear status text
                                    status_placeholder.empty()

                                    # Show addition results
                                    if success_count > 0:
                                        st.success(
                                            f"Successfully added {success_count} of {len(valid_flights)} flights")

                                        # Show balloons effect if all flights were added successfully
                                        if success_count == len(valid_flights):
                                            st.balloons()

                                    # Show failed flights
                                    if failed_flights:
                                        st.error(f"Failed to add {len(failed_flights)} flights")

                                        with st.expander("Show failed flights"):
                                            for failed_flight in failed_flights:
                                                st.markdown(f"**{failed_flight['route']}**: {failed_flight['error']}")


                elif flight_operation == "Update Flight":
                    # Search for flight by ID
                    search_col1, search_col2 = st.columns([3, 1])
                    with search_col1:
                        original_id = st.text_input("Enter Flight Original ID", key="flight_search_id")
                    with search_col2:
                        search_button = st.button("Search", key="flight_search_btn")

                    if search_button and original_id:
                        try:
                            # Search for flight
                            flight_response = requests.get(f"{API_URL}/flights/id/{original_id}")
                            # Search for segment
                            segment_response = requests.get(f"{API_URL}/segments/id/{original_id}")

                            if flight_response.status_code == 200:
                                flight = flight_response.json()
                                if flight:
                                    st.session_state.found_flight = flight[0] if isinstance(flight, list) else flight
                                    st.success(f"Found flight with ID: {original_id}")

                                    # Display the found flight
                                    st.subheader("Found Flight:")
                                    st.json(st.session_state.found_flight)

                                    # Store segment if found
                                    if segment_response.status_code == 200:
                                        segment = segment_response.json()
                                        if segment:
                                            st.session_state.found_segment = segment[0] if isinstance(segment,
                                                                                                      list) else segment

                                            # Display the found segment
                                            st.subheader("Found Segment:")
                                            st.json(st.session_state.found_segment)
                                else:
                                    st.warning(f"No flight found with ID: {original_id}")
                            else:
                                st.error(f"Error searching for flight: {flight_response.text}")
                        except Exception as e:
                            st.error(f"Error connecting to API: {str(e)}")

                    # If flight was found, show update form
                    if hasattr(st.session_state, 'found_flight'):
                        flight = st.session_state.found_flight
                        segment = getattr(st.session_state, 'found_segment', None)

                        with st.form("update_flight_form"):
                            col1, col2 = st.columns(2)
                            with col1:
                                starting_airport = st.text_input("Starting Airport Code",
                                                                 value=flight.get('startingAirport', ''))
                                destination_airport = st.text_input("Destination Airport Code",
                                                                    value=flight.get('destinationAirport', ''))
                            with col2:
                                airline_name = st.text_input("Airline Name", value=segment.get('segmentsAirlineName',
                                                                                               '') if segment else '')
                                total_fare = st.number_input("Total Fare ($)", min_value=0.0,
                                                             value=float(flight.get('totalFare', 100.0)), format="%.2f")

                            trip_duration = st.number_input("Trip Duration (minutes)", min_value=0,
                                                            value=int(flight.get('totalTripDuration', 180)))

                            update_button = st.form_submit_button("Update Flight")

                            if update_button:
                                # Update both flight and segment
                                updated_flight = {
                                    "startingAirport": starting_airport.upper(),
                                    "destinationAirport": destination_airport.upper(),
                                    "totalFare": total_fare,
                                    "totalTripDuration": trip_duration
                                }

                                updated_segment = {
                                    "segmentsAirlineName": airline_name
                                }

                                try:
                                    # Update flight
                                    flight_response = requests.put(f"{API_URL}/flights/id/{original_id}",
                                                                   json=updated_flight)
                                    success_flight = flight_response.status_code == 200

                                    # Update segment if it exists
                                    success_segment = True
                                    if segment:
                                        segment_response = requests.put(f"{API_URL}/segments/id/{original_id}",
                                                                        json=updated_segment)
                                        success_segment = segment_response.status_code == 200

                                    if success_flight and success_segment:
                                        st.success(
                                            f"Successfully updated flight from {starting_airport} to {destination_airport}")

                                        # Display the MongoDB update queries
                                        st.subheader("MongoDB Update Queries:")

                                        # Flight update query
                                        flight_update_query = f"""
                                                    db.flights.updateOne(
                                                        {{ originalId: "{original_id}" }},
                                                        {{ $set: {{
                                                            startingAirport: "{starting_airport.upper()}",
                                                            destinationAirport: "{destination_airport.upper()}",
                                                            totalFare: {total_fare},
                                                            totalTripDuration: {trip_duration}
                                                        }} }}
                                                    )
                                                    """
                                        st.code(flight_update_query, language="javascript")

                                        # Segment update query (if exists)
                                        if segment:
                                            segment_update_query = f"""
                                                        db.flights_segments.updateOne(
                                                            {{ originalId: "{original_id}" }},
                                                            {{ $set: {{
                                                                segmentsAirlineName: "{airline_name}"
                                                            }} }}
                                                        )
                                                        """
                                            st.code(segment_update_query, language="javascript")

                                        # Clear stored data
                                        if 'found_flight' in st.session_state:
                                            del st.session_state.found_flight
                                        if 'found_segment' in st.session_state:
                                            del st.session_state.found_segment
                                    else:
                                        if not success_flight:
                                            st.error(f"Error updating flight: {flight_response.text}")
                                        if not success_segment:
                                            st.error(f"Error updating segment: {segment_response.text}")
                                except Exception as e:
                                    st.error(f"Error connecting to API: {str(e)}")

                elif flight_operation == "Delete Flight":
                    st.markdown("### Delete Flight by ID")

                    # Single input and delete button
                    original_id = st.text_input("Enter Flight ID to delete", key="flight_delete_id")

                    if st.button("🗑️ Delete Flight", key="delete_flight_btn"):
                        try:
                            # Immediate deletion without search step
                            with st.spinner('Deleting flight...'):
                                # Delete both flight and segment
                                flight_response = requests.delete(f"{API_URL}/flights/id/{original_id}")
                                segment_response = requests.delete(f"{API_URL}/segments/id/{original_id}")

                                success_flight = flight_response.status_code == 200
                                success_segment = segment_response.status_code == 200

                                if success_flight or success_segment:
                                    st.success("✅ Flight successfully deleted")

                                    # Display the executed MongoDB queries
                                    st.subheader("Executed MongoDB Queries:")

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Flight Deletion:**")
                                        st.code(
                                            f"db.flights.deleteOne({{ originalId: \"{original_id}\" }})",
                                            language="javascript"
                                        )

                                    with col2:
                                        st.markdown("**Segment Deletion:**")
                                        st.code(
                                            f"db.flights_segments.deleteOne({{ originalId: \"{original_id}\" }})",
                                            language="javascript"
                                        )

                                    st.balloons()
                                else:
                                    if not success_flight:
                                        st.error(f"Failed to delete flight: {flight_response.text}")
                                    if not success_segment:
                                        st.warning(
                                            f"Flight deleted but segment deletion failed: {segment_response.text}")
                        except Exception as e:
                            st.error(f"Deletion error: {str(e)}")

    st.sidebar.markdown("## Database Schema Reference")
    with st.sidebar.expander("Click to view schema details"):
        st.markdown("### Hotel Database Schema:")
        st.markdown("- **hotel_complete_view**(ID, hotel_name, county, state)")
        st.markdown("- **rate_complete_view**(ID, rating, service, rooms, cleanliness)")

        st.markdown("### Flight Database Schema:")
        st.markdown(
            "- **flights**(_id, originalId, startingAirport, destinationAirport, totalFare, totalTripDuration, ...)")
        st.markdown("- **flights_segments**(_id, originalId, segmentsAirlineName, ...)")

# Run the main application
if __name__ == "__main__":
    main()