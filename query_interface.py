import streamlit as st
import pandas as pd
import os
import logging
import re
import requests
import json
import socket

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a consistent default limit across the application
DEFAULT_LIMIT = 20

# Environment-aware API URL
def get_api_url():
    """
    Get the appropriate API URL based on environment
    """
    # Default Docker service name
    backend_host = "backend"
    backend_port = "8000"
    
    # Environment variable takes precedence if set
    api_url = os.environ.get("API_URL")
    if api_url:
        return api_url
    
    # Check if backend hostname is resolvable (in Docker network)
    try:
        socket.gethostbyname(backend_host)
        return f"http://{backend_host}:{backend_port}"
    except socket.gaierror:
        # Not in Docker or hostname not resolvable, use localhost
        return f"http://localhost:{backend_port}"

# Use this function to get the API URL
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
            """, "LOCATION_DB_PATH"
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
            """, "RATE_DB_PATH"
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
            """, "LOCATION_DB_PATH"
    
    # Sample rows exploration
    if "sample" in query_lower:
        table_match = re.search(r'from\s+(\w+)', query_lower)
        if table_match:
            table_name = table_match.group(1)
            return f"SELECT * FROM {table_name} LIMIT {DEFAULT_LIMIT}", "LOCATION_DB_PATH" if "location" in table_name.lower() or "hotel" in table_name.lower() else "RATE_DB_PATH"
    
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
        """, "LOCATION_DB_PATH"
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
        """, "LOCATION_DB_PATH"
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
        """, "LOCATION_DB_PATH"
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
        """, "LOCATION_DB_PATH"
    
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
    """, "LOCATION_DB_PATH"

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
    """
    Execute a query based on natural language input
    """
    query_lower = query_text.lower()
    
    # Determine if this is a flight or hotel query
    is_flight_query = any(term in query_lower for term in ["flight", "airline", "airport", "plane"])
    
    if is_flight_query:
        # Process as a flight query
        mongo_query, query_type, params = process_flight_nl_query(query_text)
        
        # Ensure params has a limit
        if "limit" not in params:
            params["limit"] = DEFAULT_LIMIT
        
        # Determine the correct API endpoint based on query type
        if query_type == "by_airports":
            endpoint = f"{API_URL}/flights/airports"
        elif query_type == "by_airline":
            endpoint = f"{API_URL}/flights/airline"
        else:
            endpoint = f"{API_URL}/flights"
        
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
        
        return {
            "type": "mongodb",
            "query": mongo_query,
            "params": params,
            "results": results,
            "query_type": query_type
        }
    else:
        # Process as a hotel query
        sql_query, params = process_hotel_nl_query(query_text)
        
        # Ensure params has a limit
        if "limit" not in params:
            params["limit"] = DEFAULT_LIMIT
        
        # Execute API call
        try:
            response = requests.get(f"{API_URL}/hotels", params=params)
            if response.status_code == 200:
                hotels = response.json()
                # Convert to DataFrame for consistent handling
                hotels_df = pd.DataFrame(hotels) if hotels else pd.DataFrame()
                logger.info(f"Received {len(hotels)} hotel results")
            else:
                hotels_df = pd.DataFrame()
                logger.error(f"API error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error calling API: {str(e)}")
            hotels_df = pd.DataFrame()
        
        return {
            "type": "sql",
            "query": sql_query,
            "params": params,
            "results": hotels_df
        }

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
                        st.session_state.query_input = example
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
                        st.session_state.query_input = example
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
                        st.session_state.query_input = example
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
                        st.session_state.query_input = example
                        st.rerun()  # Changed from st.experimental_rerun()

# Main function for Streamlit display
def display_natural_language_query():
    """
    Main function for Streamlit display (Fixed version)
    """
    st.title("Natural Language Database Query")
    
    # Initialize session state for query input if not already set
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ''
    
    # Add example queries section at the top
    add_example_queries_section()
    
    # Set up tabs for different query interfaces
    tab1, tab2, tab3 = st.tabs(["Query Interface", "Hotel Database", "Flight Database"])
    with tab1:
        st.markdown("### Enter your database query in natural language")
    
        # Create two columns - one for the text area and one for the button
        input_col, button_col = st.columns([4, 1])
        with input_col:
            query_input = st.text_area(
                "Query input", 
                value=st.session_state.get('query_input', ''),
                placeholder="Example: Show me hotels in Orange County with ratings above 4.5",
                height=100,
                label_visibility="collapsed"  # This fixes the empty label warning
            )
    
        with button_col:
            execute_button = st.button("Execute Query", use_container_width=True)
    
    # Display current API URL (useful for debugging)
    st.sidebar.markdown(f"**Current API URL:** {API_URL}")
    
    # Quick command buttons
    st.markdown("### Quick Commands")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Show Database Schema"):
            st.session_state.query_input = "What tables exist in the hotel database?"
            st.rerun()  # Changed from st.experimental_rerun()
            
    with col2:
        if st.button("Find Top Hotels"):
            st.session_state.query_input = "Show me the top 10 highest-rated hotels"
            st.rerun()  # Changed from st.experimental_rerun()
            
    with col3:
        if st.button("Find Cheapest Flights"):
            st.session_state.query_input = "Find the cheapest flights from LAX to JFK"
            st.rerun()  # Changed from st.experimental_rerun()
    
    # Process query when either input is submitted OR button is clicked
    if query_input and (execute_button or 'last_query' not in st.session_state or st.session_state.last_query != query_input):
        # Update the last query to avoid re-processing
        st.session_state.last_query = query_input
        
        # Process query
        if query_input:
            try:
                # Log query and API URL for debugging
                logger.info(f"Processing query: {query_input}")
                logger.info(f"Using API URL: {API_URL}")
                
                # Execute query
                result = execute_query(query_input)
                
                # Create a clear visual separation with a divider
                st.markdown("---")
                st.markdown("## Query Results")
                
                # Display query type with more context
                if result["type"] == "sql":
                    st.markdown("### Database: SQL (Hotel Database)")
                else:
                    st.markdown("### Database: MongoDB (Flight Database)")
                
                # Display generated query in a well-defined expandable section
                with st.expander("View Generated Query", expanded=False):
                    st.markdown("The system translated your natural language query into the following database query:")
                    if result["type"] == "sql":
                        st.code(result["query"], language="sql")
                    else:
                        st.code(result["query"], language="javascript")
                    
                    # Also show the API parameters that were actually used
                    st.markdown("### API Parameters Used:")
                    st.json(result["params"])
                
                # Check if result has error
                if isinstance(result["results"], dict) and "error" in result["results"]:
                    st.error(f"Error: {result['results']['error']}")
                    
                # For data modification queries (detection remains the same)
                elif ("INSERT" in result["query"] or 
                      "UPDATE" in result["query"] or 
                      "DELETE" in result["query"] or 
                      "insertOne" in result["query"] or 
                      "updateOne" in result["query"] or 
                      "deleteOne" in result["query"]):
                    st.warning("This is a data modification query. For safety, it was not executed in this demo.")
                    st.info("In a production environment, appropriate validation and execution would be implemented.")
                
                # For SQL query results (now in DataFrame format)
                elif result["type"] == "sql" and isinstance(result["results"], pd.DataFrame):
                    st.markdown("### Query Results:")
                    if not result["results"].empty:
                        # Use our custom display function ONLY ONCE
                        display_dataframe_with_limit(result["results"], result["params"].get("limit", DEFAULT_LIMIT))
                    else:
                        st.info("No results found matching your criteria.")
                    
                # For MongoDB query results (list of dicts)
                elif result["type"] == "mongodb" and isinstance(result["results"], list):
                    # Get specific query information based on query type
                    query_type = result.get("query_type", "")
                    
                    if query_type == "flight_route":
                        # This is a route-specific query
                        from_airport = result["params"].get("starting", "")
                        to_airport = result["params"].get("destination", "")
                        
                        # Create specific header
                        st.markdown(f"### Flight Information: {from_airport} to {to_airport}")
                        
                        # Check if we have any results
                        if not result["results"]:
                            st.warning(f"No direct flights found from {from_airport} to {to_airport}.")
                        
                    elif query_type == "airline":
                        # This is an airline-specific query
                        airline = result["params"].get("airline", "")
                        st.markdown(f"### Flights operated by {airline}")
                        
                        # Check if we have any results
                        if not result["results"]:
                            st.warning(f"No flights found operated by {airline}.")
                    
                    elif query_type == "flight_price":
                        # This is a price-based query
                        max_price = result["params"].get("max_price", "")
                        st.markdown(f"### Flights under ${max_price}")
                        
                        # Check if we have any results
                        if not result["results"]:
                            st.warning(f"No flights found under ${max_price}.")
                    
                    else:
                        # Generic flight information
                        st.markdown("### Flight Information")
                    
                    # Create DataFrame for any results we have - ONLY ONCE
                    if result["results"]:
                        try:
                            df = pd.DataFrame(result["results"])
                            
                            # Show the most relevant columns for flights
                            if "startingAirport" in df.columns:
                                cols = ["startingAirport", "destinationAirport", "totalFare", "isNonStop"]
                                display_cols = [col for col in cols if col in df.columns]
                                # Use custom display function ONLY ONCE
                                display_dataframe_with_limit(df[display_cols] if display_cols else df, result["params"].get("limit", DEFAULT_LIMIT))
                            else:
                                # Use custom display function ONLY ONCE
                                display_dataframe_with_limit(df, result["params"].get("limit", DEFAULT_LIMIT))
                        except Exception as e:
                            st.error(f"Error displaying results: {str(e)}")
                            st.json(result["results"])
                    else:
                        st.info("No results found matching your criteria.")
                else:
                    # Fallback display
                    st.markdown("### Raw Results:")
                    st.json(result["results"])
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                logger.error(f"Error processing query '{query_input}': {str(e)}")
    
    with tab2:
        st.markdown("## Hotel Database")
        
        # Create subtabs for basic and advanced queries
        hotel_tab1, hotel_tab2 = st.tabs(["Basic Queries", "Advanced Queries"])
        
        with hotel_tab1:
            st.markdown("### Basic Hotel Queries")
            st.markdown("Try these example queries to explore the hotel database:")
            
            # Schema Exploration
            with st.expander("Schema Exploration", expanded=True):
                schema_examples = [
                    "What tables exist in the hotel database?",
                    "Show me the structure of the hotel_name1 table",
                    "Give me sample rows from the location table"
                ]
                
                for i, example in enumerate(schema_examples):
                    if st.button(f"{example}", key=f"schema_{i}"):
                        st.session_state.query_input = example
                        st.rerun()  # Changed from st.experimental_rerun()
            
            # Basic Filter Queries
            with st.expander("Basic Filter Queries", expanded=True):
                basic_examples = [
                    "Show me all hotels in Orange County",
                    "List hotels with ratings above 4.5",
                    "What are the top 10 highest-rated hotels?",
                    "Show hotels in California with good service ratings"
                ]
                
                for i, example in enumerate(basic_examples):
                    if st.button(f"{example}", key=f"basic_{i}"):
                        st.session_state.query_input = example
                        st.rerun()  # Changed from st.experimental_rerun()
        
        with hotel_tab2:
            st.markdown("### Advanced Hotel Queries")
            
            # Aggregation Queries
            with st.expander("Aggregation Queries", expanded=True):
                agg_examples = [
                    "Show the average rating of hotels in each state",
                    "Count the number of hotels in each state",
                    "Find the state with the highest average hotel rating",
                    "Show the top 5 counties with the most hotels"
                ]
                
                for i, example in enumerate(agg_examples):
                    if st.button(f"{example}", key=f"agg_{i}"):
                        st.session_state.query_input = example
                        st.rerun()  # Changed from st.experimental_rerun()
            
            # Complex Filtering
            with st.expander("Complex Filtering", expanded=True):
                complex_examples = [
                    "Find hotels with ratings above 4.5 and excellent cleanliness",
                    "Show hotels in California where service rating is higher than cleanliness rating"
                ]
                
                for i, example in enumerate(complex_examples):
                    if st.button(f"{example}", key=f"complex_{i}"):
                        st.session_state.query_input = example
                        st.rerun()  # Changed from st.experimental_rerun()
    
    with tab3:
        st.markdown("## Flight Database")
        
        # Create subtabs for basic and advanced queries
        flight_tab1, flight_tab2 = st.tabs(["Basic Queries", "Advanced Queries"])
        
        with flight_tab1:
            st.markdown("### Basic Flight Queries")
            st.markdown("Try these example queries to explore the flight database:")
            
            # Schema Exploration
            with st.expander("Schema Exploration", expanded=True):
                mongo_schema_examples = [
                    "What collections exist in the flight database?",
                    "Show me a sample document from flights collection",
                    "What fields are in the flights_segments collection?"
                ]
                
                for i, example in enumerate(mongo_schema_examples):
                    if st.button(f"{example}", key=f"mongo_schema_{i}"):
                        st.session_state.query_input = example
                        st.rerun()  # Changed from st.experimental_rerun()
            
            # Basic Find Queries
            with st.expander("Basic Find Queries", expanded=True):
                find_examples = [
                    "Find flights from LAX to JFK",
                    "Show flights operated by Delta Airlines",
                    "List all non-stop flights under $300",
                    "Show me the cheapest flights to Chicago"
                ]
                
                for i, example in enumerate(find_examples):
                    if st.button(f"{example}", key=f"find_{i}"):
                        st.session_state.query_input = example
                        st.rerun()  # Changed from st.experimental_rerun()
        
        with flight_tab2:
            st.markdown("### Advanced Flight Queries")
            
            # Aggregation Queries
            with st.expander("Aggregation Queries", expanded=True):
                mongo_agg_examples = [
                    "What's the average price of flights for each airline?",
                    "Which are the most popular flight routes?",
                    "Find the cheapest airline for flights to New York",
                    "Show average flight prices by destination"
                ]
                
                for i, example in enumerate(mongo_agg_examples):
                    if st.button(f"{example}", key=f"mongo_agg_{i}"):
                        st.session_state.query_input = example
                        st.rerun()  # Changed from st.experimental_rerun()
            
            # Join/Lookup Queries
            with st.expander("Join/Lookup Queries", expanded=True):
                lookup_examples = [
                    "Show flights with their detailed segment information",
                    "Find Delta flights with their segment details",
                    "Connect flight information with airline segments"
                ]
                
                for i, example in enumerate(lookup_examples):
                    if st.button(f"{example}", key=f"lookup_{i}"):
                        st.session_state.query_input = example
                        st.rerun()  # Changed from st.experimental_rerun()

# If this file is run directly, display the interface
if __name__ == "__main__":
    display_natural_language_query()