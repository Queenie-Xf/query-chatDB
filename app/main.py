from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import os
import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from .sql_agent import (
    get_all_reviews,
    get_reviews_by_county,
    get_reviews_by_state,       
    find_hotels_with_min_rating,
    get_connection,
    LOCATION_DB_PATH,
    RATE_DB_PATH
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import database functions
from app.mongo_agent import get_all_flights, get_flights_by_airports, get_flights_by_airline, find_with_projection, search_flights
from app.sql_agent import get_all_reviews, get_reviews_by_county, get_reviews_by_state

# Define models for request validation
from pydantic import BaseModel
from typing import Optional

# Define a consistent default limit across the application
DEFAULT_LIMIT = 20

app = FastAPI(title="Travel Database API")

# Add timeout middleware
class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        try:
            # Set a timeout for this request
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        except Exception as e:
            process_time = time.time() - start_time
            if process_time > 4.5:  # If taking too long
                # Return a timeout response instead of hanging
                return JSONResponse(
                    status_code=504,
                    content={"detail": "Request timeout - database operation taking too long"}
                )
            # Re-raise other exceptions
            raise

app.add_middleware(TimeoutMiddleware)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Travel Database API"}

# Flight endpoints with improved error handling
@app.get("/flights", response_model=List[Dict[str, Any]])
def get_flights(
    limit: int = DEFAULT_LIMIT,  # Changed from 10 to DEFAULT_LIMIT
    max_price: Optional[float] = None,
    starting: Optional[str] = None,
    destination: Optional[str] = None,
    airline: Optional[str] = None
):
    """
    Flexible flight search with support for price and other filters
    """
    try:
        query_params = {
            "limit": limit,
            "max_price": max_price,
            "starting": starting,
            "destination": destination,
            "airline": airline
        }
        # Remove None values
        clean_params = {k: v for k, v in query_params.items() if v is not None}
        # Make sure search_flights properly uses the limit parameter
        results = search_flights(clean_params)
        
        # Ensure limit is applied at API level as well
        if isinstance(results, list) and len(results) > limit:
            results = results[:limit]
            
        logger.info(f"Found {len(results)} flights (limited to {limit})")
        return results
    except Exception as e:
        logger.error(f"Error retrieving flights: {e}")
        return []


@app.get("/flights/airports", response_model=List[Dict[str, Any]])
def get_flights_by_airport(
    starting: str = Query(..., description="Starting airport code"),
    destination: str = Query(..., description="Destination airport code"),
    limit: int = DEFAULT_LIMIT  # Added limit parameter with DEFAULT_LIMIT
):
    """
    Get flights between specific airports
    """
    try:
        flights = get_flights_by_airports(starting, destination)
        
        # Apply limit at API level
        if flights and len(flights) > limit:
            flights = flights[:limit]
            
        logger.info(f"Found {len(flights)} flights from {starting} to {destination} (limited to {limit})")
        return flights
    except Exception as e:
        logger.error(f"Error retrieving flights: {e}")
        # Return empty list instead of raising exception
        return []

@app.get("/flights/airline", response_model=List[Dict[str, Any]])
def get_flights_by_airline_name(
    airline: str = Query(..., description="Airline name"),
    limit: int = DEFAULT_LIMIT  # Added limit parameter with DEFAULT_LIMIT
):
    """
    Get flights operated by a specific airline
    """
    try:
        flights = get_flights_by_airline(airline)
        
        # Apply limit at API level
        if flights and len(flights) > limit:
            flights = flights[:limit]
            
        logger.info(f"Found {len(flights)} flights operated by {airline} (limited to {limit})")
        return flights
    except Exception as e:
        logger.error(f"Error retrieving flights: {e}")
        # Return empty list instead of raising exception
        return []

@app.get("/flights/segments", response_model=List[Dict[str, Any]])
def get_flight_segments(limit: int = DEFAULT_LIMIT):  # Changed from 10 to DEFAULT_LIMIT
    """
    Get flight segments data
    """
    try:
        segments = find_with_projection("segments", {}, None, limit)
        logger.info(f"Found {len(segments)} flight segments (limited to {limit})")
        return segments
    except Exception as e:
        logger.error(f"Error retrieving flight segments: {e}")
        # Return empty list instead of raising exception
        return []

# Hotel endpoints

@app.get("/hotels/county/{county}", response_model=List[Dict[str, Any]])
def get_hotels_by_county(
    county: str,
    limit: int = DEFAULT_LIMIT  # Added limit parameter with DEFAULT_LIMIT
):
    """
    Get hotel reviews for a specific county
    """
    try:
        logger.info(f"Getting hotels for county: {county}")
        hotels = get_reviews_by_county(county)
        
        # Convert tuple data to dictionaries
        result = []
        for hotel in hotels:
            result.append({
                "rating": hotel[0],
                "service": hotel[2],
                "rooms": hotel[3],
                "cleanliness": hotel[4],
                "value": hotel[5],
                "hotel_name": hotel[6],
                "county": hotel[7],
                "state": hotel[8]
            })
        
        # Apply limit
        if len(result) > limit:
            result = result[:limit]
        
        logger.info(f"Found {len(result)} hotels in county: {county} (limited to {limit})")
        return result
    except Exception as e:
        logger.error(f"Error retrieving hotels for county {county}: {e}")
        # Return empty list instead of raising exception
        return []

@app.get("/hotels/state/{state}", response_model=List[Dict[str, Any]])
def get_hotels_by_state(
    state: str,
    limit: int = DEFAULT_LIMIT  # Added limit parameter with DEFAULT_LIMIT
):
    """
    Get hotel reviews for a specific state
    """
    try:
        # Clean up and normalize the state parameter
        clean_state = state.strip()
        # Remove any query parameters that might have been included
        if "?" in clean_state:
            clean_state = clean_state.split("?")[0]
            
        logger.info(f"Getting hotels for state: {clean_state}")
        
        # Get hotel data
        hotels = get_reviews_by_state(clean_state)
        
        # Convert tuple data to dictionaries
        result = []
        for hotel in hotels:
            result.append({
                "rating": hotel[0],
                "service": hotel[2],
                "rooms": hotel[3],
                "cleanliness": hotel[4],
                "value": hotel[5],
                "hotel_name": hotel[6],
                "county": hotel[7],
                "state": hotel[8]
            })
        
        # Apply limit
        if len(result) > limit:
            result = result[:limit]
        
        logger.info(f"Found {len(result)} hotels in state: {clean_state} (limited to {limit})")
        return result
    except Exception as e:
        logger.error(f"Error retrieving hotels for state {state}: {e}")
        # Return empty list instead of raising exception
        return []

# Models for flight and segment data
class FlightModel(BaseModel):
    originalId: str
    startingAirport: str
    destinationAirport: str
    totalFare: float
    totalTripDuration: Optional[int] = None

class SegmentModel(BaseModel):
    originalId: str
    segmentsAirlineName: str

class FlightUpdateModel(BaseModel):
    originalId: Optional[str] = None
    startingAirport: Optional[str] = None
    destinationAirport: Optional[str] = None
    totalFare: Optional[float] = None
    totalTripDuration: Optional[int] = None

class SegmentUpdateModel(BaseModel):
    originalId: Optional[str] = None
    segmentsAirlineName: Optional[str] = None

# Flight CRUD operations
@app.post("/flights", status_code=201)
async def create_flight(flight: FlightModel):
    """Add a new flight to the database"""
    from app.mongo_agent import insert_one

    flight_data = flight.dict()
    result = insert_one("flights", flight_data)

    if result["acknowledged"]:
        return {"success": True, "message": "Flight added successfully", "id": result["inserted_id"]}
    else:
        raise HTTPException(status_code=500, detail="Failed to add flight")

@app.post("/segments", status_code=201)
async def create_segment(segment: SegmentModel):
    """Add a new flight segment to the database"""
    from app.mongo_agent import insert_one

    segment_data = segment.dict()
    result = insert_one("segments", segment_data)

    if result["acknowledged"]:
        return {"success": True, "message": "Segment added successfully", "id": result["inserted_id"]}
    else:
        raise HTTPException(status_code=500, detail="Failed to add segment")


@app.get("/flights/id/{original_id}")
async def get_flight_by_id(original_id: str):
    """Get a flight by its originalId"""
    from app.mongo_agent import find_with_projection
    from bson import ObjectId

    flights = find_with_projection("flights", {"originalId": original_id})

    if not flights:
        try:
            object_id = ObjectId(original_id)
            flights = find_with_projection("flights", {"_id": object_id})
        except:
            pass

    if not flights:
        raise HTTPException(status_code=404, detail="Flight not found")

    return flights

@app.get("/segments/id/{original_id}")
async def get_segment_by_id(original_id: str):
    """Get a flight segment by its originalId"""
    from app.mongo_agent import find_with_projection

    # Query the segments collection
    segments = find_with_projection("segments", {"originalId": original_id})

    if not segments:
        raise HTTPException(status_code=404, detail="Segment not found")

    return segments


@app.delete("/flights/id/{original_id}")
async def delete_flight(original_id: str):
    """Delete a flight by its originalId or _id"""
    from app.mongo_agent import delete_one
    from bson import ObjectId

    # 先尝试按originalId字段删除
    flight_result = delete_one("flights", {"originalId": original_id})

    # 如果没有删除成功，尝试按_id删除
    if flight_result["deleted_count"] == 0:
        try:
            object_id = ObjectId(original_id)
            flight_result = delete_one("flights", {"_id": object_id})
        except:
            pass

    if flight_result["deleted_count"] == 0:
        raise HTTPException(status_code=404, detail="Flight not found")

    return {"success": True, "message": f"Flight with ID {original_id} successfully deleted"}

@app.put("/segments/id/{original_id}")
async def update_segment(original_id: str, segment: SegmentUpdateModel):
    """Update a flight segment by its originalId"""
    from app.mongo_agent import update_one

    # Extract update data, excluding any None values
    update_data = {k: v for k, v in segment.dict().items() if v is not None}

    # Create the update query
    update_query = {"$set": update_data}

    # Update the segment record
    result = update_one("segments", {"originalId": original_id}, update_query)

    # Check if segment was found and updated
    if result["matched_count"] == 0:
        raise HTTPException(status_code=404, detail="Segment not found")

    return {"success": True, "message": f"Segment with ID {original_id} successfully updated"}

@app.delete("/flights/id/{original_id}")
async def delete_flight(original_id: str):
    """Delete a flight by its originalId"""
    from app.mongo_agent import delete_one

    # Delete the flight record
    flight_result = delete_one("flights", {"originalId": original_id})

    # Check if flight was deleted
    if flight_result["deleted_count"] == 0:
        raise HTTPException(status_code=404, detail="Flight not found")

    return {"success": True, "message": f"Flight with ID {original_id} successfully deleted"}

@app.delete("/segments/id/{original_id}")
async def delete_segment(original_id: str):
    """Delete a flight segment by its originalId"""
    from app.mongo_agent import delete_one

    # Delete the segment record
    segment_result = delete_one("segments", {"originalId": original_id})

    # Return success even if no segments were found (they might have been deleted already)
    return {"success": True, "message": f"Segment with ID {original_id} deleted if it existed"}

@app.get("/flights/list")
async def list_all_flights(limit: int = DEFAULT_LIMIT):  # Added limit parameter with DEFAULT_LIMIT
    """List all flights in the database"""
    from app.mongo_agent import find_with_projection

    # Get flights with specified limit
    flights = find_with_projection("flights", {}, limit=limit)

    # Return the list of flights with their IDs
    result = [{"id": flight.get("originalId", "unknown"), 
              "from": flight.get("startingAirport", ""),
              "to": flight.get("destinationAirport", "")} for flight in flights]
    
    logger.info(f"Listed {len(result)} flights (limited to {limit})")
    return result

@app.post("/execute_mongo_query")
async def execute_mongo_query(request: dict):
    """Execute a MongoDB query from the Streamlit app"""
    from app.mongo_agent import find_with_projection
    
    try:
        collection = request.get("collection")
        query_string = request.get("query", "")
        limit = request.get("limit", DEFAULT_LIMIT)  # Use DEFAULT_LIMIT if not specified
        
        if not collection:
            raise HTTPException(status_code=400, detail="Missing collection parameter")
        
        # Execute a simple query based on collection
        if collection == "flights":
            results = find_with_projection("flights", {}, limit=limit)
        elif collection == "segments":
            results = find_with_projection("segments", {}, limit=limit)
        else:
            results = find_with_projection(collection, {}, limit=limit)
        
        logger.info(f"Executed query on {collection}, found {len(results)} results (limited to {limit})")
        return results
        
    except Exception as e:
        logger.error(f"Error executing MongoDB query: {e}")
        # Return empty list instead of raising exception
        return []


@app.get("/hotels", response_model=List[Dict[str, Any]])
def get_hotels(
        county: Optional[str] = None,
        state: Optional[str] = None,
        min_rating: Optional[float] = None,
        limit: Optional[int] = DEFAULT_LIMIT  # Already using DEFAULT_LIMIT
):
    try:
        logger.info(f"Getting hotels with filters: county={county}, state={state}, min_rating={min_rating}, limit={limit}")
        
        # Get base query results
        if county and state:
            hotels = get_reviews_by_county(county)
        elif min_rating is not None:
            hotels = find_hotels_with_min_rating(min_rating)
        elif county:
            hotels = get_reviews_by_county(county)
        elif state:
            hotels = get_reviews_by_state(state)
        else:
            hotels = get_all_reviews()
        
        # Apply limit here - very important!
        if hotels and len(hotels) > limit:
            hotels = hotels[:limit]  # Enforce limit at API level
        
        # Convert to dictionary
        result = []
        for hotel in hotels:
            try:
                hotel_dict = {
                    "ID": hotel[0] if len(hotel) > 0 else None,
                    "hotel_name": hotel[1] if len(hotel) > 1 else None,
                    "rating": hotel[2] if len(hotel) > 2 else None,
                    "service": hotel[3] if len(hotel) > 3 else None,
                    "rooms": hotel[4] if len(hotel) > 4 else None,
                    "cleanliness": hotel[5] if len(hotel) > 5 else None
                }
                hotel_dict = {k: v for k, v in hotel_dict.items() if v is not None}
                result.append(hotel_dict)
            except Exception as e:
                logger.error(f"Error processing hotel record: {e}")
                continue
                
        logger.info(f"Found {len(result)} hotels matching criteria (limited to {limit})")
        return result
    except Exception as e:
        logger.error(f"Error retrieving hotels: {e}")
        return []

@app.get("/debug/hotel_schema")
async def debug_hotel_schema():
    """Debug endpoint to check hotel schema"""
    try:
        # Get a sample hotel from rate_complete_view
        from app.sql_agent import execute_sql_query, RATE_DB_PATH
        import sqlite3  # Add this import

        sample_query = "SELECT * FROM rate_complete_view LIMIT 1"
        sample = execute_sql_query(RATE_DB_PATH, sample_query)

        result = {
            "schema": {
                "columns": [],
                "sample_data": {}
            },
            "tables": {}
        }

        # Get column names if sample data exists
        if sample and len(sample) > 0:
            conn = sqlite3.connect(RATE_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(sample_query)
            result["schema"]["columns"] = [description[0] for description in cursor.description]

            # Add sample data values
            sample_data = sample[0]
            for i, column in enumerate(result["schema"]["columns"]):
                if i < len(sample_data):
                    result["schema"]["sample_data"][column] = sample_data[i]

            conn.close()

        # Get tables and views in both databases
        for db_path, db_name in [(RATE_DB_PATH, "rate_db"), (LOCATION_DB_PATH, "location_db")]:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            # Get views
            cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
            views = [row[0] for row in cursor.fetchall()]

            result["tables"][db_name] = {
                "tables": tables,
                "views": views
            }

            conn.close()

        return result
    except Exception as e:
        return {"error": str(e)}
    
    
class HotelCreateModel(BaseModel):
    hotel_name: str
    county: str
    state: str
    rating: float
    service: float
    rooms: float
    cleanliness: float

@app.post("/hotels", status_code=201)
async def create_hotel(hotel: HotelCreateModel):
    """Add a new hotel to the database"""
    from app.sql_agent import add_hotel
    
    try:
        # Log the hotel creation attempt
        logger.info(f"Attempting to add hotel: {hotel.hotel_name} in {hotel.county}, {hotel.state}")
        
        # Call the add_hotel function from sql_agent.py
        hotel_id = add_hotel(
            hotel_name=hotel.hotel_name,
            county=hotel.county,
            state=hotel.state.upper(),  # Ensure state is uppercase
            rating=hotel.rating,
            service=hotel.service,
            rooms=hotel.rooms,
            cleanliness=hotel.cleanliness
        )
        
        logger.info(f"Successfully added hotel with ID: {hotel_id}")
        
        return {
            "success": True, 
            "message": "Hotel added successfully", 
            "id": hotel_id
        }
        
    except Exception as e:
        # Log the error
        logger.error(f"Error adding hotel: {e}")
        # Raise an HTTP exception
        raise HTTPException(status_code=500, detail=str(e))
# 为更新酒店定义模型
class HotelUpdateModel(BaseModel):
    hotel_name: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    rating: Optional[float] = None
    service: Optional[float] = None
    rooms: Optional[float] = None
    cleanliness: Optional[float] = None

# 更新酒店的API端点
@app.put("/hotels/{hotel_id}", status_code=200)
async def update_hotel(hotel_id: int, hotel: HotelUpdateModel):
    """更新酒店信息"""
    from app.sql_agent import update_hotel, get_hotel_by_id
    
    try:
        # 检查酒店是否存在
        existing_hotel = get_hotel_by_id(hotel_id)
        if not existing_hotel:
            raise HTTPException(status_code=404, detail=f"Hotel with ID {hotel_id} not found")
        
        # 记录更新尝试
        logger.info(f"Attempting to update hotel ID: {hotel_id}")
        
        # 将None值替换为现有值
        hotel_name = hotel.hotel_name if hotel.hotel_name is not None else existing_hotel.get("hotel_name")
        county = hotel.county if hotel.county is not None else existing_hotel.get("county")
        state = hotel.state.upper() if hotel.state is not None else existing_hotel.get("state")
        rating = hotel.rating if hotel.rating is not None else existing_hotel.get("rating")
        service = hotel.service if hotel.service is not None else existing_hotel.get("service")
        rooms = hotel.rooms if hotel.rooms is not None else existing_hotel.get("rooms")
        cleanliness = hotel.cleanliness if hotel.cleanliness is not None else existing_hotel.get("cleanliness")
        
        # 更新酒店信息
        result = update_hotel(
            hotel_id=hotel_id,
            hotel_name=hotel_name,
            county=county,
            state=state,
            rating=rating,
            service=service,
            rooms=rooms,
            cleanliness=cleanliness
        )
        
        if result:
            logger.info(f"Successfully updated hotel with ID: {hotel_id}")
            return {
                "success": True,
                "message": f"Hotel with ID {hotel_id} successfully updated"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update hotel")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating hotel: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 删除酒店的API端点
@app.delete("/hotels/{hotel_id}", status_code=200)
async def delete_hotel(hotel_id: int):
    """删除酒店"""
    from app.sql_agent import delete_hotel, get_hotel_by_id
    
    try:
        # 检查酒店是否存在
        existing_hotel = get_hotel_by_id(hotel_id)
        if not existing_hotel:
            raise HTTPException(status_code=404, detail=f"Hotel with ID {hotel_id} not found")
        
        # 记录删除尝试
        logger.info(f"Attempting to delete hotel ID: {hotel_id}")
        
        # 删除酒店
        result = delete_hotel(hotel_id)
        
        if result:
            logger.info(f"Successfully deleted hotel with ID: {hotel_id}")
            return {
                "success": True,
                "message": f"Hotel with ID {hotel_id} successfully deleted"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete hotel")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting hotel: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 添加一个获取酒店详情的API端点，便于修改前查看
@app.get("/hotels/{hotel_id}", status_code=200)
async def get_hotel(hotel_id: int):
    from app.sql_agent import get_hotel_by_id

    try:
        hotel = get_hotel_by_id(hotel_id)
        if not hotel:
            raise HTTPException(status_code=404, detail=f"Hotel with ID {hotel_id} not found")

        return hotel
    except Exception as e:
        logger.error(f"Error getting hotel: {e}")
        raise HTTPException(status_code=404, detail=f"Hotel with ID {hotel_id} not found")