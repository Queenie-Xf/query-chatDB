import os
import logging
from sqlalchemy import create_engine
import sqlite3

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://flightsdata:dsci551@flightsdata.y57hp.mongodb.net/?retryWrites=true&w=majority",
)
# Use consistent database name
MONGO_DB = "flights"  # Changed from "flightsdata" to match the db name used elsewhere
MONGO_COLLECTIONS = {
    "flights": "flights_basic",
    "segments": "flights_segments",
}

# MongoDB local connection (as fallback)
MONGO_HOST = os.environ.get("MONGO_HOST", "mongodb")
MONGO_PORT = os.environ.get("MONGO_PORT", "27017")
LOCAL_MONGO_URI = f"mongodb://{MONGO_HOST}:{MONGO_PORT}"

# Log MongoDB connection information
logger.info(f"MongoDB configuration:")
logger.info(f"  Cloud URI: {MONGO_URI}")
logger.info(f"  Local URI: {LOCAL_MONGO_URI}")
logger.info(f"  Database: {MONGO_DB}")
logger.info(f"  Collections: {MONGO_COLLECTIONS}")

# SQLite Configuration
SQLITE_DB_DIR = os.environ.get("SQLITE_DB_DIR", os.path.join(os.getcwd(), "data"))
LOCATION_DB_PATH = os.path.join(SQLITE_DB_DIR, "hotel_location.db")
RATE_DB_PATH = os.path.join(SQLITE_DB_DIR, "hotel_rate.db")

# Log SQLite configuration
logger.info(f"SQLite configuration:")
logger.info(f"  Database directory: {SQLITE_DB_DIR}")
logger.info(f"  Location database: {LOCATION_DB_PATH}")
logger.info(f"  Rate database: {RATE_DB_PATH}")

# Check if SQLite database files exist
if not os.path.exists(SQLITE_DB_DIR):
    logger.warning(f"SQLite database directory {SQLITE_DB_DIR} does not exist. Attempting to create it.")
    try:
        os.makedirs(SQLITE_DB_DIR, exist_ok=True)
        logger.info(f"Created directory: {SQLITE_DB_DIR}")
    except Exception as e:
        logger.error(f"Failed to create SQLite database directory: {e}")

if not os.path.exists(LOCATION_DB_PATH):
    logger.warning(f"Hotel location database file {LOCATION_DB_PATH} does not exist.")
else:
    logger.info(f"Hotel location database file exists.")

if not os.path.exists(RATE_DB_PATH):
    logger.warning(f"Hotel rate database file {RATE_DB_PATH} does not exist.")
else:
    logger.info(f"Hotel rate database file exists.")

# Create SQLAlchemy engines with error handling
try:
    location_engine = create_engine(f"sqlite:///{LOCATION_DB_PATH}")
    logger.info("Created SQLAlchemy engine for location database.")
except Exception as e:
    logger.error(f"Failed to create SQLAlchemy engine for location database: {e}")
    location_engine = None

try:
    rate_engine = create_engine(f"sqlite:///{RATE_DB_PATH}")
    logger.info("Created SQLAlchemy engine for rate database.")
except Exception as e:
    logger.error(f"Failed to create SQLAlchemy engine for rate database: {e}")
    rate_engine = None

# Function to create empty SQLite databases if they don't exist
def create_empty_databases():
    """
    Create empty SQLite databases with minimal required schema if they don't exist
    """
    # Create location database
    if not os.path.exists(LOCATION_DB_PATH):
        logger.info(f"Creating new location database at {LOCATION_DB_PATH}")
        try:
            conn = sqlite3.connect(LOCATION_DB_PATH)
            cursor = conn.cursor()
            
            # Create minimal required tables
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hotel_name1 (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                hotel_name TEXT
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hotel_name2 (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                hotel_name TEXT
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hotel_name3 (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                hotel_name TEXT
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS location (
                ID INTEGER PRIMARY KEY,
                county TEXT,
                state TEXT
            )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Successfully created location database with minimal schema")
        except Exception as e:
            logger.error(f"Failed to create location database: {e}")
    
    # Create rate database
    if not os.path.exists(RATE_DB_PATH):
        logger.info(f"Creating new rate database at {RATE_DB_PATH}")
        try:
            conn = sqlite3.connect(RATE_DB_PATH)
            cursor = conn.cursor()
            
            # Create minimal required tables
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS rate (
                ID INTEGER PRIMARY KEY,
                rating REAL,
                service REAL,
                rooms REAL,
                cleanliness REAL,
                value REAL
            )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Successfully created rate database with minimal schema")
        except Exception as e:
            logger.error(f"Failed to create rate database: {e}")

# Create empty databases if they don't exist (optional)
# Uncomment the following line to enable automatic database creation
# create_empty_databases()