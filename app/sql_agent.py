import sqlite3
import os
import logging
from sqlalchemy import create_engine

# ——— Logging configuration ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ——— Database file paths ———
SQLITE_DB_DIR = os.environ.get("SQLITE_DB_DIR", os.path.join(os.getcwd(), "data"))
LOCATION_DB_PATH = os.path.join(SQLITE_DB_DIR, "hotel_location.db")
RATE_DB_PATH = os.path.join(SQLITE_DB_DIR, "hotel_rate.db")

# Optional SQLAlchemy engines
location_engine = create_engine(f"sqlite:///{LOCATION_DB_PATH}")
rate_engine = create_engine(f"sqlite:///{RATE_DB_PATH}")


def get_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def execute_sql_query(db_path, sql, params=(), fetch_all=True, enforce_limit=True):
    conn = get_connection(db_path)
    cur = conn.cursor()
    try:
        # Check if we need to enforce a limit and if one isn't already present
        if enforce_limit and fetch_all and "LIMIT" not in sql.upper():
            sql = f"{sql} LIMIT 20"  # Add default limit of 10

        cur.execute(sql, params)
        if fetch_all:
            rows = cur.fetchall()
        else:
            rows = cur.fetchone()
        conn.commit()
    except Exception as e:
        logger.error(f"SQL execution error: {e}, Query: {sql}")
        rows = [] if fetch_all else None
    finally:
        conn.close()
    return rows


def create_views():
    try:
        # 1) hotel_complete_view in hotel_location.db
        execute_sql_query(LOCATION_DB_PATH, "DROP VIEW IF EXISTS hotel_complete_view;")
        execute_sql_query(
            LOCATION_DB_PATH,
            """
            CREATE VIEW hotel_complete_view AS
              SELECT h1.ID,       h1.hotel_name, l.county, l.state
                FROM hotel_name1 h1 JOIN location l ON h1.ID = l.ID
              UNION ALL
              SELECT h2.ID,       h2.hotel_name, l.county, l.state
                FROM hotel_name2 h2 JOIN location l ON h2.ID = l.ID
              UNION ALL
              SELECT h3.ID,       h3.hotel_name, l.county, l.state
                FROM hotel_name3 h3 JOIN location l ON h3.ID = l.ID;
        """,
        )
        logger.info("Dropped & recreated hotel_complete_view")

        # 2) unified_hotel & rate_complete_view in hotel_rate.db
        conn = sqlite3.connect(RATE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("DROP VIEW IF EXISTS unified_hotel;")
        cur.execute("DROP VIEW IF EXISTS rate_complete_view;")

        cur.execute(
            """
            CREATE VIEW unified_hotel AS
              SELECT ID, hotel_name FROM hotel_name1
              UNION ALL
              SELECT ID, hotel_name FROM hotel_name2
              UNION ALL
              SELECT ID, hotel_name FROM hotel_name3;
        """
        )

        cur.execute(
            """
            CREATE VIEW rate_complete_view AS
            SELECT
              r.ID,
              u.hotel_name,
              r.rating,
              r.service,
              r.rooms,
              r.cleanliness
            FROM rate r
            LEFT JOIN unified_hotel u ON r.ID = u.ID;
        """
        )

        conn.commit()
        conn.close()
        logger.info("Dropped & recreated unified_hotel and rate_complete_view")
        return True

    except Exception as e:
        logger.error(f"Error creating views: {e}")
        return False


# create views at startup
if not create_views():
    logger.warning("View creation failed; falling back to manual queries")


# ——— Public query functions ———


def get_all_hotels():
    return execute_sql_query(
        LOCATION_DB_PATH,
        "SELECT DISTINCT ID, hotel_name, county, state FROM hotel_complete_view",
    )


def get_all_reviews():
    return execute_sql_query(RATE_DB_PATH, "SELECT * FROM rate_complete_view")


def get_reviews_by_county(county):
    hotels = execute_sql_query(
        LOCATION_DB_PATH,
        "SELECT ID FROM hotel_complete_view WHERE county = ?",
        (county,),
    )
    ids = [h["ID"] for h in hotels]
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    return execute_sql_query(
        RATE_DB_PATH, f"SELECT * FROM rate_complete_view WHERE ID IN ({ph})", ids
    )


def get_reviews_by_state(state):
    hotels = execute_sql_query(
        LOCATION_DB_PATH,
        "SELECT ID FROM hotel_complete_view WHERE UPPER(state)=UPPER(?)",
        (state,),
    )
    ids = [h["ID"] for h in hotels]
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    return execute_sql_query(
        RATE_DB_PATH, f"SELECT * FROM rate_complete_view WHERE ID IN ({ph})", ids
    )


def find_hotels_with_min_rating(min_rating):
    return execute_sql_query(
        RATE_DB_PATH,
        "SELECT * FROM rate_complete_view WHERE rating >= ?",
        (min_rating,),
    )


def execute_custom_query(query, is_location_db=True):
    db = LOCATION_DB_PATH if is_location_db else RATE_DB_PATH
    return execute_sql_query(db, query)


# ——— Hotel CRUD operations ———


def add_hotel(hotel_name, county, state, rating, service, rooms, cleanliness):
    """Add a new hotel and return its ID"""
    try:
        # First insert into hotel_name1
        location_query = "INSERT INTO hotel_name1 (hotel_name) VALUES (?);"
        execute_sql_query(
            LOCATION_DB_PATH, location_query, (hotel_name,), fetch_all=False
        )

        # Get the last inserted row ID
        conn = sqlite3.connect(LOCATION_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT last_insert_rowid()")
        hotel_id = cursor.fetchone()[0]
        conn.close()

        # Insert location data
        location_query = "INSERT INTO location (ID, county, state) VALUES (?, ?, ?);"
        execute_sql_query(
            LOCATION_DB_PATH, location_query, (hotel_id, county, state), fetch_all=False
        )

        # Insert rating data - FIX HERE
        rate_query = """
        INSERT INTO rate (ID, rating, service, rooms, cleanliness)
        VALUES (?, ?, ?, ?, ?);
        """
        execute_sql_query(
            RATE_DB_PATH,
            rate_query,
            (hotel_id, rating, service, rooms, cleanliness),
            fetch_all=False,
        )

        return hotel_id
    except Exception as e:
        logger.error(f"Error adding hotel: {e}")
        raise




def delete_hotel(hotel_id):
    """Delete a hotel by ID"""
    try:
        # Delete from hotel_name1
        name_query = "DELETE FROM hotel_name1 WHERE ID = ?;"
        execute_sql_query(LOCATION_DB_PATH, name_query, (hotel_id,), fetch_all=False)

        # Delete from location
        location_query = "DELETE FROM location WHERE ID = ?;"
        execute_sql_query(
            LOCATION_DB_PATH, location_query, (hotel_id,), fetch_all=False
        )

        # Delete from rate
        rate_query = "DELETE FROM rate WHERE ID = ?;"
        execute_sql_query(RATE_DB_PATH, rate_query, (hotel_id,), fetch_all=False)

        return True
    except Exception as e:
        logger.error(f"Error deleting hotel: {e}")
        return False


def get_hotel_by_id(hotel_id):
    """Get a hotel by ID"""
    try:
        try:
            hotel_id = int(hotel_id)
        except ValueError:
            logger.error(f"Invalid hotel ID: {hotel_id}")
            return None

        location_query = """
        SELECT ID, hotel_name, county, state
        FROM hotel_complete_view
        WHERE ID = ?;
        """
        hotel_data = execute_sql_query(LOCATION_DB_PATH, location_query, (hotel_id,), fetch_all=False)

        if not hotel_data:
            query1 = """
            SELECT h.ID, h.hotel_name, l.county, l.state
            FROM hotel_name1 h
            JOIN location l ON h.ID = l.ID
            WHERE h.ID = ?;
            """
            hotel_data = execute_sql_query(LOCATION_DB_PATH, query1, (hotel_id,), fetch_all=False)

            if not hotel_data:
                query2 = """
                SELECT h.ID, h.hotel_name, l.county, l.state
                FROM hotel_name2 h
                JOIN location l ON h.ID = l.ID
                WHERE h.ID = ?;
                """
                hotel_data = execute_sql_query(LOCATION_DB_PATH, query2, (hotel_id,), fetch_all=False)

                if not hotel_data:
                    query3 = """
                    SELECT h.ID, h.hotel_name, l.county, l.state
                    FROM hotel_name3 h
                    JOIN location l ON h.ID = l.ID
                    WHERE h.ID = ?;
                    """
                    hotel_data = execute_sql_query(LOCATION_DB_PATH, query3, (hotel_id,), fetch_all=False)

        if not hotel_data:
            logger.warning(f"No hotel found with ID: {hotel_id}")
            return None

        rate_query = "SELECT * FROM rate WHERE ID = ?;"
        rate_data = execute_sql_query(
            RATE_DB_PATH, rate_query, (hotel_id,), fetch_all=False
        )

        result = {
            "ID": hotel_data["ID"],
            "hotel_name": hotel_data["hotel_name"],
            "county": hotel_data["county"],
            "state": hotel_data["state"],
            "rating": rate_data["rating"] if rate_data else None,
            "service": rate_data["service"] if rate_data else None,
            "rooms": rate_data["rooms"] if rate_data else None,
            "cleanliness": rate_data["cleanliness"] if rate_data else None,
        }

        return result
    except Exception as e:
        logger.error(f"Error getting hotel by ID: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def update_hotel(hotel_id, hotel_name, county, state, rating, service, rooms, cleanliness):
    """Update an existing hotel"""
    try:
        # Update hotel name
        name_query = "UPDATE hotel_name1 SET hotel_name = ? WHERE ID = ?;"
        execute_sql_query(LOCATION_DB_PATH, name_query, (hotel_name, hotel_id), fetch_all=False)

        # Update location data
        location_query = "UPDATE location SET county = ?, state = ? WHERE ID = ?;"
        execute_sql_query(LOCATION_DB_PATH, location_query, (county, state, hotel_id), fetch_all=False)

        # Update rating data
        rate_query = """
        UPDATE rate SET rating = ?, service = ?, rooms = ?, cleanliness = ?
        WHERE ID = ?;
        """
        execute_sql_query(RATE_DB_PATH, rate_query,
                          (rating, service, rooms, cleanliness, hotel_id),
                          fetch_all=False)

        return True
    except Exception as e:
        logger.error(f"Error updating hotel: {e}")
        return False


def search_hotels_by_name(name):
    """Search for hotels by name"""
    try:
        # Search for hotels with similar names
        search_query = """
        SELECT h.ID, h.hotel_name, l.county, l.state
        FROM hotel_name1 h
        JOIN location l ON h.ID = l.ID
        WHERE h.hotel_name LIKE ?
        LIMIT 10;
        """
        search_param = f"%{name}%"
        hotels = execute_sql_query(LOCATION_DB_PATH, search_query, (search_param,))

        # Format the results
        result = []
        for hotel in hotels:
            # Get rating data
            rate_query = "SELECT * FROM rate WHERE ID = ?;"
            # Ensure ID is passed as int
            hotel_id = int(hotel["ID"]) if hotel["ID"] else None
            rate_data = (
                execute_sql_query(
                    RATE_DB_PATH, rate_query, (hotel_id,), fetch_all=False
                )
                if hotel_id
                else None
            )

            result.append(
                {
                    "ID": hotel_id,  # Ensure ID is stored as int
                    "hotel_name": hotel["hotel_name"],
                    "county": hotel["county"],
                    "state": hotel["state"],
                    "rating": rate_data["rating"] if rate_data else None,
                    "service": rate_data["service"] if rate_data else None,
                    "rooms": rate_data["rooms"] if rate_data else None,
                    "cleanliness": rate_data["cleanliness"] if rate_data else None,
                }
            )

        return result
    except Exception as e:
        logger.error(f"Error searching hotels by name: {e}")
        return []


def answer_sql_question(q: str):
    """Very simple keyword-based natural language -> SQL logic"""
    q_lower = q.lower()

    if "orange county" in q_lower:
        return get_reviews_by_county("Orange")
    elif "california" in q_lower or "state ca" in q_lower:
        return get_reviews_by_state("CA")
    elif "rating above" in q_lower or "ratings above" in q_lower:
        return find_hotels_with_min_rating(4.5)
    elif "all hotels" in q_lower:
        return get_all_hotels()
    elif "all reviews" in q_lower:
        return get_all_reviews()
    elif "hilton" in q_lower or "marriott" in q_lower:
        return search_hotels_by_name("Hilton" if "hilton" in q_lower else "Marriott")
    else:
        # fallback query
        return get_all_reviews()
