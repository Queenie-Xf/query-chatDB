import sqlite3
import os
import requests
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import json

# Database path constants
SQLITE_DB_DIR = "./data"
LOCATION_DB_PATH = os.path.join(SQLITE_DB_DIR, "hotel_location.db")
RATE_DB_PATH = os.path.join(SQLITE_DB_DIR, "hotel_rate.db")
API_URL = os.environ.get("API_URL", "http://backend:8000")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongodb:27017")

def get_sql_schema():
    """
    Get the schema information for the SQLite databases in a formatted string
    Returns a string with the schema in the format:
    table_name(column1, column2, column3, ...)
    """
    schema_text = "**Hotel Database (SQLite)**\n\n"
    
    try:
        # Connect to SQLite databases
        loc_conn = sqlite3.connect(LOCATION_DB_PATH)
        loc_cursor = loc_conn.cursor()
        
        rate_conn = sqlite3.connect(RATE_DB_PATH)
        rate_cursor = rate_conn.cursor()
        
        # Get tables from location database
        loc_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        loc_tables = loc_cursor.fetchall()
        
        schema_text += "**Location Database Tables:**\n\n"
        for table in loc_tables:
            table_name = table[0]
            # Get column information
            loc_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = loc_cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            # Format as table_name(col1, col2, ...)
            schema_text += f"- {table_name}({', '.join(column_names)})\n"
        
        # Get views from location database
        loc_cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        loc_views = loc_cursor.fetchall()
        
        if loc_views:
            schema_text += "\n**Location Database Views:**\n\n"
            for view in loc_views:
                view_name = view[0]
                # Get column information
                loc_cursor.execute(f"PRAGMA table_info({view_name})")
                columns = loc_cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                # Format as view_name(col1, col2, ...)
                schema_text += f"- {view_name}({', '.join(column_names)})\n"
        
        # Get tables from rate database
        rate_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        rate_tables = rate_cursor.fetchall()
        
        schema_text += "\n**Rate Database Tables:**\n\n"
        for table in rate_tables:
            table_name = table[0]
            # Get column information
            rate_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = rate_cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            # Format as table_name(col1, col2, ...)
            schema_text += f"- {table_name}({', '.join(column_names)})\n"
        
        # Get views from rate database
        rate_cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        rate_views = rate_cursor.fetchall()
        
        if rate_views:
            schema_text += "\n**Rate Database Views:**\n\n"
            for view in rate_views:
                view_name = view[0]
                # Get column information
                rate_cursor.execute(f"PRAGMA table_info({view_name})")
                columns = rate_cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                # Format as view_name(col1, col2, ...)
                schema_text += f"- {view_name}({', '.join(column_names)})\n"
        
    except Exception as e:
        schema_text += f"\nError retrieving SQL schema: {str(e)}"
    finally:
        if 'loc_conn' in locals():
            loc_conn.close()
        if 'rate_conn' in locals():
            rate_conn.close()
    
    return schema_text

def get_mongo_schema():
    """
    Get the schema information for MongoDB collections in a formatted string
    Returns a string with the schema in the format:
    collection_name – (field1, field2, field3, ...)
    """
    schema_text = "**Flight Database (MongoDB)**\n\n"
    
    try:
        # Try to use the API to get sample data first
        response = requests.get(f"{API_URL}/flights", params={"limit": 1})
        
        if response.status_code == 200:
            flights_data = response.json()
            
            if flights_data and len(flights_data) > 0:
                # Extract field names from the sample document
                fields = list(flights_data[0].keys())
                schema_text += f"- flights – (_id, {', '.join(fields)})\n"
            else:
                schema_text += "- flights – (No sample data available)\n"
            
            # Try to get segments collection data if possible
            try:
                segments_response = requests.get(f"{API_URL}/flights/segments", params={"limit": 1})
                if segments_response.status_code == 200:
                    segments_data = segments_response.json()
                    if segments_data and len(segments_data) > 0:
                        fields = list(segments_data[0].keys())
                        schema_text += f"- flights_segments – (_id, {', '.join(fields)})\n"
                    else:
                        schema_text += "- flights_segments – (No sample data available)\n"
            except:
                # Fallback to hardcoded schema if API doesn't support segments endpoint
                schema_text += "- flights_segments – (_id, originalId, segmentsAirlineName, segmentsDepartureTimeRaw, segmentsArrivalTimeRaw, ...)\n"
        else:
            # Fallback to hardcoded schema based on the provided code files
            schema_text += "- flights – (_id, originalId, flightDate, startingAirport, destinationAirport, fareBasisCode, travelDuration, elapsedDays, isBasicEconomy, isRefundable, isNonStop, baseFare, totalFare, totalTravelDistance, segmentDetails)\n"
            schema_text += "- flights_segments – (_id, originalId, segmentsDepartureTimeEpochSeconds, segmentsDepartureTimeRaw, segmentsArrivalTimeEpochSeconds, segmentsArrivalTimeRaw, segmentsArrivalAirportCode, segmentsDepartureAirportCode, segmentsAirlineName, segmentsAirlineCode, segmentsEquipmentDescription, segmentsDurationInSeconds, segmentsDistance, segmentsCabinCode)\n"
    
    except Exception as e:
        schema_text += f"\nError retrieving MongoDB schema: {str(e)}"
    
    return schema_text

def get_complete_schema():
    """
    Get the complete schema for both databases
    Returns a dictionary with both SQL and MongoDB schemas
    """
    return {
        "sql": get_sql_schema_dict(),
        "mongo": get_mongo_schema_dict()
    }

def get_mongo_schema_dict():
    """
    Get the MongoDB schema as a dictionary for programmatic use
    """
    schema = {}
    
    try:
        # Try to use the API to get sample data
        response = requests.get(f"{API_URL}/flights", params={"limit": 1})
        
        if response.status_code == 200:
            flights_data = response.json()
            
            if flights_data and len(flights_data) > 0:
                # Extract field names and types from the sample document
                fields = {}
                for key, value in flights_data[0].items():
                    fields[key] = type(value).__name__
                
                schema["flights"] = fields
            
            # Try to get segments collection data if possible
            try:
                segments_response = requests.get(f"{API_URL}/flights/segments", params={"limit": 1})
                if segments_response.status_code == 200:
                    segments_data = segments_response.json()
                    if segments_data and len(segments_data) > 0:
                        fields = {}
                        for key, value in segments_data[0].items():
                            fields[key] = type(value).__name__
                        
                        schema["flights_segments"] = fields
            except:
                # Fallback for segments collection
                schema["flights_segments"] = {
                    "originalId": "str",
                    "segmentsAirlineName": "str",
                    "segmentsDepartureTimeRaw": "str",
                    "segmentsArrivalTimeRaw": "str"
                }
        else:
            # Fallback for both collections
            schema["flights"] = {
                "originalId": "str",
                "startingAirport": "str",
                "destinationAirport": "str",
                "totalFare": "float",
                "totalTripDuration": "int"
            }
            schema["flights_segments"] = {
                "originalId": "str",
                "segmentsAirlineName": "str",
                "segmentsDepartureTimeRaw": "str",
                "segmentsArrivalTimeRaw": "str"
            }
    
    except Exception as e:
        schema["error"] = str(e)
    
    return schema

def get_sql_schema_dict():
    """
    Get the SQLite schema as a dictionary for programmatic use
    """
    schema = {
        "location_db": {},
        "rate_db": {}
    }
    
    try:
        # Connect to SQLite databases
        loc_conn = sqlite3.connect(LOCATION_DB_PATH)
        loc_cursor = loc_conn.cursor()
        
        rate_conn = sqlite3.connect(RATE_DB_PATH)
        rate_cursor = rate_conn.cursor()
        
        # Get tables from location database
        loc_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        loc_tables = loc_cursor.fetchall()
        
        for table in loc_tables:
            table_name = table[0]
            # Get column information
            loc_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = loc_cursor.fetchall()
            
            column_info = {}
            for col in columns:
                # col[1] is column name, col[2] is data type
                column_info[col[1]] = col[2]
            
            schema["location_db"][table_name] = column_info
        
        # Get views from location database
        loc_cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        loc_views = loc_cursor.fetchall()
        
        for view in loc_views:
            view_name = view[0]
            # Get column information
            loc_cursor.execute(f"PRAGMA table_info({view_name})")
            columns = loc_cursor.fetchall()
            
            column_info = {}
            for col in columns:
                column_info[col[1]] = col[2]
            
            schema["location_db"][view_name] = column_info
        
        # Get tables from rate database
        rate_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        rate_tables = rate_cursor.fetchall()
        
        for table in rate_tables:
            table_name = table[0]
            # Get column information
            rate_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = rate_cursor.fetchall()
            
            column_info = {}
            for col in columns:
                column_info[col[1]] = col[2]
            
            schema["rate_db"][table_name] = column_info
        
        # Get views from rate database
        rate_cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        rate_views = rate_cursor.fetchall()
        
        for view in rate_views:
            view_name = view[0]
            # Get column information
            rate_cursor.execute(f"PRAGMA table_info({view_name})")
            columns = rate_cursor.fetchall()
            
            column_info = {}
            for col in columns:
                column_info[col[1]] = col[2]
            
            schema["rate_db"][view_name] = column_info
    
    except Exception as e:
        schema["error"] = str(e)
    finally:
        if 'loc_conn' in locals():
            loc_conn.close()
        if 'rate_conn' in locals():
            rate_conn.close()
    
    return schema

def display_schema_in_streamlit():
    """
    Display the database schema in a Streamlit app with improved user experience
    """
    st.title("Database Schema")
    
    # Create tabs for different databases
    hotel_tab, flight_tab = st.tabs(["Hotel Database (SQL)", "Flight Database (MongoDB)"])
    
    with hotel_tab:
        # Create an expandable section to ask what tables exist
        with st.expander("What tables exist in the Hotel Database?", expanded=True):
            try:
                # Connect to SQLite databases
                loc_conn = sqlite3.connect(LOCATION_DB_PATH)
                loc_cursor = loc_conn.cursor()
                
                rate_conn = sqlite3.connect(RATE_DB_PATH)
                rate_cursor = rate_conn.cursor()
                
                # Get tables from location database
                loc_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                loc_tables = loc_cursor.fetchall()
                
                # Get tables from rate database
                rate_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                rate_tables = rate_cursor.fetchall()
                
                # Get views from both databases
                loc_cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
                loc_views = loc_cursor.fetchall()
                
                rate_cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
                rate_views = rate_cursor.fetchall()
                
                # Display tables and views
                st.subheader("Location Database (hotel_location.db)")
                st.write("Tables:")
                for table in loc_tables:
                    st.write(f"- {table[0]}")
                
                if loc_views:
                    st.write("Views:")
                    for view in loc_views:
                        st.write(f"- {view[0]}")
                
                st.subheader("Rate Database (hotel_rate.db)")
                st.write("Tables:")
                for table in rate_tables:
                    st.write(f"- {table[0]}")
                
                if rate_views:
                    st.write("Views:")
                    for view in rate_views:
                        st.write(f"- {view[0]}")
                
                loc_conn.close()
                rate_conn.close()
            
            except Exception as e:
                st.error(f"Error retrieving tables: {str(e)}")
        
        # Create an expandable section to view table attributes
        with st.expander("View Table Attributes", expanded=False):
            try:
                # Connect to SQLite databases
                loc_conn = sqlite3.connect(LOCATION_DB_PATH)
                rate_conn = sqlite3.connect(RATE_DB_PATH)
                
                # Get tables from location database
                loc_cursor = loc_conn.cursor()
                loc_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' OR type='view'")
                loc_tables_views = [item[0] for item in loc_cursor.fetchall()]
                
                # Get tables from rate database
                rate_cursor = rate_conn.cursor()
                rate_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' OR type='view'")
                rate_tables_views = [item[0] for item in rate_cursor.fetchall()]
                
                # Create selection boxes for tables
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Location Database")
                    selected_loc_table = st.selectbox("Select table or view:", loc_tables_views, key="loc_table_select")
                    
                    if selected_loc_table:
                        # Get column information
                        loc_cursor.execute(f"PRAGMA table_info({selected_loc_table})")
                        columns = loc_cursor.fetchall()
                        
                        # Create a DataFrame for better display
                        column_data = [(col[0], col[1], col[2], "Primary Key" if col[5] == 1 else "") 
                                       for col in columns]
                        df = pd.DataFrame(column_data, columns=["Index", "Column Name", "Data Type", "Key"])
                        st.dataframe(df)
                
                with col2:
                    st.subheader("Rate Database")
                    selected_rate_table = st.selectbox("Select table or view:", rate_tables_views, key="rate_table_select")
                    
                    if selected_rate_table:
                        # Get column information
                        rate_cursor.execute(f"PRAGMA table_info({selected_rate_table})")
                        columns = rate_cursor.fetchall()
                        
                        # Create a DataFrame for better display
                        column_data = [(col[0], col[1], col[2], "Primary Key" if col[5] == 1 else "") 
                                       for col in columns]
                        df = pd.DataFrame(column_data, columns=["Index", "Column Name", "Data Type", "Key"])
                        st.dataframe(df)
                
                loc_conn.close()
                rate_conn.close()
                
            except Exception as e:
                st.error(f"Error retrieving table attributes: {str(e)}")
        
        # Create an expandable section to retrieve sample rows
        with st.expander("Retrieve Sample Rows", expanded=True):
            try:
                # Connect to SQLite databases
                loc_conn = sqlite3.connect(LOCATION_DB_PATH)
                rate_conn = sqlite3.connect(RATE_DB_PATH)
                
                # Get tables from location database
                loc_cursor = loc_conn.cursor()
                loc_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' OR type='view'")
                loc_tables_views = [item[0] for item in loc_cursor.fetchall()]
                
                # Get tables from rate database
                rate_cursor = rate_conn.cursor()
                rate_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' OR type='view'")
                rate_tables_views = [item[0] for item in rate_cursor.fetchall()]
                
                # Create selection boxes for tables
                st.subheader("Select Database and Table")
                db_selection = st.radio("Database:", ["hotel_location.db", "hotel_rate.db"], key="db_radio")
                
                if db_selection == "hotel_location.db":
                    selected_table = st.selectbox("Select table or view:", loc_tables_views, key="loc_sample_select")
                    
                    if selected_table:
                        sample_size = st.slider("Number of sample rows", 5, 50, 10, key="loc_sample_slider")
                        
                        # Show the SQL query that will be executed
                        st.subheader("SQL Query:")
                        query = f"SELECT * FROM {selected_table} LIMIT {sample_size}"
                        st.code(query, language="sql")
                        
                        if st.button(f"Execute Query and Show Results", key="loc_sample_button"):
                            df = pd.read_sql_query(query, loc_conn)
                            st.dataframe(df)
                
                else:  # hotel_rate.db
                    selected_table = st.selectbox("Select table or view:", rate_tables_views, key="rate_sample_select")
                    
                    if selected_table:
                        sample_size = st.slider("Number of sample rows", 5, 50, 10, key="rate_sample_slider")
                        
                        # Show the SQL query that will be executed
                        st.subheader("SQL Query:")
                        query = f"SELECT * FROM {selected_table} LIMIT {sample_size}"
                        st.code(query, language="sql")
                        
                        if st.button(f"Execute Query and Show Results", key="rate_sample_button"):
                            df = pd.read_sql_query(query, rate_conn)
                            st.dataframe(df)
                
                loc_conn.close()
                rate_conn.close()
                
            except Exception as e:
                st.error(f"Error retrieving sample rows: {str(e)}")
    
    with flight_tab:
        # Display MongoDB collections schema in a clean format
        st.markdown("""
        ## Flight Database (MongoDB)
        
        The flight database contains information about flights and their segments:
        
        ### Collections:
        
        - **flights** – (_id, originalId, flightDate, startingAirport, destinationAirport, fareBasisCode, travelDuration, elapsedDays, isBasicEconomy, isRefundable, isNonStop, baseFare, totalFare, totalTravelDistance, segmentDetails)
        
        - **flights_segments** – (_id, originalId, segmentsDepartureTimeEpochSeconds, segmentsDepartureTimeRaw, segmentsArrivalTimeEpochSeconds, segmentsArrivalTimeRaw, segmentsArrivalAirportCode, segmentsDepartureAirportCode, segmentsAirlineName, segmentsAirlineCode, segmentsEquipmentDescription, segmentsDurationInSeconds, segmentsDistance, segmentsCabinCode)
        """)

        # Create an expandable section to view sample documents
        with st.expander("View Sample Documents", expanded=True):
            collection_name = st.radio("Select Collection:", ["flights", "flights_segments"], key="mongo_collection_radio")
            sample_size = st.slider("Number of sample documents", 1, 10, 3, key="mongo_sample_slider")
            
            # Show the MongoDB query that will be executed
            st.subheader("MongoDB Query:")
            query_code = f"""db.{collection_name}.find({{}}).limit({sample_size})"""
            st.code(query_code, language="javascript")
            
            if st.button(f"Execute Query and Show Results", key="mongo_sample_button"):
                try:
                    # Try to fetch sample documents via API
                    endpoint = f"/flights" if collection_name == "flights" else "/flights/segments"
                    response = requests.get(f"{API_URL}{endpoint}", params={"limit": sample_size})
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data:
                            # Convert to DataFrame for better display
                            df = pd.DataFrame(data)
                            st.dataframe(df)
                        else:
                            st.info("No sample data available")
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")

# If this file is run directly, display the schema
if __name__ == "__main__":
    display_schema_in_streamlit()