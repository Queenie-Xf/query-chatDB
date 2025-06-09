# ✈️ Travel Information System

A comprehensive travel information system that integrates flight and hotel data into a user-friendly web interface. It leverages FastAPI for backend API services, Streamlit for the frontend interface, and a hybrid database system using MongoDB and SQLite. A local large language model (LLM) via Ollama enables natural language queries.

---

## 🧱 System Architecture

This project consists of the following integrated components:

- **Backend API**  
  Built with **FastAPI**, it handles requests for both flight and hotel data.

- **Frontend Interface**  
  Created with **Streamlit**, it provides users with a clean and interactive experience for querying and exploring data.

- **Database Layer**
  - **MongoDB**
    - `flights`: Stores basic flight details (departure, destination, fare, duration)
    - `flights_segments`: Contains detailed segment-level data including airline names
  - **SQLite**
    - `hotel_location.db`: Stores hotel names and location info (county, state)
    - `hotel_rate.db`: Stores review metrics (ratings, cleanliness, service, etc.)
  - **Ollama (LLM Engine)**
    - Powers natural language processing for intuitive querying

---

## 🗃️ Database Structure

### MongoDB Collections

- `flights`:  
  - Departure and destination airports  
  - Travel duration and fare  

- `flights_segments`:  
  - Segment-level flight details  
  - Airline names  

### SQLite Databases

- `hotel_location.db`:  
  - Hotel name  
  - County  
  - State  

- `hotel_rate.db`:  
  - Overall rating  
  - Cleanliness  
  - Sleep quality  
  - Service  
  - Value  
  - Rooms  

---

## 🔍 Features

### ✈️ Flight Data

- Search flights by departure and destination
- Search flights by airline name
- View all flights

### 🏨 Hotel Data

- Search hotels by county or state
- Filter hotels based on  rating
- View detailed hotel reviews

### 💻 Frontend

- Schema exploration for MongoDB and SQLite
- Structured form-based query tools
- Natural language query support via LLM
- Data modification interface (add/update/delete records)

---

## 🚀 How to Use

### Flight Search

- Search by:
  - Departure & destination airports
  - Airline name
  - Price

### Hotel Search

- Search by:
  - County or state
  - Rating

### Data Modification

- Add new flight or hotel records
- Edit existing entries
- Delete outdated records



