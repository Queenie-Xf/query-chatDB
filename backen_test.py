import requests
import socket
import time
import sys

def test_backend_connection():
    """Test if the backend API is accessible"""
    # Try different possible URLs
    urls_to_try = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://backend:8000"  # Only works in Docker environment
    ]
    
    print("Testing backend API connection...")
    
    for url in urls_to_try:
        try:
            # Extract hostname for resolution test
            hostname = url.split("//")[1].split(":")[0]
            
            # Check if hostname is resolvable
            try:
                ip_address = socket.gethostbyname(hostname)
                print(f"✅ Hostname '{hostname}' resolves to {ip_address}")
            except socket.gaierror:
                print(f"❌ Hostname '{hostname}' cannot be resolved")
                continue
            
            # Try to connect to API
            print(f"Trying to connect to {url}...")
            response = requests.get(f"{url}/", timeout=5)
            
            if response.status_code == 200:
                print(f"✅ Successfully connected to backend API at {url}")
                print(f"Response: {response.text}")
                return url
            else:
                print(f"❌ Got response code {response.status_code} from {url}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection error when connecting to {url}")
        except requests.exceptions.Timeout:
            print(f"❌ Timeout when connecting to {url}")
        except Exception as e:
            print(f"❌ Error connecting to {url}: {e}")
    
    print("\n❌ Failed to connect to backend API using any URL")
    print("\nPossible reasons:")
    print("1. Backend server is not running")
    print("2. Backend is running on a different port")
    print("3. Firewall is blocking the connection")
    
    print("\nSuggested actions:")
    print("1. Make sure the backend server is running (check with: docker-compose ps)")
    print("2. If running outside Docker, start the backend with: uvicorn app.main:app --host 0.0.0.0 --port 8000")
    
    return None

if __name__ == "__main__":
    backend_url = test_backend_connection()
    if not backend_url:
        sys.exit(1)
    
    # Test specific endpoints
    print("\nTesting specific endpoints...")
    
    endpoints = [
        "/flights?limit=2",
        "/hotels?limit=2"
    ]
    
    for endpoint in endpoints:
        try:
            url = f"{backend_url}{endpoint}"
            print(f"Testing: {url}")
            # Increase timeout to 15 seconds
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                count = len(data) if isinstance(data, list) else "N/A"
                print(f"✅ Success! Received {count} results")
                # Print first result for inspection
                if isinstance(data, list) and len(data) > 0:
                    print(f"Sample result: {data[0]}")
            else:
                print(f"❌ Error: Status code {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")