import requests
import json

def test_recommendation():
    url = "http://localhost:8000/recommend"
    headers = {"Content-Type": "application/json"}
    data = {
        "query": "protein supplement",
        "limit": 3
    }
    
    print("Making request to recommendation endpoint...")
    response = requests.post(url, headers=headers, json=data)
    
    print(f"\nStatus Code: {response.status_code}")
    print("\nResponse:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_recommendation() 