import os
import json
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import database
import prompts

# --- Initialization ---
load_dotenv()
app = Flask(__name__)
CORS(app) # Allows frontend to call the backend

# Configure Gemini API
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

# --- Helper Functions ---
def get_wikipedia_image_url(place_name):
    """Fetches the main image URL for a place from Wikipedia."""
    session = requests.Session()
    url = "https://en.wikipedia.org/w/api.php"
    
    # Add a User-Agent header to identify our script, as required by many APIs.
    headers = {
        'User-Agent': 'AITravelPlanner/1.0 (MyCoolApp; myemail@example.com)'
    }

    params = {
        "action": "query",
        "format": "json",
        "titles": place_name,
        "prop": "pageimages",
        "pithumbsize": 500, # Image width in pixels
        "pilicense": "any"
    }
    try:
        response = session.get(url=url, params=params, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        pages = data["query"]["pages"]
        for page_id in pages:
            if "thumbnail" in pages[page_id]:
                return pages[page_id]["thumbnail"]["source"]
    except Exception as e:
        print(f"Wikipedia API error for {place_name}: {e}")
    return None # Return None if no image is found or an error occurs

# --- API Endpoints ---
@app.route('/search-places', methods=['POST'])
def search_places():
    """
    Endpoint to get a list of 10 famous places.
    Checks the database for a cached search before calling the LLM.
    """
    if not model:
        return jsonify({"error": "AI Model not configured"}), 500

    data = request.get_json()
    if not data or 'location' not in data:
        return jsonify({"error": "Location not provided"}), 400

    location = data['location'].strip().lower()

    # 1. Check for a cached search result first
    cached_places = database.get_cached_search(location)
    if cached_places:
        print(f"Cache hit for search term: '{location}'")
        # Standardize the keys to match what the frontend expects
        standardized_places = [
            {
                "name": p["place_name"],
                "description": p["short_description"],
                "image_url": p["image_url"],
                "has_details": p["has_details"]
            }
            for p in cached_places
        ]
        return jsonify({"places": standardized_places, "token_count": 0})

    # 2. If not cached, call LLM
    print(f"Cache miss for search term: '{location}'. Calling LLM.")
    prompt = prompts.get_initial_search_prompt(location)
    try:
        response = model.generate_content(prompt)
        token_count = response.usage_metadata.total_token_count
        print(f"Search places token count: {token_count}")

        clean_response = response.text.strip().replace("```json", "").replace("```", "")
        places_from_llm = json.loads(clean_response)

        # Fetch images and add to the objects
        for place in places_from_llm:
            place['image_url'] = get_wikipedia_image_url(place['name'])

        # Add 'has_details' flag for consistent response format
        for place in places_from_llm:
            place['has_details'] = False
            # Rename 'description' to 'short_description' for consistency before saving
            place['short_description'] = place['description']

        # 3. Save the new result to the database for future requests
        database.save_search_result(location, places_from_llm)

        return jsonify({"places": places_from_llm, "token_count": token_count})

    except Exception as e:
        print(f"Error during place search: {e}")
        return jsonify({"error": "Failed to fetch places from AI model."}), 500

@app.route('/place-details', methods=['POST'])
def get_place_details():
    """
    Endpoint to get detailed, conversational info about a single place.
    Checks the database first before making the second, more expensive LLM call.
    """
    if not model:
        return jsonify({"error": "AI Model not configured"}), 500

    data = request.get_json()
    if not data or 'place_name' not in data:
        return jsonify({"error": "Place name not provided"}), 400

    place_name = data['place_name']

    # 1. Check database first
    cached_details = database.get_place_details(place_name)
    if cached_details:
        print(f"Cache hit for {place_name}")
        return jsonify({"description": cached_details, "token_count": 0})

    # 2. If not in DB, call LLM
    print(f"Cache miss for {place_name}. Calling LLM.")
    prompt = prompts.get_detailed_description_prompt(place_name)
    try:
        response = model.generate_content(prompt)
        token_count = response.usage_metadata.total_token_count
        print(f"Place details token count: {token_count}")
        detailed_description = response.text

        # 3. Save to database for future requests
        database.save_place_details(place_name, detailed_description)

        return jsonify({"description": detailed_description, "token_count": token_count})
    except Exception as e:
        print(f"Error during detail generation: {e}")
        return jsonify({"error": "Failed to generate details from AI model."}), 500



    # Initialize the database (creates table if it doesn't exist)
    database.init_db()
    # Run the Flask app
    app.run(debug=True, port=5000)
