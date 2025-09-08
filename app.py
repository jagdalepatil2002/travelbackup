import os
import json
import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import prompts 
from database import db, init_db, get_cached_search, save_search_result, get_place_details, save_place_details
import google.generativeai as genai
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv()
app = Flask(__name__)

# CORS Configuration - simplified since no TTS endpoint needed
CORS(app, 
     origins=["https://travelgenie-9t7r.onrender.com"], 
     methods=["GET", "POST"], 
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

# Configure Database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Add a custom CLI command to initialize the database
@app.cli.command("db-init-command")
def init_db_command():
    """Initializes the database."""
    init_db()

# Configure Gemini API
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.0-flash-latest')
except Exception as e:
    model = None

# --- Helper Functions ---
def get_wikipedia_image_url(place_name):
    """Fetches the main image URL for a place from Wikipedia."""
    session = requests.Session()
    url = "https://en.wikipedia.org/w/api.php"
    
    headers = {
        'User-Agent': 'AITravelPlanner/1.0 (github.com/your-username/your-repo)'
    }

    params = {
        "action": "query",
        "format": "json",
        "titles": place_name,
        "prop": "pageimages",
        "pithumbsize": 500,
        "pilicense": "any"
    }
    try:
        response = session.get(url=url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        pages = data["query"]["pages"]
        for page_id in pages:
            if "thumbnail" in pages[page_id]:
                return pages[page_id]["thumbnail"]["source"]
    except Exception as e:
        pass
    return None

# --- API Endpoints ---
@app.route('/')
def health_check():
    """A simple health check endpoint to confirm the server is running."""
    return "Backend is running."

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
    cached_places = get_cached_search(location)
    if cached_places:
        return jsonify({"places": cached_places, "token_count": 0})

    # 2. If not cached, call LLM
    prompt = prompts.get_initial_search_prompt(location)
    try:
        response = model.generate_content(prompt)

        clean_response = response.text.strip().replace("```json", "").replace("```", "")
        places_from_llm = json.loads(clean_response)

        for place in places_from_llm:
            place['image_url'] = get_wikipedia_image_url(place['name'])
            place['has_details'] = False 

        # 3. Save the new result to the database for future requests
        save_search_result(location, places_from_llm)
        token_count = response.usage_metadata.total_token_count
        return jsonify({"places": places_from_llm, "token_count": token_count})

    except Exception as e:
        return jsonify({"error": "Failed to fetch places from AI model."}), 500

@app.route('/place-details', methods=['POST'])
def get_place_details_route():
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
    cached_details = get_place_details(place_name)
    if cached_details:
        return jsonify({"description": cached_details, "token_count": 0})

    # 2. If not in DB, call LLM
    prompt = prompts.get_detailed_description_prompt(place_name)
    try:
        response = model.generate_content(prompt)
        detailed_description = response.text
        token_count = response.usage_metadata.total_token_count

        # 3. Save to database for future requests
        save_place_details(place_name, detailed_description)

        return jsonify({"description": detailed_description, "token_count": token_count})
    except Exception as e:
        return jsonify({"error": "Failed to generate details from AI model."}), 500

# --- Run Application ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
