import os
import json
import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import prompts 
from database import db, init_db, get_cached_search, save_search_result, get_place_details, save_place_details
import google.generativeai as genai
from dotenv import load_dotenv
from murf import Murf

import logging
# --- Initialization ---
load_dotenv()
app = Flask(__name__)
# In production, it's better to restrict this to your frontend's domain
# For example: CORS(app, resources={r"/*": {"origins": "https://your-frontend.onrender.com"}})
CORS(app) 

# Configure Database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Add a custom CLI command to initialize the database
@app.cli.command("db-init-command")
def init_db_command():
    """Initializes the database."""
    init_db()
    app.logger.info("Database initialized.")

# Configure Gemini API
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    app.logger.error(f"Error configuring Gemini API: {e}")
    model = None

logging.basicConfig(level=logging.INFO)
# --- Helper Functions ---
def get_wikipedia_image_url(place_name):
    """Fetches the main image URL for a place from Wikipedia."""
    session = requests.Session()
    url = "https://en.wikipedia.org/w/api.php"
    
    headers = {
        'User-Agent': 'AITravelPlanner/1.0 (github.com/your-username/your-repo)' # Be a good internet citizen!
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
        app.logger.error(f"Wikipedia API error for {place_name}: {e}")
    return None

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
    cached_places = get_cached_search(location)
    if cached_places:
        app.logger.info(f"Cache hit for search term: '{location}'")
        return jsonify({"places": cached_places, "token_count": 0})

    # 2. If not cached, call LLM
    app.logger.info(f"Cache miss for search term: '{location}'. Calling LLM.")
    prompt = prompts.get_initial_search_prompt(location)
    try:
        response = model.generate_content(prompt)
        token_count = response.usage_metadata.total_token_count
        app.logger.info(f"Search places token count: {token_count}")

        clean_response = response.text.strip().replace("```json", "").replace("```", "")
        places_from_llm = json.loads(clean_response)

        for place in places_from_llm:
            place['image_url'] = get_wikipedia_image_url(place['name'])
            place['has_details'] = False 

        # 3. Save the new result to the database for future requests
        save_search_result(location, places_from_llm)

        return jsonify({"places": places_from_llm, "token_count": token_count})

    except Exception as e:
        app.logger.error(f"Error during place search: {e}")
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
        app.logger.info(f"Cache hit for {place_name}")
        return jsonify({"description": cached_details, "token_count": 0})

    # 2. If not in DB, call LLM
    app.logger.info(f"Cache miss for {place_name}. Calling LLM.")
    prompt = prompts.get_detailed_description_prompt(place_name)
    try:
        response = model.generate_content(prompt)
        token_count = response.usage_metadata.total_token_count
        app.logger.info(f"Place details token count: {token_count}")
        detailed_description = response.text

        # 3. Save to database for future requests
        save_place_details(place_name, detailed_description)

        return jsonify({"description": detailed_description, "token_count": token_count})
    except Exception as e:
        app.logger.error(f"Error during detail generation: {e}")
        return jsonify({"error": "Failed to generate details from AI model."}), 500

# --- Text-to-Speech Endpoint ---
@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    logging.info(f"text_to_speech request, data = {data}")
    text_to_speak = data.get('text')

    if not text_to_speak:
        return jsonify({"error": "No text provided"}), 400

    # Truncate text to fit within Murf's 3000 character limit
    if len(text_to_speak) > 3000:
        app.logger.warning(f"Text length ({len(text_to_speak)}) exceeds 3000 characters. Truncating.")
        text_to_speak = text_to_speak[:3000]

    try:
        if not os.getenv("MURF_API_KEY"):
            raise ValueError("MURF_API_KEY not found in environment variables.")
        
        client = Murf()
        voice_id = "en-US-terrell"
        
        app.logger.info(f"Calling Murf API for text-to-speech (voice: {voice_id}).")
        speech_response = client.text_to_speech.generate(
            text=text_to_speak,
            voice_id=voice_id
        )

        # The murf-api library response object has an 'audio_file' attribute with the URL
        audio_url = getattr(speech_response, 'audio_file', None)

        if not audio_url:
            app.logger.error(f"Murf API call succeeded but no audio_file URL was found in the response. Response: {speech_response}")
            raise ValueError("No audio URL in Murf API response.")

        app.logger.info(f"Successfully generated audio URL: {audio_url}")
        return jsonify({"audio_url": audio_url})

    except Exception as e:
        app.logger.error(f"Error calling Murf API: {e}", exc_info=True)
        return jsonify({"error": f"Failed to convert text to speech: {str(e)}"}), 502

# --- Run Application ---
if __name__ == '__main__':
    # This block is for local development only.
    # In production, use a WSGI server like Gunicorn.
    # The database can be initialized via the command in render.yaml or a separate script.
    app.run(debug=True, port=5000)
