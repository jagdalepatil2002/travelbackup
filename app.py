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
CORS(app) # Allows frontend to call the backend

# Configure Database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Configure Gemini API
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

logging.basicConfig(level=logging.INFO)
# --- Helper Functions ---
def get_wikipedia_image_url(place_name):
    """Fetches the main image URL for a place from Wikipedia."""
    session = requests.Session()
    url = "https://en.wikipedia.org/w/api.php"
    
    headers = {
        'User-Agent': 'AITravelPlanner/1.0 (myemail@example.com)'
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
        print(f"Wikipedia API error for {place_name}: {e}")
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
        print(f"Cache hit for search term: '{location}'")
        return jsonify({"places": cached_places, "token_count": 0})

    # 2. If not cached, call LLM
    print(f"Cache miss for search term: '{location}'. Calling LLM.")
    prompt = prompts.get_initial_search_prompt(location)
    try:
        response = model.generate_content(prompt)
        token_count = response.usage_metadata.total_token_count
        print(f"Search places token count: {token_count}")

        clean_response = response.text.strip().replace("```json", "").replace("```", "")
        places_from_llm = json.loads(clean_response)

        for place in places_from_llm:
            place['image_url'] = get_wikipedia_image_url(place['name'])
            place['has_details'] = False 

        # 3. Save the new result to the database for future requests
        save_search_result(location, places_from_llm)

        return jsonify({"places": places_from_llm, "token_count": token_count})

    except Exception as e:
        print(f"Error during place search: {e}")
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
        save_place_details(place_name, detailed_description)

        return jsonify({"description": detailed_description, "token_count": token_count})
    except Exception as e:
        print(f"Error during detail generation: {e}")
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
        print(f"Text length ({len(text_to_speak)}) exceeds 3000 characters. Truncating.")
        text_to_speak = text_to_speak[:3000]

    try:
        logging.info(f"Calling murf API with text_to_speak = {text_to_speak}[:50]")
        if not os.getenv("MURF_API_KEY"):
            raise ValueError("MURF_API_KEY not found in environment variables.")
        
        client = Murf()
        voice_id = "en-US-terrell"
        logging.info(f"voice_id = {voice_id}")
        
        speech_response = client.text_to_speech.generate(
            text=text_to_speak,
            voice_id=voice_id
        )

        # Enhanced debugging to understand the response structure
        logging.info(f"Murf response type: {type(speech_response)}")
        logging.info(f"Response object: {speech_response}")

        audio_url = None
        # Try to convert to dict if the attribute has one
        if hasattr(speech_response, '__dict__'):
            response_dict = speech_response.__dict__
            print(f"Response dict keys: {response_dict.keys()}")
            for key, value in response_dict.items():
                print(f"  {key}: {type(value)} - {str(value)[:100]}...")
        
        # Directly access the audio_file attribute
        
        if hasattr(speech_response, 'audio_file'):
            audio_url = speech_response.audio_file  
        if audio_url:
            logging.info(f"Found audio URL: {audio_url}")
            return jsonify({"audio_url": audio_url})

        # Check for audio length
        audio_length = speech_response.__dict__.get('audio_length_in_seconds') if hasattr(speech_response, '__dict__') else None
        if audio_length and audio_length <= 0:
            raise ValueError(f"No audio generated, audio_length_in_seconds: {audio_length}")
        logging.info(f"no audio_url found audio_length: {audio_length}")
        # Last resort: serialize the entire response to see what we have
        try:
            response_data = speech_response.__dict__ if hasattr(speech_response, '__dict__') else str(speech_response)
            logging.info(f"Full response data: {response_data}")
        except Exception as debug_error:
            logging.error(f"Could not serialize response for debugging: {debug_error}")

        raise ValueError("No valid audio data or URL found in Murf API response despite thorough checks. Check logs for response structure, and confirm Murf API is functioning correctly.")
        logging.error(f"ValueError exception")

    except Exception as e:
        print(f"Error calling Murf API: {e}")
        return jsonify({"error": f"Failed to convert text to speech: {str(e)}"}), 502

# --- Run Application ---
if __name__ == '__main__':
    logging.info(f"Starting app.run")
    with app.app_context():
        init_db()
    app.run(debug=True, port=5000)
