import os
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

# 1. Create the SQLAlchemy instance. This is the central object.
db = SQLAlchemy()

# 2. Define your database models using the 'db' object.
#    SQLAlchemy will use these models to create the table schema.
class Place(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    image_url = db.Column(db.String(1024), nullable=True)

class Cache(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search_term = db.Column(db.String(255), unique=True, nullable=False)
    results = db.Column(db.JSON, nullable=False)

# 3. Define a simple init function that uses SQLAlchemy to create tables.
#    This function will create tables based on the models defined above.
def init_db():
    """Initializes and updates the database schema."""
    db.create_all()
    print("Database schema checked/updated.")

# 4. Define your database helper functions using the ORM.
def get_cached_search(location):
    """Retrieves cached search results for a location."""
    cache_entry = Cache.query.filter_by(search_term=location).first()
    return cache_entry.results if cache_entry else None

def save_search_result(location, places):
    """Saves a list of places to the search cache."""
    existing_cache = Cache.query.filter_by(search_term=location).first()
    if existing_cache:
        existing_cache.results = places
    else:
        new_cache = Cache(search_term=location, results=places)
        db.session.add(new_cache)
    db.session.commit()

def get_place_details(place_name):
    """Retrieves detailed description for a place."""
    place = Place.query.filter_by(name=place_name).first()
    return place.description if place and place.description else None

def save_place_details(place_name, description, image_url=None):
    """Saves or updates the detailed description and image for a place."""
    place = Place.query.filter_by(name=place_name).first()
    if place:
        place.description = description
        if image_url:
            place.image_url = image_url
    else:
        new_place = Place(name=place_name, description=description, image_url=image_url)
        db.session.add(new_place)
    db.session.commit()

# Initialize the database when the application starts
if __name__ == '__main__':
    load_dotenv()
    init_db()
