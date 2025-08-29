def get_initial_search_prompt(location):
    """
    Creates the prompt to get a list of 10 famous places for a given location.
    Asks the LLM to return a valid JSON string.
    """
    return f"""
    You are a travel expert. Your task is to identify the 10 most famous and must-see tourist attractions in or very near '{location}'.
    If the location itself is a specific landmark (like 'Eiffel Tower'), list it first, followed by 9 other famous places nearby.

    Provide the output as a single, minified JSON array of objects. Do not include any text before or after the JSON array.
    Each object in the array must have two keys:
    1. "name": The name of the attraction.
    2. "description": A brief, engaging 3-4 sentence summary for a tourist.

    Example format:
    [{{"name":"Louvre Museum","description":"The Louvre is the world's largest art museum and a historic monument in Paris, France. It is best known for being the home of the Mona Lisa. A central landmark of the city, it is located on the Right Bank of the Seine."}}]
    """

def get_detailed_description_prompt(place_name):
    """
    Creates the prompt for a detailed, conversational guide-style description of a place.
    This is intended to be read aloud.
    """
    return f"""
    You are a friendly, enthusiastic, and knowledgeable tour guide.
    A tourist is asking for a detailed guide about "{place_name}".
    Generate a captivating and comprehensive travel guide for them, written in a conversational and engaging tone, as if you are speaking to them directly. The total length should be between 800 and 1000 words.

    Your response must be a single block of text, perfect for a text-to-speech service.

    Structure your guide like this:
    1.  **Introduction:** Start with a warm welcome. Greet the traveler and introduce "{place_name}" in an exciting way.
    2.  **History and Significance:** Briefly share the story behind the place. Make it interesting, like telling a story, not like a dry history lesson.
    3.  **What to See and Do:** Describe the main highlights. What are the must-see things inside or around? What activities can they do? Use vivid language.
    4.  **Best Photo Spots:** Tell them where they can get the best photos. Be specific, like "For a stunning sunset shot, stand on the western corner...".
    5.  **Local Cuisine and Food:** Recommend nearby food or specific dishes they should try that are famous in the area.
    6.  **Best Time to Visit:** Advise on the best season, day of the week, or time of day to visit to avoid crowds or to get the best experience.
    7.  **Parting Tip:** End with a friendly closing remark or a final insider tip to make their visit memorable.

    Remember, speak directly to the user (e.g., "You'll want to...", "Imagine yourself..."). Your tone should be cheerful and passionate about the location.
    """
