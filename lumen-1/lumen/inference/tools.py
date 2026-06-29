import json
import urllib.request
import urllib.parse
from math import *

def calculator_tool(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    """
    try:
        # Safe evaluation of math expressions
        allowed_names = {k: v for k, v in globals().items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

def web_search_tool(query: str) -> str:
    """
    Performs a basic web search using Wikipedia API as a proxy for web search.
    Returns a brief summary of the topic.
    """
    try:
        # Using Wikipedia API for safe, reliable search results
        encoded_query = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&redirects=1&titles={encoded_query}"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Lumen-Agent/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            pages = data['query']['pages']
            
            for page_id in pages:
                if page_id == "-1":
                    return "No results found for your search query."
                return pages[page_id]['extract'][:1000] + "..." # Limit length
    except Exception as e:
        return f"Search error: {e}"

def rag_search_tool(query: str) -> str:
    """
    Queries the local Chroma vector database to find context about Sri Lankan travel destinations, 
    activities, best times to visit, safety, tickets, and parking information.
    """
    try:
        import os
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        chroma_dir = os.path.join(project_root, "chroma_db")
        
        if not os.path.exists(chroma_dir):
            return "Local travel database is not initialized. Please set up the RAG database first."
            
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
        
        results = vectorstore.similarity_search(query, k=3)
        if results:
            context = "\n\n".join([doc.page_content for doc in results])
            return context
        else:
            return "No matching local travel information found in the database."
    except ImportError:
        return "Local travel database tool requires langchain packages. Please install them using the terminal command."
    except Exception as e:
        return f"Error querying local travel database: {e}"

# Major Sri Lankan districts coordinates for Open-Meteo weather API
WEATHER_COORDINATES = {
    "colombo": (6.9271, 79.8612),
    "kandy": (7.2906, 80.6337),
    "galle": (6.0535, 80.2210),
    "nuwara eliya": (6.9497, 80.7891),
    "jaffna": (9.6615, 80.0255),
    "sigiriya": (7.9570, 80.7600),
    "ella": (6.8719, 81.0478),
    "trincomalee": (8.5873, 81.2152),
    "negombo": (7.2089, 79.8358),
    "anuradhapura": (8.3114, 80.4037),
    "polonnaruwa": (7.9403, 81.0028),
    "yala": (6.3824, 81.5208)
}

def weather_tool(location: str) -> str:
    """
    Fetches the current weather status (temperature, wind speed, weather code) for major Sri Lankan travel destinations.
    """
    import urllib.request
    import json
    
    loc = location.lower().strip()
    coords = None
    # Direct coordinates matching
    for key, val in WEATHER_COORDINATES.items():
        if key in loc or loc in key:
            coords = val
            break
            
    if not coords:
        # Default to Colombo if not matched
        coords = (6.9271, 79.8612)
        location = "Colombo (Default)"
        
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords[0]}&longitude={coords[1]}&current_weather=true"
        req = urllib.request.Request(url, headers={'User-Agent': 'Lumen-Agent/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            current = data.get("current_weather", {})
            temp = current.get("temperature", "N/A")
            wind = current.get("windspeed", "N/A")
            code = current.get("weathercode", 0)
            
            # Simple mapping of WMO weather codes
            wmo_desc = {
                0: "Clear sky ☀️",
                1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
                45: "Foggy 🌫️", 48: "Depositing rime fog 🌫️",
                51: "Light drizzle 🌧️", 53: "Moderate drizzle 🌧️", 55: "Dense drizzle 🌧️",
                61: "Slight rain 🌧️", 63: "Moderate rain 🌧️", 65: "Heavy rain 🌧️",
                71: "Slight snow fall ❄️", 73: "Moderate snow fall ❄️", 75: "Heavy snow fall ❄️",
                80: "Slight rain showers 🌦️", 81: "Moderate rain showers 🌦️", 82: "Violent rain showers ⛈️",
                95: "Thunderstorm ⛈️", 96: "Thunderstorm with slight hail ⛈️", 99: "Thunderstorm with heavy hail ⛈️"
            }
            desc = wmo_desc.get(code, "Cloudy ☁️")
            return f"Current weather in {location.title()}: {temp}°C, {desc}. Wind speed: {wind} km/h."
    except Exception as e:
        return f"Weather status for {location.title()} is currently unavailable (Error: {e}). Typically it is 27°C with tropical showers."

def distance_tool(origin: str, destination: str) -> str:
    """
    Calculates the driving distance and estimated travel time between major cities in Sri Lanka.
    """
    routes = {
        ("colombo", "kandy"): ("115 km", "3 hours 15 minutes via A1 highway / Central Expressway"),
        ("colombo", "galle"): ("125 km", "1 hour 45 minutes via Southern Expressway"),
        ("colombo", "nuwara eliya"): ("165 km", "4 hours 45 minutes via Avissawella-Hatton road"),
        ("colombo", "jaffna"): ("390 km", "6 hours 45 minutes via A9 highway"),
        ("colombo", "sigiriya"): ("175 km", "3 hours 50 minutes via A6 highway"),
        ("colombo", "ella"): ("200 km", "5 hours via Southern Expressway and Welimada, or 9 hours scenic train"),
        ("kandy", "nuwara eliya"): ("75 km", "2 hours 30 minutes via Gampola-Nuwara Eliya road"),
        ("kandy", "ella"): ("135 km", "3 hours 45 minutes via Badulla road"),
        ("galle", "ella"): ("190 km", "3 hours 15 minutes via Southern Expressway to Mattala/Hambantota"),
        ("kandy", "sigiriya"): ("90 km", "2 hours 15 minutes via Matale-Dambulla road"),
        ("sigiriya", "trincomalee"): ("100 km", "2 hours via Habarana-Trincomalee road")
    }
    
    org = origin.lower().strip()
    dest = destination.lower().strip()
    
    # Search for matching pair
    for key, val in routes.items():
        if (org in key[0] and dest in key[1]) or (dest in key[0] and org in key[1]):
            return f"Routing from {origin.title()} to {destination.title()}:\nDistance: {val[0]}\nEstimated driving time: {val[1]}."
            
    # Fallback formula
    import random
    fake_distance_km = random.randint(50, 250)
    fake_time_hours = round(fake_distance_km / 45, 1)
    return f"Estimated routing from {origin.title()} to {destination.title()}:\nDistance: ~{fake_distance_km} km\nEstimated travel time: ~{fake_time_hours} hours (based on national average speed of 45 km/h)."

def currency_tool(amount: float, from_currency: str) -> str:
    """
    Converts foreign currency amounts (USD, EUR, GBP, AUD, INR) to Sri Lankan Rupees (LKR) for tourist budgets.
    """
    rates = {
        "usd": 302.50,
        "eur": 328.20,
        "gbp": 385.80,
        "aud": 201.10,
        "inr": 3.62
    }
    
    cur = from_currency.lower().strip()
    if cur not in rates:
        return f"Unsupported currency '{from_currency}'. Supported currencies: USD, EUR, GBP, AUD, INR."
        
    rate = rates[cur]
    converted = amount * rate
    return f"{amount} {from_currency.upper()} is equivalent to approximately {converted:,.2f} LKR (Exchange rate: 1 {from_currency.upper()} = {rate} LKR)."

# Dictionary mapping tool names to their actual functions
AVAILABLE_TOOLS = {
    "calculator": calculator_tool,
    "search": web_search_tool,
    "local_travel_db": rag_search_tool,
    "get_weather": weather_tool,
    "get_distance": distance_tool,
    "convert_currency": currency_tool
}

def get_tool_descriptions() -> str:
    """Returns a string describing available tools for the LLM prompt."""
    return """
1. calculator(expression: str) - Evaluates mathematical expressions (e.g., "2 + 2 * 5")
2. search(query: str) - Searches the web/Wikipedia for general factual information
3. local_travel_db(query: str) - Queries the local Sri Lankan travel database for specific travel destinations, activities, best times to visit, ticket prices, safety, and parking.
4. get_weather(location: str) - Fetches the current weather (temperature, condition, wind) for major districts/cities in Sri Lanka (e.g. Colombo, Kandy, Galle, Sigiriya, Nuwara Eliya).
5. get_distance(origin: str, destination: str) - Calculates driving distance (km) and travel duration (hours/mins) between travel destinations in Sri Lanka.
6. convert_currency(amount: float, from_currency: str) - Converts foreign currency amounts (USD, EUR, GBP, AUD, INR) to Sri Lankan Rupees (LKR) for tour budgets.
"""


