from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.abspath(os.path.join(base_dir, "..", ".env"))
load_dotenv(dotenv_path=dotenv_path)


def get_genai_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()

def lookup_local_crime_rate(city, county, state, country):
    """
    Step 1: Uses dynamic Google Search Grounding using BOTH City and County definitions.
    """
    client = get_genai_client()
    
    # If no city or county is available, exit early to safety baseline
    if (not city or city in ["Unknown", "n/a", "N/A"]) and (not county or county in ["Unknown", "n/a", "N/A"]):
        return 2.5, True
        
    location_string = f"City: {city}, County/Region: {county}, State: {state}, Country: {country}"
    
    search_prompt = f"""
    Perform a targeted web search on the current crime indicators, safety reports, and 
    property/violent crime statistics for this location: {location_string}.
    Prioritize the city metrics; if the city is rural, unincorporated, or N/A, evaluate the entire County/Regional metric.
    Based on your web findings, output a single floating point number from 1.0 to 5.0 
    representing the overall crime danger (1.0 = extraordinarily safe, 5.0 = highly dangerous).
    Only reply with the float number (e.g., 1.5 or 4.6). Do not include any other text.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=search_prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.0
            )
        )
        score_text = response.text.strip()
        return float(score_text), False
    except Exception as search_err:
        print(f"[PIPELINE WARNING] Web search grounding failed, applying fallback index: {str(search_err)}")
        return 2.5, True

def process_raw_dump_to_database_row(raw_api_payload, location_name="loc_01"):
    client = get_genai_client()
    
    demographics = raw_api_payload.get("demographics", {}) or {}
    city = demographics.get("city_town", "Unknown")
    county = demographics.get("county_municipality", "Unknown")
    state = demographics.get("state_region", "")
    country = demographics.get("country", "")
    
    soil_block = raw_api_payload.get("soil", {}) or {}
    soil_was_fallback = 1 if "global_fallback_applied" in str(soil_block.get("status", "")) else 0
    
    print(f"[PIPELINE] Running targeted search for Location Focus -> City: {city} | County: {county}")
    crime_score, crime_was_fallback = lookup_local_crime_rate(city, county, state, country)
    crime_flag_val = 1 if crime_was_fallback else 0

    prompt = f"""
    Analyze this raw multi-API geographical response dictionary. 
    Condense, isolate, and convert all structural elements, text blocks, and arrays 
    into the requested flat numerical features defined by the schema.
    
    CRITICAL INPUT OVERRIDES:
    1. Set 'crime_risk_index' to exactly: {crime_score}
    2. Set 'is_soil_fallback' to exactly: {soil_was_fallback}
    3. Set 'is_crime_fallback' to exactly: {crime_flag_val}
    
    Target Raw Payload:
    {json.dumps(raw_api_payload)}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        
        data_row = json.loads(response.text)
        data_row["location_id"] = location_name
        
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_condensed_row.txt")
        with open(log_path, "w", encoding="utf-8") as out_file:
            out_file.write(f"=== GEMINI DATA HARMONIZATION PIPELINE DUMP ===\n")
            out_file.write(f"Processed Location Identifier: {location_name}\n")
            out_file.write(f"Resolved Boundaries -> City: {city} | County: {county}\n")
            out_file.write("="*50 + "\n\n")
            out_file.write(json.dumps(data_row, indent=4))
            
        print(f"[SUCCESS] Pipeline wrote features with Dual-Boundary Logic to: {log_path}")
        return data_row
        
    except Exception as e:
        print(f"Data aggregation failed: {str(e)}")
        return None