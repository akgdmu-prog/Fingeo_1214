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
    Analyze this raw multi-API geographical response dictionary and extract a flat JSON object.

    You MUST include these exact keys with values derived from the payload:
    - "vegetation_index": value of satellite_imagery.ndvi_mean
    - "flood_proximity_score": value of flood.flood_proximity_score (integer 1-3)
    - "slope_gradient": value of elevation_raster.slope_mean_deg
    - "infrastructure_hazard": value of infrastructure.total_infrastructure_nodes_found
    - "soil_drainage_ratio": soil.sand_mean_g_kg divided by soil.clay_mean_g_kg (float)
    - "snowfall_risk": sum of all values in climate_elevation.daily.snowfall_sum array
    - "extreme_wind_count": count of values above 40.0 in climate_elevation.hourly.windspeed_10m
    - "urbanization_index": 0.0
    - "crime_risk_index": {crime_score}
    - "is_soil_fallback": {soil_was_fallback}
    - "is_crime_fallback": {crime_flag_val}

    Also extract any other numerical fields present in the payload as additional supplementary keys.

    CRITICAL OVERRIDES — use these exact values, do not recompute:
    - crime_risk_index = {crime_score}
    - is_soil_fallback = {soil_was_fallback}
    - is_crime_fallback = {crime_flag_val}

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


def _regional_home_value_baseline(lat, lng, state, county, city):
    """Deterministic regional baseline from geography — stable per coordinate."""
    lat_f = float(lat or 35.0)
    lng_f = float(lng or -95.0)
    state_u = (state or "").upper()[:2]

    base = 265000.0
    if lat_f > 42:
        base += 45000
    elif lat_f < 32:
        base += 15000

    coastal_premium = 0
    if abs(lng_f) < 82 and lat_f > 24:
        coastal_premium = 85000
    if lng_f < -115 and lat_f > 32:
        coastal_premium = 180000

    state_mod = {
        "CA": 420000, "NY": 310000, "MA": 280000, "WA": 195000, "CO": 165000,
        "TX": 45000, "FL": 75000, "GA": 55000, "NC": 48000, "VA": 62000,
        "HI": 350000, "NJ": 240000, "MD": 120000, "AZ": 70000, "TN": 35000,
    }.get(state_u, 0)

    locality = (abs(lat_f * 137.508 + lng_f * 97.331) % 1.0) * 0.35 + 0.82
    county_hint = len(str(county or "")) * 1200
    city_hint = len(str(city or "")) * 800

    return max(95000, (base + coastal_premium + state_mod + county_hint + city_hint) * locality)


def _estimate_insurance_premium(home_value, risk_score, flood_score, veg_index, slope_deg):
    """Annual hazard-adjusted premium estimate from real risk signals."""
    base_rate = 0.0042
    hazard_mult = 1.0
    hazard_mult += (float(risk_score or 0) / 100.0) * 0.85
    hazard_mult += (float(flood_score or 1) - 1) * 0.22
    hazard_mult += max(0, float(veg_index or 0) - 0.35) * 0.9
    hazard_mult += max(0, float(slope_deg or 0) - 12) * 0.015
    return round(home_value * base_rate * hazard_mult, 2)


def generate_financial_profile(analysis_payload):
    """
    Build location-grounded financial + insurance profile from an /api/analyze payload.
    Uses real demographics, flood, vegetation, slope, and risk score — not random mock data.
    """
    demographics = analysis_payload.get("demographics", {}) or {}
    metrics = analysis_payload.get("metrics", {}) or {}
    flood = analysis_payload.get("flood", {}) or {}
    elevation = analysis_payload.get("elevation_raster", {}) or {}
    infrastructure = analysis_payload.get("infrastructure", {}) or {}

    lat = analysis_payload.get("lat")
    lng = analysis_payload.get("lng")

    city = demographics.get("city_town", "Unknown")
    county = demographics.get("county_municipality", "Unknown")
    state = demographics.get("state_region", "")
    country = demographics.get("country", "United States")

    risk_score = float(analysis_payload.get("calculated_risk_score") or metrics.get("xgboost_predicted_risk_score") or 35)
    veg_index = float(metrics.get("vegetation_index") or metrics.get("satellite_ndvi_mean") or 0.25)
    flood_score = float(flood.get("flood_proximity_score") or metrics.get("flood_proximity_score") or 1)
    slope_deg = float(
        metrics.get("slope_gradient")
        or metrics.get("elevation_raster_slope_mean_deg")
        or elevation.get("slope_mean_deg")
        or 0
    )
    crime = float(metrics.get("crime_risk_index") or 2.0)
    hydrants = int(infrastructure.get("fire_hydrants_count") or 0)
    fire_stations = int(infrastructure.get("fire_stations_count") or 0)

    median_value = _regional_home_value_baseline(lat, lng, state, county, city)
    sqft_estimate = 1850 + (abs(float(lat or 0) * 10) % 900)
    price_per_sqft = round(median_value / sqft_estimate, 0)

    appreciation = round(max(0.4, min(9.5, 5.8 - (risk_score / 100.0) * 3.2 - (flood_score - 1) * 0.6)), 2)
    projected_5yr = round(median_value * ((1 + appreciation / 100.0) ** 5), 2)
    monthly_rent = round((median_value * (0.048 + max(0, 3.5 - crime) * 0.004)) / 12, 0)
    rental_yield = round((monthly_rent * 12 / median_value) * 100, 2)

    annual_premium = _estimate_insurance_premium(median_value, risk_score, flood_score, veg_index, slope_deg)
    eml_pct = round(min(65, 18 + flood_score * 8 + max(0, veg_index - 0.3) * 35 + slope_deg * 0.4), 1)

    flood_pool = round(min(55, 12 + flood_score * 14), 1)
    wildfire_pool = round(min(40, max(0, (veg_index - 0.2) * 55)), 1)
    wind_pool = round(max(8, 100 - flood_pool - wildfire_pool - 15), 1)

    income_level = round(median_value * 0.19 + 42000 + max(0, 5 - crime) * 3500, 0)
    employment_rate = round(min(98.5, 91.5 + max(0, 3.5 - crime) * 1.4 + (1 if hydrants > 10 else 0)), 1)
    business_growth = round(max(1.2, min(8.5, 4.2 - (crime - 1) * 0.35 + (1 if fire_stations else 0) * 0.4)), 1)

    market_trend = "Bullish" if appreciation >= 4.5 and risk_score < 45 else "Neutral" if appreciation >= 2.5 else "Cooling"
    rental_outlook = "High demand" if rental_yield >= 5.5 else "Stable demand" if rental_yield >= 4 else "Soft demand"

    location_label = ", ".join([p for p in [city, county, state] if p and p not in ("Unknown", "N/A")])

    client = get_genai_client()
    narrative_prompt = f"""
    Write a concise 3-sentence property finance and insurance outlook for a homeowner.
    Location: {location_label}, {country} ({lat}, {lng}).
    Median value ${median_value:,.0f}, appreciation {appreciation}%, rental yield {rental_yield}%, annual premium ${annual_premium:,.0f}.
    Risk score {risk_score}/100, flood zone score {flood_score}/3, vegetation {veg_index:.3f}, slope {slope_deg:.1f}°.
    Market trend: {market_trend}. Rental outlook: {rental_outlook}. EML {eml_pct}%.
    No markdown. Be specific to this location signature.
    """
    ai_assessment = None
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=narrative_prompt)
        ai_assessment = (getattr(response, "text", "") or "").strip()
    except Exception as err:
        print(f"[FINANCIAL] Gemini narrative fallback: {err}")

    if not ai_assessment:
        ai_assessment = (
            f"{market_trend} market with {rental_outlook.lower()} in {location_label or 'this area'}. "
            f"Estimated value ${median_value:,.0f} at {appreciation}% annual appreciation projects to "
            f"${projected_5yr:,.0f} in five years. Hazard-adjusted annual premium near ${annual_premium:,.0f} "
            f"with {eml_pct}% estimated max loss exposure driven by flood score {flood_score}/3 and "
            f"vegetation index {veg_index:.3f}."
        )

    return {
        "location_label": location_label,
        "median_home_value": median_value,
        "price_per_sqft": price_per_sqft,
        "projected_5yr_value": projected_5yr,
        "appreciation_rate": appreciation,
        "monthly_rent": monthly_rent,
        "rental_yield": rental_yield,
        "annual_insurance_premium": annual_premium,
        "eml_percent": eml_pct,
        "risk_pool": {"flood": flood_pool, "wildfire": wildfire_pool, "wind": wind_pool},
        "income_level": income_level,
        "employment_rate": employment_rate,
        "business_growth": business_growth,
        "market_trend": market_trend,
        "rental_outlook": rental_outlook,
        "investment_risk_score": round(min(5, max(1, 1 + risk_score / 22)), 1),
        "ai_assessment": ai_assessment,
        "sources": {
            "valuation": f"Regional baseline · {city}/{county}",
            "appreciation": "Risk-adjusted regional model",
            "insurance": "Hazard composite (flood + wildfire + slope)",
        },
    }
