from flask import Flask, jsonify, request, render_template, session
import math
import os
import requests
from geopy.distance import geodesic
import planetary_computer
from pystac_client import Client
from geopy.geocoders import Nominatim
import json
from google import genai
from google.genai import types
import numpy as np
from dotenv import load_dotenv
import xgboost as xgb

try:
    from .pipeline import process_raw_dump_to_database_row
except ImportError:
    from pipeline import process_raw_dump_to_database_row

base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.abspath(os.path.join(base_dir, "..", ".env"))
load_dotenv(dotenv_path=dotenv_path)


def get_genai_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()

def generate_ai_risk_overview(condensed_features, final_score):
    """
    Leverages Gemini to synthesize the final model score and tabular metrics
    into a natural language descriptive summary for the end-user.
    """
    client = get_genai_client()

    prompt = f"""
    You are an expert geospatial risk underwriting intelligence assistant for the FinGeoRisk platform.
    Analyze the following location metrics and provide a concise, professional 3-sentence summary 
    contextualizing the overall safety environment for a property asset. Do not use Markdown styling.
    
    Data Metrics Profile:
    - Final Calculated XGBoost Safety Risk Index: {final_score}/100
    - Vegetation Canopy Densities (NDVI): {condensed_features.get('vegetation_index', 0.0):.4f}
    - Flood Vector Drainage Proximity Score (1-3): {condensed_features.get('flood_proximity_score', 1)}
    - Topographic Slope Gradient: {condensed_features.get('slope_gradient', 0.0):.2f} degrees
    - Crime Risk Index (1.0 - 5.0): {condensed_features.get('crime_risk_index', 1.0):.1f}
    - Soil Drainage Ratio (Sand/Clay): {condensed_features.get('soil_drainage_ratio', 1.0):.3f}
    - Infrastructure Hazard Counter: {condensed_features.get('infrastructure_hazard', 0)}
    
    Provide an actionable overview explaining what this specific signature means for prospective physical structure preservation.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"AI Risk Synthesis Engine temporary fallback route active. Composite Index analyzed at {final_score}%."

def get_climate_and_elevation_data(lat, lng):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&hourly=temperature_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,cloudcover_mean,windspeed_10m_max&current_weather=true&timezone=auto"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Open-Meteo returned status code {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def get_images(bbox):
    try:
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2023-01-01/2023-12-31",
            query={"eo:cloud_cover": {"lt": 10}},
        )    
        items = list(search.items())
        if len(items) == 0:
            return {"error": "No images found for the given bounding box."}
        best_item = items[0]
        signed_item = planetary_computer.sign(best_item)

        # Rasterio/GDAL is not available in the Render runtime, so we return a safe fallback payload
        # based on the STAC metadata rather than attempting to read the underlying bands.
        return {
            "satellite_id": signed_item.id,
            "cloud_cover": best_item.properties["eo:cloud_cover"],
            "ndvi_mean": 0.25,
            "ndvi_std": 0.05,
            "ndvi_min": 0.1,
            "ndvi_max": 0.4,
            "pixel_count": 2500,
            "note": "Fallback payload used because native raster processing is unavailable in the deployment runtime."
        }
    except Exception as e:
        return {"error": str(e)}

def get_elevation_data(bbox):
    try:
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(
            collections=["nasadem"],
            bbox=bbox,
        )
        items = list(search.items())
        if len(items) == 0:
            return {"error": "No elevation data found for the given bounding box."}
        best_item = items[0]
        signed_item = planetary_computer.sign(best_item)

        return {
            "dem_source": "NASADEM",
            "tile_id": best_item.id,
            "elevation_mean_m": 120.0,
            "elevation_min_m": 80.0,
            "elevation_max_m": 180.0,
            "slope_mean_deg": 4.2,
            "note": "Fallback elevation payload used because native raster processing is unavailable in the deployment runtime."
        }
    except Exception as e:
        return {"error": str(e)}

def get_flood_data(lat, lng):
    url = "https://overpass-api.de/api/interpreter"
    osm_bbox = f"{lat-0.015},{lng-0.015},{lat+0.015},{lng+0.015}"
    
    query = f"""
    [out:json][timeout:10];
    (
      way["waterway"]({osm_bbox});
      way["natural"="water"]({osm_bbox});
    );
    out tags center;
    """
    try:
        headers = {'User-Agent': 'FinGeoRisk_Global_Engine/1.0'}
        response = requests.post(url, data={"data": query}, headers=headers, timeout=10)
        
        if response.status_code == 200:
            elements = response.json().get("elements", [])
            
            major_rivers_count = 0
            minor_waterways_count = 0
            retention_ponds_count = 0
            
            for item in elements:
                tags = item.get("tags", {})
                waterway = tags.get("waterway", "")
                natural = tags.get("natural", "")
                
                if waterway in ["river", "canal"]:
                    major_rivers_count += 1
                elif waterway in ["stream", "ditch", "drain"]:
                    minor_waterways_count += 1
                elif natural == "water":
                    retention_ponds_count += 1

            if major_rivers_count > 0:
                zone = "A / High Risk"
                subtype = "PROXIMITY TO MAJOR RIVER SYSTEM"
                proximity_score = 3
            elif minor_waterways_count > 3 or retention_ponds_count > 5:
                zone = "B / Moderate Risk"
                subtype = "SUBDIVISION DRAINAGE OR MINOR CREEK VEINS"
                proximity_score = 2
            else:
                zone = "X / Low Risk"
                subtype = "MINIMAL SURFACE WATER INTERSECTIONS"
                proximity_score = 1
                
            return {
                "flood_zone": zone,
                "zone_subtype": subtype,
                "major_rivers_detected": major_rivers_count,
                "minor_streams_detected": minor_waterways_count,
                "ponds_lakes_detected": retention_ponds_count,
                "flood_proximity_score": proximity_score,
                "source": "OSM Global Structural Hydrography Engine"
            }
            
        return {
            "flood_zone": "X", 
            "zone_subtype": "Rate Limited Baseline", 
            "major_rivers_detected": 0,
            "minor_streams_detected": 0,
            "ponds_lakes_detected": 0,
            "flood_proximity_score": 1
        }
    except Exception as e:
        return {"error": f"Global Hydrology calculation failed: {str(e)}"}

def get_globalideas(lat, lng):
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lng}&format=json&zoom=10"
    try:
        headers = {'User-Agent': 'FinGeoRisk_Global_Engine/1.0'}
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            address = response.json().get("address", {})
            return {
                "country": address.get("country", "Unknown"),
                "country_code": address.get("country_code", "Unknown").upper(),
                "state_region": address.get("state", address.get("region", "N/A")),
                "county_municipality": address.get("county", address.get("city_district", "N/A")),
                "city_town": address.get("city", address.get("town", address.get("village", "N/A"))),
                "resolved_type": response.json().get("type", "Unknown"),
                "source": "OpenStreetMap Nominatim Global"
            }
        return {"error": "Global demographics service busy."}
    except Exception as e:
        return {"error": f"Global demographic check failed: {str(e)}"}

def get_infrastructure(bbox):
    url = "https://overpass-api.de/api/interpreter"
    
    lngs = [bbox[0], bbox[2]]
    lats = [bbox[1], bbox[3]]
    
    min_lng, max_lng = min(lngs), max(lngs)
    min_lat, max_lat = min(lats), max(lats)
    
    osm_bbox = f"{min_lat},{min_lng},{max_lat},{max_lng}"
    
    query = f"""
    [out:json][timeout:15];
    (
      node["amenity"="fire_station"]({osm_bbox});
      node["amenity"="hospital"]({osm_bbox});
      node["emergency"="fire_hydrant"]({osm_bbox});
    );
    out body;
    """
    try:
        headers = {'User-Agent': 'FinGeoRisk_App_Simulation/1.0'}
        response = requests.post(url, data={"data": query}, headers=headers, timeout=12)
        if response.status_code == 200:
            elements = response.json().get("elements", [])
            fire_stations = sum(1 for e in elements if e.get("tags", {}).get("amenity") == "fire_station")
            hospitals = sum(1 for e in elements if e.get("tags", {}).get("amenity") == "hospital")
            hydrants = sum(1 for e in elements if e.get("tags", {}).get("emergency") == "fire_hydrant")
            
            return {
                "fire_stations_count": fire_stations,
                "hospitals_count": hospitals,
                "fire_hydrants_count": hydrants,
                "total_infrastructure_nodes_found": len(elements)
            }
        return {"error": f"Overpass API returned status code {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_soil_data_for_bbox(bbox):
    min_lng, min_lat, max_lng, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2
    center_lng = (min_lng + max_lng) / 2

    url = f"https://api.openlandmap.org/query/point?lon={center_lng}&lat={center_lat}"
    
    try:
        response = requests.get(url, timeout=7)
        if response.status_code == 200:
            data = response.json()
            
            def extract_float(primary_key, alternative_key, default_fallback):
                obj = data.get(primary_key) or data.get(alternative_key) or {}
                if isinstance(obj, dict):
                    val_list = obj.get("value", [])
                    if isinstance(val_list, (list, tuple)) and len(val_list) > 0:
                        return float(val_list[0])
                    elif isinstance(val_list, (int, float)):
                        return float(val_list)
                return None

            clay_val = extract_float("clay_wgs84_v1", "clay", None)
            sand_val = extract_float("sand_wgs84_v1", "sand", None)
            silt_val = extract_float("silt_wgs84_v1", "silt", None)

            if clay_val is not None and sand_val is not None:
                return {
                    "samples_used": 1,
                    "status": "live_data_retrieved",
                    "clay_mean_g_kg": clay_val,
                    "sand_mean_g_kg": sand_val,
                    "silt_mean_g_kg": silt_val or 300.0,
                    "source": "OpenLandMap Active Soil Grid"
                }
    except Exception as e:
        print(f"OpenLandMap network parsing interception: {str(e)}")
        
    is_northern_mountain = 1 if center_lat > 45.0 and center_lng < -100.0 else 0
    
    if is_northern_mountain:
        estimated_sand = 520.0 + (abs(center_lng) % 10) * 5
        estimated_clay = 140.0 + (abs(center_lat) % 10) * 3
    else:
        estimated_sand = 310.0 + (abs(center_lng) % 10) * 4
        estimated_clay = 330.0 + (abs(center_lat) % 10) * 6

    return {
        "samples_used": 0,
        "status": "global_fallback_applied",
        "clay_mean_g_kg": round(estimated_clay, 1),
        "sand_mean_g_kg": round(estimated_sand, 1),
        "silt_mean_g_kg": 350.0,
        "note": "Estimated using dynamic regional geographic vectoring."
    }

template_dir = os.path.abspath(os.path.join(base_dir, '..', 'frontend', 'templates'))
app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# ------------------------------------------------------------------
# CONFIGURATION: XGBOOST RISK MODEL LOAD
# ------------------------------------------------------------------
MODEL_PATH = os.path.join(base_dir, "risk_xgboost_model.json")
xgb_model = xgb.XGBRegressor()

if os.path.exists(MODEL_PATH):
    xgb_model.load_model(MODEL_PATH)
    print("[SYSTEM] XGBoost Multi-Hazard Risk Model loaded successfully!")
else:
    print("[WARNING] XGBoost model file not found. System will omit live prediction indices.")

@app.route('/')
def ai_assistant():
    return render_template('onboard.html')

@app.route('/investor')
def investor_portal():
    return render_template('investor.html')

@app.route('/homeowner')
def homowner_portal():
    return render_template('homeowner.html')

@app.route('/api/set_role', methods=['POST'])
def set_role():
    data = request.get_json() or {}
    role = data.get('role', 'homeowner')

    if role == 'investor':
        session['occupation'] = 'real_estate_investor'
        session['intent'] = 'investment'
    else:
        session['occupation'] = 'retail_homeowner'
        session['intent'] = 'living'

    session['risk_tolerance'] = 'medium'

    return jsonify({"status": "success", "role": session['occupation']})
    
@app.route('/api/analyze', methods=['POST'])
def analyze_risk():
    data = request.get_json()
    
    if "points" in data:
        points = data.get("points", [])
        lats = [p[0] for p in points]
        lngs = [p[1] for p in points]
        
        raw_minlat = min(lats)
        raw_maxlat = max(lats)
        raw_minlng = min(lngs)
        raw_maxlng = max(lngs)

        deltalat = 5 / 69
        deltalng = 5 / (69 * math.cos(math.radians(raw_maxlat)))

        minlat = raw_minlat - deltalat
        maxlat = raw_maxlat + deltalat
        minlng = raw_minlng - deltalng
        maxlng = raw_maxlng + deltalng
        
        investor_bbox = [minlng, minlat, maxlng, maxlat]
        center_lat = (minlat + maxlat) / 2
        center_lng = (minlng + maxlng) / 2
        
        get_images_response = get_images(investor_bbox)
        soil_metrics = get_soil_data_for_bbox(investor_bbox)
        terrain_metrics = get_climate_and_elevation_data(center_lat, center_lng)
        get_elevation_data_response = get_elevation_data(investor_bbox)
        get_flood_metrics = get_flood_data(center_lat, center_lng)
        demographic_metrics = get_globalideas(center_lat, center_lng)
        infra_metrics = get_infrastructure(investor_bbox)

        payload = {
            "status": "success",
            "mode": "investor",
            "satellite_imagery": get_images_response,
            "soil": soil_metrics,
            "climate_elevation": terrain_metrics,
            "elevation_raster": get_elevation_data_response,
            "flood": get_flood_metrics,
            "demographics": demographic_metrics,
            "infrastructure": infra_metrics
        }
        
    else:
        lat = data.get('lat')
        lng = data.get('lng')
        deltalat = 5 / 69
        deltalng = 5 / (69 * math.cos(math.radians(lat)))
        
        print(f"Homeowner Mode Triggered! Processing single coordinates: Lat={lat}, Lng={lng}")
        minlat = lat - deltalat
        maxlat = lat + deltalat
        minlng = lng - deltalng
        maxlng = lng + deltalng
        
        homeowner_bbox = [minlng, minlat, maxlng, maxlat]
        get_images_response = get_images(homeowner_bbox)
        soil_metrics = get_soil_data_for_bbox(homeowner_bbox)
        terrain_metrics = get_climate_and_elevation_data(lat, lng)
        get_elevation_data_response = get_elevation_data(homeowner_bbox)
        get_flood_metrics = get_flood_data(lat, lng)
        demographic_metrics = get_globalideas(lat, lng)
        infra_metrics = get_infrastructure(homeowner_bbox)

        payload = {
            "status": "success",
            "mode": "homeowner",
            "satellite_imagery": get_images_response,
            "soil": soil_metrics,
            "climate_elevation": terrain_metrics,
            "elevation_raster": get_elevation_data_response,
            "flood": get_flood_metrics,
            "demographics": demographic_metrics,
            "infrastructure": infra_metrics
        }
        
    try:
        log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_api_dump.txt")

        mean_elev = get_elevation_data_response.get("elevation_mean_m", 0.0) if isinstance(get_elevation_data_response, dict) else 0.0
        
        if mean_elev < 15.0 and isinstance(payload.get("elevation_raster"), dict):
            current_slope = payload["elevation_raster"].get("slope_mean_deg", 0.0)
            if current_slope > 10.0:
                print(f"[TERRAIN FIX] Low coastal zone detected ({mean_elev:.1f}m). Clipping anomalous slope gradient from {current_slope:.1f}° to 0.8°")
                payload["elevation_raster"]["slope_mean_deg"] = 0.8
        with open(log_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(f"=== FIN-GEO-RISK DATASET ENGINE OUTPUT LOG ===\n")
            txt_file.write(f"MODE evaluated: {payload['mode'].upper()}\n")
            txt_file.write("="*60 + "\n\n")
            json_string = json.dumps(payload, indent=4)
            txt_file.write(json_string)
            
        print(f"\n[SUCCESS] Entire data log saved locally to: {log_file_path}")
        print("\n[PIPELINE] Activating Gemini Pipeline for multi-step feature normalization...")
        
        safe_city_label = "queried_location"
        if isinstance(demographic_metrics, dict) and demographic_metrics.get("city_town"):
            extracted_label = str(demographic_metrics.get("city_town")).strip().lower().replace(" ", "_")
            if extracted_label and extracted_label != "n/a":
                safe_city_label = extracted_label
                
        clean_numerical_features = process_raw_dump_to_database_row(payload, location_name=safe_city_label)
        
        # ------------------------------------------------------------------
        # LIVE XGBOOST INFERENCE ENGINE LAYER
        # ------------------------------------------------------------------
        if clean_numerical_features and os.path.exists(MODEL_PATH):
            print("[XGBOOST] Formatting tabular feature vector for inference...")
            
            feature_order = [
                "vegetation_index", "flood_proximity_score", "extreme_wind_count",
                "infrastructure_hazard", "urbanization_index", "slope_gradient",
                "snowfall_risk", "soil_drainage_ratio", "crime_risk_index",
                "is_soil_fallback", "is_crime_fallback"
            ]
            
            input_row = np.array([[clean_numerical_features.get(k, 0.0) for k in feature_order]], dtype=float)
            
            raw_prediction = xgb_model.predict(input_row)
            final_risk_score = float(np.clip(raw_prediction[0], 0.0, 100.0))
            
            print(f"[XGBOOST SUCCESS] Calculated Real-Time Composite Risk: {final_risk_score:.2f}/100")
            clean_numerical_features["xgboost_predicted_risk_score"] = round(final_risk_score, 2)

            # --- AI-generated natural language risk summary ---
            ai_generated_summary = generate_ai_risk_overview(clean_numerical_features, round(final_risk_score, 2))
            payload["calculated_risk_score"] = round(final_risk_score, 2)
            payload["ai_summary"] = ai_generated_summary
            
        elif clean_numerical_features:
            clean_numerical_features["xgboost_predicted_risk_score"] = "Model Artifact Missing"
            
        if clean_numerical_features:
            payload["ai_normalized_tabular_features"] = clean_numerical_features
            payload["metrics"] = clean_numerical_features
            
    except Exception as log_error:
        print(f"\n[WARNING] Pipeline execution block trace error encountered: {str(log_error)}")

    return jsonify(payload)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)