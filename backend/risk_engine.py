import math
import numpy as np
import xgboost as xgb

FEATURE_ORDER = [
    "vegetation_index",
    "flood_proximity_score",
    "extreme_wind_count",
    "infrastructure_hazard",
    "urbanization_index",
    "slope_gradient",
    "snowfall_risk",
    "soil_drainage_ratio",
    "crime_risk_index",
    "is_soil_fallback",
    "is_crime_fallback",
]


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_feature_vector(features):
    vector = []
    for name in FEATURE_ORDER:
        if name in {"is_soil_fallback", "is_crime_fallback"}:
            raw = features.get(name, 0)
            if isinstance(raw, bool):
                vector.append(1.0 if raw else 0.0)
            else:
                try:
                    vector.append(1.0 if int(raw) > 0 else 0.0)
                except (TypeError, ValueError):
                    vector.append(0.0)
        else:
            vector.append(_to_float(features.get(name), 0.0))
    return vector


def build_feature_map_from_payload(payload, prior_features=None):
    prior = dict(prior_features or {})
    satellite = payload.get("satellite_imagery", {}) or {}
    soil = payload.get("soil", {}) or {}
    elevation = payload.get("elevation_raster", {}) or {}
    flood = payload.get("flood", {}) or {}
    demographics = payload.get("demographics", {}) or {}
    infrastructure = payload.get("infrastructure", {}) or {}

    soil_ratio = 1.0
    if isinstance(soil, dict):
        sand = _to_float(soil.get("sand_mean_g_kg"), 0.0)
        clay = _to_float(soil.get("clay_mean_g_kg"), 0.0)
        if clay > 0:
            soil_ratio = sand / clay

    return {
        "vegetation_index": prior.get("vegetation_index", _to_float(satellite.get("ndvi_mean"), 0.0)),
        "flood_proximity_score": prior.get("flood_proximity_score", _to_float(flood.get("flood_proximity_score"), 1.0)),
        "extreme_wind_count": prior.get("extreme_wind_count", 0),
        "infrastructure_hazard": prior.get("infrastructure_hazard", _to_float(infrastructure.get("total_infrastructure_nodes_found"), 0.0)),
        "urbanization_index": prior.get("urbanization_index", _to_float(demographics.get("urban_growth"), 0.0) + _to_float(demographics.get("pop_density"), 0.0) / 1000.0),
        "slope_gradient": prior.get("slope_gradient", _to_float(elevation.get("slope_mean_deg"), 0.0)),
        "snowfall_risk": prior.get("snowfall_risk", 0.0),
        "soil_drainage_ratio": prior.get("soil_drainage_ratio", soil_ratio),
        "crime_risk_index": prior.get("crime_risk_index", 1.0),
        "is_soil_fallback": prior.get("is_soil_fallback", 1 if soil.get("status") == "global_fallback_applied" else 0),
        "is_crime_fallback": prior.get("is_crime_fallback", 0),
    }


def score_with_fallback(model, features):
    if model is not None:
        try:
            row = np.array([build_feature_vector(features)], dtype=float)
            dmatrix = xgb.DMatrix(row, feature_names=FEATURE_ORDER)
            prediction = model.predict(dmatrix)
            if prediction is not None and len(prediction) > 0:
                score = float(prediction[0])
                if math.isfinite(score):
                    return float(np.clip(score, 0.0, 100.0))
        except Exception as exc:
            print(f"[XGBOOST] Inference fallback triggered: {exc}")

    vegetation_index = _to_float(features.get("vegetation_index"), 0.0)
    flood = _to_float(features.get("flood_proximity_score"), 1.0)
    slope = _to_float(features.get("slope_gradient"), 0.0)
    crime = _to_float(features.get("crime_risk_index"), 1.0)
    soil_ratio = _to_float(features.get("soil_drainage_ratio"), 1.0)
    infra = _to_float(features.get("infrastructure_hazard"), 0.0)
    wind = _to_float(features.get("extreme_wind_count"), 0.0)
    urban = _to_float(features.get("urbanization_index"), 0.0)
    snowfall = _to_float(features.get("snowfall_risk"), 0.0)
    soil_fallback = 1 if features.get("is_soil_fallback") else 0
    crime_fallback = 1 if features.get("is_crime_fallback") else 0

    score = 8.0
    score += flood * 12.0
    score += max(0.0, crime - 1.0) * 10.0
    score += min(25.0, max(0.0, slope * 1.1))
    score += vegetation_index * 20.0
    score += min(20.0, infra * 3.0)
    score += wind * 6.0
    score += urban * 3.0
    score += snowfall * 7.0
    score += soil_fallback * 6.0
    score += crime_fallback * 4.0
    if soil_ratio < 1.2:
        score += 8.0
    elif soil_ratio > 3.5:
        score -= 4.0

    return float(np.clip(score, 0.0, 100.0))
