import json
import os

def generate_perfect_sandbox_dataset():
    sandbox_scenarios = {
        "Scenario_1_The_Tipping_Point": {
            "description": "High clay content combined with moderate slopes and heavy seasonal rain causing potential shifting vulnerabilities.",
            "raw_data": {
                "satellite_imagery": {
                    "satellite_id": "S2B_MSIL2A_20230514",
                    "cloud_cover": 1.2,
                    "ndvi_mean": 0.45,
                    "ndvi_std": 0.08,
                    "ndvi_min": 0.12,
                    "ndvi_max": 0.68,
                    "pixel_count": 45000
                },
                "soil": {
                    "samples_used": 1,
                    "raw_samples": [{
                        "properties": {
                            "clay": {"M": {"mean": [450]}},        # 45% clay
                            "bdod": {"M": {"mean": [1200]}},       # Standard density
                            "cec": {"M": {"mean": [210]}},
                            "nitrogen": {"M": {"mean": [180]}},
                            "ocd": {"M": {"mean": [310]}},
                            "phh2o": {"M": {"mean": [62]}},        # pH 6.2
                            "sand": {"M": {"mean": [250]}},
                            "silt": {"M": {"mean": [300]}}
                        }
                    }]
                },
                "climate_elevation": {
                    "current_weather": {"temperature": 22.5, "windspeed": 11.0},
                    "daily": {
                        "temperature_2m_max": [29.0],
                        "temperature_2m_min": [14.0],
                        "precipitation_sum": [75.5],               # Severe storm dump
                        "rain_sum": [75.5],
                        "snowfall_sum": [0.0],
                        "cloudcover_mean": [65.0],
                        "windspeed_10m_max": [24.0]
                    }
                },
                "elevation_raster": {
                    "dem_source": "NASADEM",
                    "tile_id": "n34w118",
                    "elevation_mean_m": 320.0,
                    "elevation_min_m": 290.0,
                    "elevation_max_m": 360.0,
                    "slope_mean_deg": 12.5                         # Moderate structural grade
                },
                "flood": {
                    "flood_zone": "X",
                    "zone_subtype": "AREA OF MINIMAL FLOOD HAZARD",
                    "fld_ar_id": "06037C_X_MOCK"
                },
                "demographics": {
                    "country": "United States",
                    "country_code": "USA",
                    "pop_density": 380.5,
                    "urban_growth": 0.8,
                    "gdp_per_capita": 76000.0
                },
                "infrastructure": {
                    "fire_stations_count": 0,
                    "hospitals_count": 0,
                    "fire_hydrants_count": 0,
                    "total_infrastructure_nodes_found": 0
                }
            }
        },

        "Scenario_2_Flash_Flood_Basin": {
            "description": "Steep hillside gradient nestled in a regulatory coastal high hazard area with extreme storm triggers.",
            "raw_data": {
                "satellite_imagery": {
                    "satellite_id": "S2A_MSIL2A_20230822",
                    "cloud_cover": 4.5,
                    "ndvi_mean": 0.15,
                    "ndvi_std": 0.03,
                    "ndvi_min": -0.05,
                    "ndvi_max": 0.32,
                    "pixel_count": 62000
                },
                "soil": {
                    "samples_used": 1,
                    "raw_samples": [{
                        "properties": {
                            "clay": {"M": {"mean": [120]}},
                            "bdod": {"M": {"mean": [1400]}},
                            "cec": {"M": {"mean": [110]}},
                            "nitrogen": {"M": {"mean": [95]}},
                            "ocd": {"M": {"mean": [150]}},
                            "phh2o": {"M": {"mean": [74]}},
                            "sand": {"M": {"mean": [650]}},        # Highly fluid sand profile
                            "silt": {"M": {"mean": [230]}}
                        }
                    }]
                },
                "climate_elevation": {
                    "current_weather": {"temperature": 18.0, "windspeed": 35.0},
                    "daily": {
                        "temperature_2m_max": [22.0],
                        "temperature_2m_min": [11.0],
                        "precipitation_sum": [110.0],              # Torrential flash rainfall
                        "rain_sum": [110.0],
                        "snowfall_sum": [0.0],
                        "cloudcover_mean": [98.0],
                        "windspeed_10m_max": [45.0]
                    }
                },
                "elevation_raster": {
                    "dem_source": "NASADEM",
                    "tile_id": "n34w118",
                    "elevation_mean_m": 45.0,
                    "elevation_min_m": 12.0,
                    "elevation_max_m": 110.0,
                    "slope_mean_deg": 28.0                         # High runoff grade
                },
                "flood": {
                    "flood_zone": "AE",                             # 100-Year flood standard
                    "zone_subtype": "FLOODWAY",
                    "fld_ar_id": "06037C_AE_9921"
                },
                "demographics": {
                    "country": "United States",
                    "country_code": "USA",
                    "pop_density": 110.2,
                    "urban_growth": 0.3,
                    "gdp_per_capita": 76000.0
                },
                "infrastructure": {
                    "fire_stations_count": 1,
                    "hospitals_count": 0,
                    "fire_hydrants_count": 4,
                    "total_infrastructure_nodes_found": 5
                }
            }
        },

        "Scenario_3_Wildfire_Interface_Trap": {
            "description": "Drought environment displaying high fuel load indices coupled with zero active asset buffers.",
            "raw_data": {
                "satellite_imagery": {
                    "satellite_id": "S2A_MSIL2A_20230905",
                    "cloud_cover": 0.1,
                    "ndvi_mean": 0.82,                             # Critical dry fuel load metric
                    "ndvi_std": 0.11,
                    "ndvi_min": 0.45,
                    "ndvi_max": 0.95,
                    "pixel_count": 51000
                },
                "soil": {
                    "samples_used": 1,
                    "raw_samples": [{
                        "properties": {
                            "clay": {"M": {"mean": [110]}},
                            "bdod": {"M": {"mean": [1350]}},
                            "cec": {"M": {"mean": [140]}},
                            "nitrogen": {"M": {"mean": [60]}},
                            "ocd": {"M": {"mean": [420]}},
                            "phh2o": {"M": {"mean": [51]}},
                            "sand": {"M": {"mean": [720]}},        # Arid sand mix
                            "silt": {"M": {"mean": [170]}}
                        }
                    }]
                },
                "climate_elevation": {
                    "current_weather": {"temperature": 38.5, "windspeed": 42.0},
                    "daily": {
                        "temperature_2m_max": [41.0],
                        "temperature_2m_min": [26.0],
                        "precipitation_sum": [0.0],                # Deep drought anomaly
                        "rain_sum": [0.0],
                        "snowfall_sum": [0.0],
                        "cloudcover_mean": [5.0],
                        "windspeed_10m_max": [65.0]                # Fanning wind speed
                    }
                },
                "elevation_raster": {
                    "dem_source": "NASADEM",
                    "tile_id": "n34w119",
                    "elevation_mean_m": 850.0,
                    "elevation_min_m": 720.0,
                    "elevation_max_m": 990.0,
                    "slope_mean_deg": 14.2
                },
                "flood": {
                    "flood_zone": "X",
                    "zone_subtype": "AREA OF MINIMAL FLOOD HAZARD",
                    "fld_ar_id": "N/A"
                },
                "demographics": {
                    "country": "United States",
                    "country_code": "USA",
                    "pop_density": 12.4,
                    "urban_growth": 1.4,
                    "gdp_per_capita": 76000.0
                },
                "infrastructure": {
                    "fire_stations_count": 0,                      # Isolated risk vulnerability
                    "hospitals_count": 0,
                    "fire_hydrants_count": 0,
                    "total_infrastructure_nodes_found": 0
                }
            }
        },

        "Scenario_4_Severe_Landslide_Chute": {
            "description": "Critically steep incline containing shear-vulnerable fine silt that collapses easily when saturated.",
            "raw_data": {
                "satellite_imagery": {
                    "satellite_id": "S2B_MSIL2A_20231102",
                    "cloud_cover": 2.1,
                    "ndvi_mean": 0.45,
                    "ndvi_std": 0.05,
                    "ndvi_min": 0.22,
                    "ndvi_max": 0.58,
                    "pixel_count": 39000
                },
                "soil": {
                    "samples_used": 1,
                    "raw_samples": [{
                        "properties": {
                            "clay": {"M": {"mean": [100]}},
                            "bdod": {"M": {"mean": [1280]}},
                            "cec": {"M": {"mean": [190]}},
                            "nitrogen": {"M": {"mean": [110]}},
                            "ocd": {"M": {"mean": [280]}},
                            "phh2o": {"M": {"mean": [59]}},
                            "sand": {"M": {"mean": [250]}},
                            "silt": {"M": {"mean": [650]}}         # Silt liquefies easily under stress
                        }
                    }]
                },
                "climate_elevation": {
                    "current_weather": {"temperature": 9.0, "windspeed": 12.0},
                    "daily": {
                        "temperature_2m_max": [12.0],
                        "temperature_2m_min": [3.0],
                        "precipitation_sum": [95.0],               # Severe mudslide rainfall
                        "rain_sum": [95.0],
                        "snowfall_sum": [0.0],
                        "cloudcover_mean": [95.0],
                        "windspeed_10m_max": [18.0]
                    }
                },
                "elevation_raster": {
                    "dem_source": "NASADEM",
                    "tile_id": "n46w121",
                    "elevation_mean_m": 1450.0,
                    "elevation_min_m": 1200.0,
                    "elevation_max_m": 1800.0,
                    "slope_mean_deg": 38.5                         # Critically dangerous grade
                },
                "flood": {
                    "flood_zone": "X",
                    "zone_subtype": "AREA OF MINIMAL FLOOD HAZARD",
                    "fld_ar_id": "N/A"
                },
                "demographics": {
                    "country": "United States",
                    "country_code": "USA",
                    "pop_density": 2.1,
                    "urban_growth": -0.1,
                    "gdp_per_capita": 76000.0
                },
                "infrastructure": {
                    "fire_stations_count": 0,
                    "hospitals_count": 0,
                    "fire_hydrants_count": 0,
                    "total_infrastructure_nodes_found": 0
                }
            }
        },

        "Scenario_5_Shallow_Bedrock_Excavation_Trap": {
            "description": "High soil bulk density (bdod) profile signaling shallow bedrock layers, leading to significant construction cost overruns.",
            "raw_data": {
                "satellite_imagery": {
                    "satellite_id": "S2A_MSIL2A_20230412",
                    "cloud_cover": 1.1,
                    "ndvi_mean": 0.32,
                    "ndvi_std": 0.04,
                    "ndvi_min": 0.15,
                    "ndvi_max": 0.45,
                    "pixel_count": 48000
                },
                "soil": {
                    "samples_used": 1,
                    "raw_samples": [{
                        "properties": {
                            "clay": {"M": {"mean": [200]}},
                            "bdod": {"M": {"mean": [1650]}},       # Highly compacted rocky base layer
                            "cec": {"M": {"mean": [150]}},
                            "nitrogen": {"M": {"mean": [85]}},
                            "ocd": {"M": {"mean": [110]}},
                            "phh2o": {"M": {"mean": [68]}},
                            "sand": {"M": {"mean": [400]}},
                            "silt": {"M": {"mean": [400]}}
                        }
                    }]
                },
                "climate_elevation": {
                    "current_weather": {"temperature": 15.0, "windspeed": 6.0},
                    "daily": {
                        "temperature_2m_max": [19.0],
                        "temperature_2m_min": [8.0],
                        "precipitation_sum": [8.0],
                        "rain_sum": [8.0],
                        "snowfall_sum": [0.0],
                        "cloudcover_mean": [40.0],
                        "windspeed_10m_max": [11.0]
                    }
                },
                "elevation_raster": {
                    "dem_source": "NASADEM",
                    "tile_id": "n40w105",
                    "elevation_mean_m": 410.0,
                    "elevation_min_m": 380.0,
                    "elevation_max_m": 450.0,
                    "slope_mean_deg": 6.1
                },
                "flood": {
                    "flood_zone": "X",
                    "zone_subtype": "AREA OF MINIMAL FLOOD HAZARD",
                    "fld_ar_id": "N/A"
                },
                "demographics": {
                    "country": "United States",
                    "country_code": "USA",
                    "pop_density": 850.4,
                    "urban_growth": 2.1,
                    "gdp_per_capita": 76000.0
                },
                "infrastructure": {
                    "fire_stations_count": 1,
                    "hospitals_count": 0,
                    "fire_hydrants_count": 4,
                    "total_infrastructure_nodes_found": 5
                }
            }
        }
    }
