import os
import sqlite3
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

app = Flask(__name__)
app.secret_key = "hemomatch_secret_key_secure_and_stable"

# Database configuration
DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "hemomatch.db")

# Configurable Hospital Coordinates (Step 2)
# This represents the hospital receiving the blood request.
HOSPITAL_LAT = 12.9716
HOSPITAL_LON = 77.5946

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS donors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            blood_group TEXT NOT NULL,
            phone TEXT NOT NULL,
            location TEXT NOT NULL,
            response_rate REAL,
            consistency REAL,
            last_active_days INTEGER,
            availability INTEGER,
            donation_frequency INTEGER,
            last_donation_date TEXT,
            distance REAL,
            health_status INTEGER,
            age INTEGER,
            weight REAL,
            current_latitude REAL,
            current_longitude REAL,
            accepted_request INTEGER DEFAULT 0
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS emergency_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            blood_group TEXT NOT NULL,
            hospital_lat REAL NOT NULL,
            hospital_lon REAL NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        )
    ''')
    conn.commit()

    # Dynamic schema migration for existing tables
    cursor = conn.execute("PRAGMA table_info(donors)")
    columns = [row[1] for row in cursor.fetchall()]
    
    new_cols = {
        "last_donation_date": "TEXT",
        "distance": "REAL",
        "health_status": "INTEGER",
        "age": "INTEGER",
        "weight": "REAL",
        "current_latitude": "REAL",
        "current_longitude": "REAL",
        "accepted_request": "INTEGER DEFAULT 0"
    }
    
    for col_name, col_type in new_cols.items():
        if col_name not in columns:
            conn.execute(f"ALTER TABLE donors ADD COLUMN {col_name} {col_type}")
            conn.commit()
            print(f"Migration: Added column '{col_name}' ({col_type}) to donors table.")
            
    # Dynamic schema migration for emergency_requests status column
    cursor2 = conn.execute("PRAGMA table_info(emergency_requests)")
    req_columns = [row[1] for row in cursor2.fetchall()]
    if "status" not in req_columns:
        conn.execute("ALTER TABLE emergency_requests ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        conn.commit()
        print("Migration: Added column 'status' (TEXT) to emergency_requests table.")
            
    conn.close()

# Auto-initialize database on startup
init_db()

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Modular Haversine distance calculator in kilometers (Step 3).
    Privacy-Aware/Safety design:
    - Location is temporary and used only during active matching to improve proximity accuracy.
    - No passive, continuous GPS tracking.
    """
    try:
        if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
            return None
        
        import math
        lat1_rad = math.radians(float(lat1))
        lon1_rad = math.radians(float(lon1))
        lat2_rad = math.radians(float(lat2))
        lon2_rad = math.radians(float(lon2))
        
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371.0 # Radius of earth in kilometers
        
        return round(c * r, 1)
    except Exception as e:
        print(f"Error in calculate_distance: {str(e)}")
        return None

# Demand Prediction Model Setup (Linear Regression)
import pandas as pd
from sklearn.linear_model import LinearRegression

# Registration-based Demand Model (Backward Compatibility)
REGISTRATION_MODEL = None
REG_LAST_DAYS_SINCE_START = 0
CHART_LABELS = []
CHART_VALUES = []

def train_registration_model():
    global REGISTRATION_MODEL, REG_LAST_DAYS_SINCE_START, CHART_LABELS, CHART_VALUES
    try:
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blood_donor_dataset.csv")
        if not os.path.exists(dataset_path):
            print("Warning: blood_donor_dataset.csv not found. Prediction model not trained.")
            return

        # Load dataset
        df = pd.read_csv(dataset_path)
        if 'created_at' not in df.columns:
            print("Warning: 'created_at' column not in dataset. Model not trained.")
            return

        # Group registrations by date to represent daily demand (requests/registrations)
        df_grouped = df.groupby('created_at').size().reset_index(name='number_of_requests')
        df_grouped.rename(columns={'created_at': 'date'}, inplace=True)
        
        # Convert date to datetime
        df_grouped['date'] = pd.to_datetime(df_grouped['date'])
        df_grouped = df_grouped.sort_values('date')

        # Create days_since_start numerical feature
        min_date = df_grouped['date'].min()
        df_grouped['days_since_start'] = (df_grouped['date'] - min_date).dt.days

        # Define features and target
        X = df_grouped[['days_since_start']]
        y = df_grouped['number_of_requests']

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Store in global variables
        REGISTRATION_MODEL = model
        REG_LAST_DAYS_SINCE_START = int(df_grouped['days_since_start'].max())

        # Group last 30 active days of historical data for Chart.js
        historical_last_30 = df_grouped.tail(30)
        CHART_LABELS = [d.strftime('%Y-%m-%d') for d in historical_last_30['date']]
        CHART_VALUES = list(historical_last_30['number_of_requests'])

        print("Registration demand model successfully trained.")
    except Exception as e:
        print(f"Error training registration model: {str(e)}")

def predict_registration_demand(days_ahead):
    if REGISTRATION_MODEL is None:
        raise ValueError("Registration model not trained.")
    target_day = REG_LAST_DAYS_SINCE_START + days_ahead
    X_pred = pd.DataFrame([[target_day]], columns=['days_since_start'])
    prediction = REGISTRATION_MODEL.predict(X_pred)[0]
    return max(0, int(round(prediction)))

def predict_demand(days_ahead):
    """Wrapper function to preserve original registration model requests."""
    return predict_registration_demand(days_ahead)

def get_future_prediction():
    try:
        return predict_registration_demand(1)
    except Exception as e:
        print(f"Prediction failed, returning default: {str(e)}")
        return 3 # Safe default value if model fails

# ==============================================================================
# FUTURE BLOOD DEMAND ML PREDICTION SYSTEM (UPGRADED MODULE)
# ==============================================================================
# Architectural Commentary:
# 1. Why Blood Demand Forecasting Matters:
#    Blood products have strict expiration dates (e.g., 35-42 days for red blood cells).
#    Over-stocking leads to catastrophic clinical waste, while under-stocking puts patient
#    lives at immediate risk. Machine learning forecasting bridges this clinical gap.
#
# 2. Why Model Each Blood Group Separately:
#    Different blood groups have highly divergent demographic prevalence and request frequencies.
#    For instance, O+ and A+ requests represent a massive proportion of hospital demand, whereas
#    AB- requests are exceptionally rare. Distinct linear models capture these unique channels.
#
# 3. Why Linear Regression as a Baseline:
#    Linear Regression serves as an extremely lightweight, high-performance, and mathematically
#    transparent baseline forecasting model. It trains in milliseconds on startup without requiring 
#    dedicated GPU infrastructure or external microservices.
#
# TODO / Future Production Enhancements:
# - Prophet (Meta) for rich holiday and weekly seasonality analysis.
# - XGBoost / LightGBM for non-linear correlation and weather/location feature support.
# - LSTM (RNN) Deep Learning models if high-frequency time series sequencing is needed.
# ==============================================================================

DEMAND_MODELS = {}
IS_SYNTHETIC_DATA = False

def create_default_blood_demand_csv(filepath):
    global IS_SYNTHETIC_DATA
    import random
    from datetime import date, timedelta
    
    blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    start_date = date.today() - timedelta(days=60)
    
    # Clearly mark fallback dataset as Synthetic Demo Dataset
    rows = ["# Synthetic Demo Dataset", "date,blood_group,number_of_requests"]
    
    for i in range(61):
        curr_date = start_date + timedelta(days=i)
        date_str = curr_date.strftime("%Y-%m-%d")
        for bg in blood_groups:
            if bg in ["A+", "O+"]:
                reqs = random.randint(15, 35)
            elif bg in ["A-", "B-", "O-"]:
                reqs = random.randint(5, 15)
            elif bg in ["AB+"]:
                reqs = random.randint(8, 18)
            else: # AB-
                reqs = random.randint(2, 8)
                
            rows.append(f"{date_str},{bg},{reqs}")
            
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"Default blood demand synthetic demo dataset created at: {filepath}")
    IS_SYNTHETIC_DATA = True

def load_demand_data():
    global IS_SYNTHETIC_DATA
    csv_path = os.path.join(DATABASE_DIR, "blood_demand.csv")
    if not os.path.exists(csv_path):
        create_default_blood_demand_csv(csv_path)
        IS_SYNTHETIC_DATA = True
    else:
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line.startswith("# Synthetic Demo Dataset") or "synthetic" in first_line.lower():
                    IS_SYNTHETIC_DATA = True
                else:
                    IS_SYNTHETIC_DATA = False
        except Exception:
            IS_SYNTHETIC_DATA = False
    
    try:
        # Ignore comments starting with '#' (our Synthetic Demo Dataset marker)
        df = pd.read_csv(csv_path, comment='#')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error loading blood demand data: {str(e)}")
        return pd.DataFrame()

def train_demand_model():
    global DEMAND_MODELS
    try:
        df = load_demand_data()
        if df.empty:
            print("Warning: Blood demand data is empty. Models not trained.")
            return

        # Train one lightweight model per blood group
        DEMAND_MODELS = {}
        grouped = df.groupby(['date', 'blood_group'])['number_of_requests'].sum().reset_index()
        
        for bg in grouped['blood_group'].unique():
            df_bg = grouped[grouped['blood_group'] == bg].copy()
            df_bg = df_bg.sort_values('date')
            min_date = df_bg['date'].min()
            df_bg['days_since_start'] = (df_bg['date'] - min_date).dt.days
            
            X = df_bg[['days_since_start']]
            y = df_bg['number_of_requests']
            
            model = LinearRegression()
            model.fit(X, y)
            
            DEMAND_MODELS[bg] = {
                "model": model,
                "min_date": min_date,
                "last_day": int(df_bg['days_since_start'].max())
            }
        print("Modular Blood Group Demand Models successfully trained and cached.")
    except Exception as e:
        print(f"Error training modular blood group demand models: {str(e)}")

def predict_future_demand(blood_group, days_ahead=1):
    try:
        bg = blood_group.strip().upper()
        if isinstance(DEMAND_MODELS, dict) and bg in DEMAND_MODELS:
            bg_info = DEMAND_MODELS[bg]
            model = bg_info["model"]
            last_day = bg_info["last_day"]
            
            target_day = last_day + days_ahead
            X_pred = pd.DataFrame([[target_day]], columns=['days_since_start'])
            pred = model.predict(X_pred)[0]
            return max(1, int(round(pred)))
        else:
            # Safe fallback defaults
            fallbacks = {
                "A+": 24, "A-": 8, "B+": 16, "B-": 6,
                "AB+": 12, "AB-": 4, "O+": 28, "O-": 10
            }
            base = fallbacks.get(bg, 15)
            return max(1, int(round(base)))
    except Exception as e:
        print(f"Error in predict_future_demand: {str(e)}")
        return 12

# Auto-train models on startup
train_registration_model()
train_demand_model()

@app.route("/")
def home():
    return render_template("index.html")

def classify_compatibility(patient_bg, donor_bg):
    """
    Classifies standard blood transfusion compatibility.
    Returns:
       - "exact" if the blood groups match exactly.
       - "compatible" if the donor group is compatible with the patient group.
       - "not_compatible" if transfusion is medical disallowed.
    """
    if not patient_bg or not donor_bg:
        return "not_compatible"
        
    p_bg = patient_bg.strip().upper()
    d_bg = donor_bg.strip().upper()
    
    if p_bg == d_bg:
        return "exact"
        
    # Medical standard compatible mappings (excluding exact match)
    compatible_map = {
        "A+": {"A-", "O+", "O-"},
        "A-": {"O-"},
        "B+": {"B-", "O+", "O-"},
        "B-": {"O-"},
        "AB+": {"A+", "A-", "B+", "B-", "AB-", "O+", "O-"}, # all other groups
        "AB-": {"A-", "B-", "O-"},
        "O+": {"O-"},
        "O-": set() # No other compatible group except itself (which is exact)
    }
    
    allowed = compatible_map.get(p_bg, set())
    if d_bg in allowed:
        return "compatible"
        
    return "not_compatible"

def is_compatible(patient_bg, donor_bg):
    """Simple boolean backward compatibility check wrapper."""
    return classify_compatibility(patient_bg, donor_bg) != "not_compatible"

def get_compatibility_score(match_type):
    """Compatibility Score: exact -> 1.0, compatible -> 0.7."""
    if match_type == "exact":
        return 1.0
    elif match_type == "compatible":
        return 0.7
    return 0.0

def get_donation_gap_score(last_donation_date_val):
    """
    Donation Gap Score (20%):
    - Medical Reasoning: Whole blood donations require a 56-day (8 weeks) recovery period to replenish red blood cells.
    - >= 56 days -> 1.0
    - < 56 days -> score = 0.2 + 0.8 * (days / 56)
    - Default to 30 days if missing.
    """
    if last_donation_date_val:
        try:
            parsed_date = datetime.datetime.strptime(last_donation_date_val, "%Y-%m-%d").date()
            days_since_donation = (datetime.date.today() - parsed_date).days
            days_since_donation = max(0, days_since_donation)
        except Exception:
            days_since_donation = 30  # Default to neutral 30 days on parse error
    else:
        days_since_donation = 30  # Default to neutral 30 days if missing

    if days_since_donation >= 56:
        score = 1.0
    else:
        score = 0.2 + 0.8 * (days_since_donation / 56.0)

    return max(0.0, min(score, 1.0))

def get_distance_score(distance):
    """
    Distance Score (15%):
    - Proximity is important but should not dominate availability.
    - Smooth decay with fairness floor: score = max(0.3, 1 / (1 + (distance / 5)))
    - Default distance = 5 km if missing.
    """
    if distance is None:
        dist_val = 5.0
    else:
        try:
            dist_val = max(0.0, float(distance))
        except (ValueError, TypeError):
            dist_val = 5.0

    score = 1.0 / (1.0 + (dist_val / 5.0))
    score = max(0.3, score)
    return max(0.0, min(score, 1.0))

def get_health_score(health_status):
    """
    Health Score (15%):
    - Healthy -> 1.0
    - Unknown -> 0.5
    - Not healthy -> EXCLUDE donor before scoring (handled in matching loop)
    """
    if health_status == 1:
        return 1.0
    elif health_status == 0:
        return 0.0
    else:
        return 0.5

def get_age_score(age):
    """
    Age Score (10%):
    - Gradual penalty past 50:
      if age <= 50 -> 1.0
      else -> max(0.3, 1.0 - (age - 50) * 0.02)
    - Default score = 1.0 if missing.
    """
    if age is None:
        return 1.0
    try:
        age_val = int(age)
        if age_val <= 50:
            score = 1.0
        else:
            score = 1.0 - (age_val - 50) * 0.02
            score = max(0.3, score)
    except (ValueError, TypeError):
        return 1.0

    return max(0.0, min(score, 1.0))

def get_weight_score(weight):
    """
    Weight Score (10%):
    - Safe normalization:
      if weight >= 70 -> 1.0
      elif weight >= 50 -> 0.5 + 0.5 * ((weight - 50) / 20)
      else -> 0.0 (invalid, should already be filtered)
    - Default score = 1.0 if missing.
    """
    if weight is None:
        return 1.0
    try:
        weight_val = float(weight)
        if weight_val >= 70.0:
            score = 1.0
        elif weight_val >= 50.0:
            score = 0.5 + 0.5 * ((weight_val - 50.0) / 20.0)
        else:
            score = 0.0
    except (ValueError, TypeError):
        return 1.0

    return max(0.0, min(score, 1.0))

def calculate_donor_score(donor, match_type="exact"):
    """
    Refined Unified Scoring Model combining 6 normalized parameters:
    - Compatibility Score (30%)
    - Donation Gap Score (20%)
    - Distance Score (15%)
    - Health Score (15%)
    - Age Score (10%)
    - Weight Score (10%)
    """
    comp_s = get_compatibility_score(match_type)
    gap_s = get_donation_gap_score(donor.get("last_donation_date"))
    dist_s = get_distance_score(donor.get("distance"))
    health_s = get_health_score(donor.get("health_status", 2))
    age_s = get_age_score(donor.get("age"))
    weight_s = get_weight_score(donor.get("weight"))

    weighted_sum = (
        (comp_s * 0.30) +
        (gap_s * 0.20) +
        (dist_s * 0.15) +
        (health_s * 0.15) +
        (age_s * 0.10) +
        (weight_s * 0.10)
    )

    final_score = weighted_sum * 100.0
    final_score = max(0.0, min(final_score, 100.0))
    return round(final_score, 1)

def generate_explanation(donor, match_type):
    """Generates a patient-centric, clinical explanation of why the donor was recommended."""
    reasons = []
    
    # Match Type
    if match_type == "exact":
        reasons.append("Exact match")
    else:
        reasons.append("Compatible blood group")
        
    # Distance
    dist = donor.get("distance")
    if dist is not None:
        if dist <= 2.0:
            reasons.append("extremely nearby")
        elif dist <= 5.0:
            reasons.append("nearby")
        else:
            reasons.append(f"located {dist} km away")
    else:
        reasons.append("proximity verified")
        
    # Health Status
    health = donor.get("health_status", 2)
    if health == 1:
        reasons.append("healthy clinical profile")
        
    # Donation gap
    last_donation = donor.get("last_donation_date")
    if last_donation:
        try:
            parsed_date = datetime.datetime.strptime(last_donation, "%Y-%m-%d").date()
            days = (datetime.date.today() - parsed_date).days
            if days >= 56:
                reasons.append("fully well-rested (56+ days since last donation)")
            else:
                reasons.append(f"partially rested ({days} days since last donation)")
        except Exception:
            reasons.append("acceptable donation interval")
    else:
        reasons.append("acceptable donation interval")
        
    if reasons:
        summary = ", ".join(reasons)
        return summary[0].upper() + summary[1:]
    return "Eligible and ready compatible donor."

@app.route("/find", methods=["GET", "POST"])
def find():
    # Fetch all donors for form dropdowns or general context
    all_donors = []
    try:
        conn = get_db_connection()
        cursor = conn.execute("SELECT id, name, blood_group, current_latitude, current_longitude, distance, accepted_request FROM donors ORDER BY name ASC")
        all_donors = [dict(row) for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"Error fetching donors: {str(e)}")

    if request.method == "POST":
        patient_bg = request.form.get("blood_group", "").strip()
        emergency_mode = request.form.get("emergency_mode") == "true"
        is_fresh_search = request.form.get("fresh_search", "true") == "true"
        
        if not patient_bg:
            flash("Please select a blood group to search.", "danger")
            return render_template("find.html", active_responders=[], broadcast_log=[], searched_bg="", search_performed=True, emergency_mode=emergency_mode, all_donors=all_donors)

        try:
            # Step 1: Hospital emergency blood request creation and logging
            if is_fresh_search:
                try:
                    conn = get_db_connection()
                    # Reset previous simulation states to start fresh (clinical workflow simulation)
                    conn.execute("UPDATE donors SET accepted_request = 0, current_latitude = NULL, current_longitude = NULL")
                    
                    # This represents a live hospital emergency blood request.
                    conn.execute('''
                        INSERT INTO emergency_requests (blood_group, hospital_lat, hospital_lon, timestamp, status)
                        VALUES (?, ?, ?, ?, 'active')
                    ''', (patient_bg, HOSPITAL_LAT, HOSPITAL_LON, datetime.datetime.now().isoformat(), 'active'))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"Error logging emergency request: {str(e)}")

            # Fetch matching candidates
            conn = get_db_connection()
            cursor = conn.execute("SELECT * FROM donors")
            all_donors_rows = cursor.fetchall()
            conn.close()

            threshold = 25.0 if emergency_mode else 40.0
            
            active_responders = []
            broadcast_log = []
            
            for row in all_donors_rows:
                donor = dict(row)
                
                # Exclude unhealthy donors completely from search matches before scoring
                if donor.get("health_status") == 0:
                    continue

                match_type = classify_compatibility(patient_bg, donor["blood_group"])
                if match_type in ["exact", "compatible"]:
                    score = calculate_donor_score(donor, match_type=match_type)
                    
                    # Filter out matches below threshold
                    if score >= threshold:
                        # Step 5: Calculate donor live distance from hospital
                        live_dist = calculate_distance(donor.get("current_latitude"), donor.get("current_longitude"), HOSPITAL_LAT, HOSPITAL_LON)
                        if live_dist is None:
                            # Fallback to stored approximate distance from db column
                            live_dist = donor.get("distance")
                        if live_dist is None:
                            live_dist = 999.0
                        
                        donor["live_distance_km"] = round(float(live_dist), 1)
                        donor["match_type"] = match_type
                        donor["score"] = score
                        donor["label"] = "Best Match" if match_type == "exact" else "Highly Compatible"
                        donor["why_selected"] = generate_explanation(donor, match_type)
                        
                        # Set response status display string (Step 9)
                        if donor.get("accepted_request") == 1:
                            donor["response_status"] = "Accepted"
                            active_responders.append(donor)
                        elif donor.get("accepted_request") == -1:
                            donor["response_status"] = "Declined"
                        else:
                            donor["response_status"] = "Pending Response"
                            
                        broadcast_log.append(donor)

            # Step 7: Sort accepted donors: match_type, score, and ascending proximity (LOWER distance first)
            def get_active_sort_key(d):
                match_priority = 0 if d["match_type"] == "exact" else 1
                score_priority = -d["score"]
                dist_val = d.get("live_distance_km") if d.get("live_distance_km") is not None else 999.0
                
                last_don_date = d.get("last_donation_date")
                if last_don_date:
                    try:
                        parsed_date = datetime.datetime.strptime(last_don_date, "%Y-%m-%d").date()
                        days_since = (datetime.date.today() - parsed_date).days
                        days_since = max(0, days_since)
                    except Exception:
                        days_since = 30
                else:
                    days_since = 30
                gap_priority = -days_since
                
                age_val = d.get("age") if d.get("age") is not None else 35
                
                return (match_priority, score_priority, dist_val, gap_priority, age_val)

            active_responders.sort(key=get_active_sort_key)

            return render_template("find.html", active_responders=active_responders, broadcast_log=broadcast_log, searched_bg=patient_bg, search_performed=True, emergency_mode=emergency_mode, all_donors=all_donors)

        except Exception as e:
            flash(f"An error occurred while finding donors: {str(e)}", "danger")
            return render_template("find.html", active_responders=[], broadcast_log=[], searched_bg=patient_bg, search_performed=True, emergency_mode=emergency_mode, all_donors=all_donors)

    return render_template("find.html", active_responders=[], broadcast_log=[], searched_bg="", search_performed=False, emergency_mode=False, all_donors=all_donors)

@app.route("/api/find_donor", methods=["POST"])
def api_find_donor():
    try:
        data = request.get_json() or {}
        patient_bg = data.get("blood_group", "").strip()
        emergency_mode = bool(data.get("emergency_mode", False))
        
        if not patient_bg:
            return jsonify({"error": "Blood group is required"}), 400
            
        # Step 1: Log emergency request internally
        try:
            conn = get_db_connection()
            # This represents a live hospital emergency blood request.
            conn.execute('''
                INSERT INTO emergency_requests (blood_group, hospital_lat, hospital_lon, timestamp, status)
                VALUES (?, ?, ?, ?, 'active')
            ''', (patient_bg, HOSPITAL_LAT, HOSPITAL_LON, datetime.datetime.now().isoformat(), 'active'))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging emergency request in API: {str(e)}")

        conn = get_db_connection()
        cursor = conn.execute("SELECT * FROM donors")
        all_donors_rows = cursor.fetchall()
        conn.close()
        
        threshold = 25.0 if emergency_mode else 40.0
        
        compatible_donors = []
        for row in all_donors_rows:
            donor = dict(row)
            if donor.get("health_status") == 0:
                continue
                
            match_type = classify_compatibility(patient_bg, donor["blood_group"])
            if match_type in ["exact", "compatible"]:
                score = calculate_donor_score(donor, match_type=match_type)
                if score >= threshold:
                    # Step 5: Calculate donor live distance from hospital
                    live_dist = calculate_distance(donor.get("current_latitude"), donor.get("current_longitude"), HOSPITAL_LAT, HOSPITAL_LON)
                    if live_dist is None:
                        # Fallback to stored approximate distance from db column
                        live_dist = donor.get("distance")
                    if live_dist is None:
                        live_dist = 999.0
                        
                    donor_info = {
                        "name": donor["name"],
                        "blood_group": donor["blood_group"],
                        "final_score": score,
                        "match_type": match_type,
                        "why_selected": generate_explanation(donor, match_type),
                        "distance": donor.get("distance"),
                        "live_distance_km": round(float(live_dist), 1),
                        "last_donation_date": donor.get("last_donation_date"),
                        "age": donor.get("age"),
                        "phone": donor.get("phone"),
                        "accepted_request": donor.get("accepted_request")
                    }
                    compatible_donors.append(donor_info)
                    
        # Sort donors using multi-key priority (Step 7: match type, score, ascending live distance)
        def get_donor_sort_key(d):
            match_priority = 0 if d["match_type"] == "exact" else 1
            score_priority = -d["final_score"]
            dist_val = d.get("live_distance_km") if d.get("live_distance_km") is not None else 999.0
            
            last_don_date = d.get("last_donation_date")
            if last_don_date:
                try:
                    parsed_date = datetime.datetime.strptime(last_don_date, "%Y-%m-%d").date()
                    days_since = (datetime.date.today() - parsed_date).days
                    days_since = max(0, days_since)
                except Exception:
                    days_since = 30
            else:
                days_since = 30
            gap_priority = -days_since
            
            age_val = d.get("age") if d.get("age") is not None else 35
            
            return (match_priority, score_priority, dist_val, gap_priority, age_val)
            
        compatible_donors.sort(key=get_donor_sort_key)
        
        # Format for standardized return (Step 10)
        top_donors_list = []
        for d in compatible_donors:
            if d.get("accepted_request") == 1:
                top_donors_list.append({
                    "name": d["name"],
                    "accepted_request": True,
                    "live_distance_km": d.get("live_distance_km"),
                    "response_status": "accepted"
                })
            elif d.get("accepted_request") == -1:
                top_donors_list.append({
                    "name": d["name"],
                    "accepted_request": False,
                    "live_distance_km": d.get("live_distance_km"),
                    "response_status": "declined"
                })
            else:
                top_donors_list.append({
                    "name": d["name"],
                    "accepted_request": False,
                    "live_distance_km": d.get("live_distance_km"),
                    "response_status": "pending"
                })
            
        return jsonify({"top_donors": top_donors_list})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/donor/requests", methods=["GET"])
def donor_requests():
    """
    Donor Emergency Console (Step 3 & Step 4).
    Provides a separate interface where compatible donors can view broadcasts and accept/decline alerts.
    """
    try:
        conn = get_db_connection()
        # Fetch the most recent active emergency broadcast
        cursor = conn.execute("SELECT * FROM emergency_requests WHERE status = 'active' ORDER BY id DESC LIMIT 1")
        active_req = cursor.fetchone()
        
        # Fetch all registered donors
        cursor_donors = conn.execute("SELECT id, name, blood_group, accepted_request, current_latitude, current_longitude, distance FROM donors ORDER BY name ASC")
        all_donors = [dict(row) for row in cursor_donors.fetchall()]
        conn.close()
        
        active_req_dict = dict(active_req) if active_req else None
        
        # Filter donors compatible with this active blood request group
        compatible_donors = []
        if active_req_dict:
            req_bg = active_req_dict["blood_group"]
            for d in all_donors:
                match_type = classify_compatibility(req_bg, d["blood_group"])
                if match_type in ["exact", "compatible"]:
                    d["match_type"] = match_type
                    compatible_donors.append(d)
        
        # Resolve the logged-in donor context dynamically for realistic session simulation
        logged_in_donor = None
        if compatible_donors:
            # Pick the first compatible donor as the logged-in session profile
            logged_in_donor = compatible_donors[0]
        elif all_donors:
            # Fallback to the first registered donor if no active alert is broadcasted
            logged_in_donor = all_donors[0]
        
        # Always reset logged-in donor to pending state on page load.
        # This ensures the YES/NO buttons are ALWAYS shown first.
        # State is only updated after donor explicitly clicks YES or NO (via AJAX).
        # This prevents stale "accepted" state from a previous broadcast showing on load.
        if logged_in_donor:
            try:
                reset_conn = get_db_connection()
                reset_conn.execute(
                    "UPDATE donors SET accepted_request = 0 WHERE id = ?",
                    (logged_in_donor["id"],)
                )
                reset_conn.commit()
                reset_conn.close()
                logged_in_donor["accepted_request"] = 0
            except Exception as reset_err:
                print(f"Warning: Could not reset donor pending state: {str(reset_err)}")
            
        return render_template("donor_requests.html", active_req=active_req_dict, logged_in_donor=logged_in_donor)
    except Exception as e:
        flash(f"Error accessing donor requests: {str(e)}", "danger")
        return redirect(url_for("home"))


@app.route("/api/accept_request", methods=["POST"])
def api_accept_request():
    """
    Emergency Acceptance endpoint (Step 4 & Step 5).
    - Consent-driven temporary live location sharing.
    - Location sharing is temporary and activated only after donor consent.
    - No permanent, passive, or continuous GPS tracking is performed.
    """
    try:
        data = request.get_json() or {}
        donor_id = data.get("donor_id")
        lat = data.get("latitude")
        lon = data.get("longitude")
        
        if not donor_id:
            return jsonify({"status": "error", "message": "Donor ID is required"}), 400
            
        try:
            lat_val = float(lat) if lat is not None else None
            lon_val = float(lon) if lon is not None else None
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid latitude/longitude numeric values"}), 400
            
        conn = get_db_connection()
        conn.execute('''
            UPDATE donors 
            SET accepted_request = 1, current_latitude = ?, current_longitude = ?
            WHERE id = ?
        ''', (lat_val, lon_val, donor_id))
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": "Donor accepted emergency request and shared temporary coordinates.",
            "donor_id": int(donor_id),
            "latitude": lat_val,
            "longitude": lon_val
        })
    except Exception as e:
        print(f"Error in api_accept_request: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/decline_request", methods=["POST"])
def api_decline_request():
    """
    Decline Emergency Request endpoint (Step 3 & Step 4).
    - Sets donor's status to declined (-1) to filter them from the current search results.
    """
    try:
        data = request.get_json() or {}
        donor_id = data.get("donor_id")
        
        if not donor_id:
            return jsonify({"status": "error", "message": "Donor ID is required"}), 400
            
        conn = get_db_connection()
        conn.execute('''
            UPDATE donors 
            SET accepted_request = -1, current_latitude = NULL, current_longitude = NULL
            WHERE id = ?
        ''', (donor_id,))
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": "Donor declined the emergency request.",
            "donor_id": int(donor_id)
        })
    except Exception as e:
        print(f"Error in api_decline_request: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/update_live_location", methods=["POST"])
def api_update_live_location():
    """
    Simulated Live Location Update backward-compatibility alias.
    """
    return api_accept_request()

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Required fields validation
        name = request.form.get("name", "").strip()
        blood_group = request.form.get("blood_group", "").strip()
        phone = request.form.get("phone", "").strip()
        location = request.form.get("location", "").strip()

        if not name or not blood_group or not phone or not location:
            flash("Failed to register. All asterisked (*) fields (Name, Blood Group, Phone, and Location) are strictly required.", "danger")
            return render_template("register.html", form_data=request.form)

        # Optional fields parsing
        response_rate_raw = request.form.get("response_rate", "").strip()
        consistency_raw = request.form.get("consistency", "").strip()
        last_active_days_raw = request.form.get("last_active_days", "").strip()
        donation_frequency_raw = request.form.get("donation_frequency", "").strip()
        last_donation_date_raw = request.form.get("last_donation_date", "").strip()
        distance_raw = request.form.get("distance", "").strip()
        age_raw = request.form.get("age", "").strip()
        weight_raw = request.form.get("weight", "").strip()
        current_latitude_raw = request.form.get("current_latitude", "").strip()
        current_longitude_raw = request.form.get("current_longitude", "").strip()
        
        # Checkbox availability (if checked, value will be present, otherwise not)
        availability = 1 if request.form.get("availability") else 0
        
        # Parse 3-state health status (Healthy = 1, Not Healthy = 0, Unknown/Missing = 2)
        health_status_raw = request.form.get("health_status", "2").strip()
        try:
            health_status = int(health_status_raw)
            if health_status not in [0, 1, 2]:
                health_status = 2
        except ValueError:
            health_status = 2

        # Safely convert types
        response_rate = None
        if response_rate_raw:
            try:
                response_rate = float(response_rate_raw)
            except ValueError:
                flash("Error: Response Rate must be a valid number.", "danger")
                return render_template("register.html", form_data=request.form)

        consistency = None
        if consistency_raw:
            try:
                consistency = float(consistency_raw)
            except ValueError:
                flash("Error: Consistency must be a valid number.", "danger")
                return render_template("register.html", form_data=request.form)

        last_active_days = None
        if last_active_days_raw:
            try:
                last_active_days = int(last_active_days_raw)
            except ValueError:
                flash("Error: Last Active Days must be a valid integer.", "danger")
                return render_template("register.html", form_data=request.form)

        donation_frequency = None
        if donation_frequency_raw:
            try:
                donation_frequency = int(donation_frequency_raw)
            except ValueError:
                flash("Error: Donation Frequency must be a valid integer.", "danger")
                return render_template("register.html", form_data=request.form)

        # Handle Last Donation Date validation explicitly (format and non-future verification)
        last_donation_date = None
        if last_donation_date_raw:
            try:
                parsed_date = datetime.datetime.strptime(last_donation_date_raw, "%Y-%m-%d").date()
                if parsed_date > datetime.date.today():
                    flash("Failed to register: Last Donation Date cannot be in the future.", "danger")
                    return render_template("register.html", form_data=request.form)
                last_donation_date = last_donation_date_raw
            except ValueError:
                flash("Failed to register: Last Donation Date must be in YYYY-MM-DD format.", "danger")
                return render_template("register.html", form_data=request.form)
        else:
            # Handle explicitly if missing. Saved as None in DB,
            # but treated as a neutral 30-day state in scoring calculations.
            last_donation_date = None

        # Validate Age (> 18)
        age = None
        if age_raw:
            try:
                age = int(age_raw)
                if age <= 18:
                    flash("Failed to register: Donor must be older than 18 years of age.", "danger")
                    return render_template("register.html", form_data=request.form)
            except ValueError:
                flash("Error: Age must be a valid integer.", "danger")
                return render_template("register.html", form_data=request.form)

        # Validate Weight (> 50)
        weight = None
        if weight_raw:
            try:
                weight = float(weight_raw)
                if weight <= 50.0:
                    flash("Failed to register: Donor weight must be greater than 50 kg.", "danger")
                    return render_template("register.html", form_data=request.form)
            except ValueError:
                flash("Error: Weight must be a valid number.", "danger")
                return render_template("register.html", form_data=request.form)

        # In real-world systems, distance should be dynamically calculated based on user location.
        # Here we store approximate pre-calculated or user-entered coordinates/distance.
        distance = None
        if distance_raw:
            try:
                distance = float(distance_raw)
                if distance < 0.0:
                    flash("Failed to register: Distance cannot be a negative value.", "danger")
                    return render_template("register.html", form_data=request.form)
            except ValueError:
                flash("Error: Distance must be a valid number.", "danger")
                return render_template("register.html", form_data=request.form)

        current_latitude = None
        if current_latitude_raw:
            try:
                current_latitude = float(current_latitude_raw)
            except ValueError:
                flash("Error: Current Latitude must be a valid number.", "danger")
                return render_template("register.html", form_data=request.form)

        current_longitude = None
        if current_longitude_raw:
            try:
                current_longitude = float(current_longitude_raw)
            except ValueError:
                flash("Error: Current Longitude must be a valid number.", "danger")
                return render_template("register.html", form_data=request.form)

        # Database Insertion
        try:
            conn = get_db_connection()
            conn.execute('''
                INSERT INTO donors (
                    name, blood_group, phone, location, 
                    response_rate, consistency, last_active_days, 
                    availability, donation_frequency,
                    last_donation_date, distance, health_status, age, weight,
                    current_latitude, current_longitude
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name, blood_group, phone, location, 
                response_rate, consistency, last_active_days, 
                availability, donation_frequency,
                last_donation_date, distance, health_status, age, weight,
                current_latitude, current_longitude
            ))
            conn.commit()
            conn.close()
            flash(f"Congratulations! {name} has been successfully registered as a donor.", "success")
            return redirect(url_for("home"))
        except Exception as e:
            flash(f"An unexpected database error occurred: {str(e)}", "danger")
            return render_template("register.html", form_data=request.form)

    return render_template("register.html", form_data={})

@app.route("/dashboard")
def dashboard():
    try:
        predicted_demand = get_future_prediction()

        # Fetch total donors count from SQLite database
        conn = get_db_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM donors")
        total_donors = cursor.fetchone()[0]
        conn.close()
    except Exception as e:
        print(f"Error loading dashboard metrics: {str(e)}")
        predicted_demand = 3  # Safe default
        total_donors = 0

    # Blood Demand Forecast Context
    bg_list = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    bg_predictions = {}
    total_next_day_demand = 0
    total_7_day_demand_sum = 0
    total_30_day_demand_sum = 0
    
    for bg in bg_list:
        p1 = predict_future_demand(bg, 1)
        p7 = predict_future_demand(bg, 7)
        p30 = predict_future_demand(bg, 30)
        
        bg_predictions[bg] = {
            "next_day": p1,
            "day_7": p7,
            "day_30": p30
        }
        total_next_day_demand += p1
        total_7_day_demand_sum += p7
        total_30_day_demand_sum += p30
        
    # Calculate average daily total demand over 7 and 30 days
    total_7_day_demand = int(round(sum(sum(predict_future_demand(bg, d) for bg in bg_list) for d in range(1, 8)) / 7.0))
    total_30_day_demand = int(round(sum(sum(predict_future_demand(bg, d) for bg in bg_list) for d in range(1, 31)) / 30.0))
    
    # Generate 7-day forecasted trend chart coordinates
    future_chart_labels = []
    future_chart_values = []
    today = datetime.date.today()
    for d in range(1, 8):
        future_date = today + datetime.timedelta(days=d)
        future_chart_labels.append(future_date.strftime('%Y-%m-%d'))
        
        day_total = sum(predict_future_demand(bg, d) for bg in bg_list)
        future_chart_values.append(day_total)

    return render_template(
        "dashboard.html",
        predicted_demand=predicted_demand,
        total_donors=total_donors,
        chart_labels=CHART_LABELS,
        chart_values=CHART_VALUES,
        total_next_day_demand=total_next_day_demand,
        total_7_day_demand=total_7_day_demand,
        total_30_day_demand=total_30_day_demand,
        bg_predictions=bg_predictions,
        future_chart_labels=future_chart_labels,
        future_chart_values=future_chart_values,
        is_synthetic_data=IS_SYNTHETIC_DATA
    )

@app.route("/api/predict_demand", methods=["POST"])
def api_predict_demand():
    try:
        data = request.get_json() or {}
        blood_group = data.get("blood_group", "").strip().upper()
        days_ahead = data.get("days_ahead", 1)
        
        try:
            days_ahead = int(days_ahead)
        except (ValueError, TypeError):
            days_ahead = 1
            
        if not blood_group:
            return jsonify({"status": "error", "message": "Blood group is required"}), 400
            
        valid_bgs = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
        if blood_group not in valid_bgs:
            return jsonify({
                "status": "fallback",
                "predicted_demand": 3
            })
            
        if not DEMAND_MODELS or blood_group not in DEMAND_MODELS:
            return jsonify({
                "status": "fallback",
                "predicted_demand": 3
            })
            
        predicted_val = predict_future_demand(blood_group, days_ahead)
        
        return jsonify({
            "status": "success",
            "blood_group": blood_group,
            "predicted_demand": predicted_val
        })
    except Exception as e:
        print(f"Error in api_predict_demand, returning fallback: {str(e)}")
        return jsonify({
            "status": "fallback",
            "predicted_demand": 3
        })

if __name__ == "__main__":
    app.run(debug=True)

