## Live Demo
- Frontend:https://huggingface.co/spaces/terah254/crop-advisory-dashboard
- Backend API:https://cropadvisorydashboard-production.up.railway.app
- API Docs:https://cropadvisorydashboard-production.up.railway.app/docs


# 🌾 Crop Advisory Intelligence Dashboard

A machine learning-powered web application that combines real-time weather
forecasting with agroforestry computer vision to deliver actionable crop
advisory insights for farmers across East Africa.



## Problem Statement

Smallholder farmers in East Africa make daily decisions about planting,
irrigation, fertilizer application, and harvesting with very little access
to reliable, location-specific weather intelligence. A farmer in Bomet or
Nairobi County has no easy way to know whether the next seven days are safe
for top-dressing maize, or whether an incoming rainfall event will wash away
freshly applied inputs.

This project tackles that gap directly. By pulling hyperlocal forecast data
from the WeatherAI geo-intelligence platform and running it through a trained
crop stress classifier, the system produces a plain-language daily risk
assessment alongside concrete farming recommendations — the kind of
intelligence that was previously only available to large commercial
agribusinesses with dedicated agronomists.

The addition of drone/aerial image analysis through the WeatherAI Trees API
takes it one step further: a farmer or extension officer can upload a farm
photo and instantly get a tree count, canopy health breakdown, and
AI-generated observations about the state of the crop or agroforestry system.


## Architecture Diagram










## Data Flow (WeatherAI APIs Used + How)

The application integrates three WeatherAI API endpoints, each serving a
distinct role in the data pipeline:

### 1. `/v1/daily` — 7-Day Daily Forecast
This is the core data source. For any given farm location passed as latitude
and longitude, this endpoint returns a day-by-day breakdown of expected
weather conditions. The fields extracted and used downstream are:

- `temp_max` and `temp_min` — used to detect heat stress and cold stress
- `precipitation_sum` — total expected rainfall in millimetres per day
- `precipitation_probability` — percentage likelihood of rain, used as a
  proxy for irrigation and spraying decisions
- `wind_max` — maximum wind speed, flagged when above 40 km/h

The response is converted into a pandas DataFrame and passed directly into
the feature engineering pipeline before hitting the ML model.

### 2. `/v1/current` — Live Current Conditions
Called in parallel with the daily forecast to populate the top metrics strip
on the dashboard. Returns the present-moment temperature, humidity, wind
speed, and condition description for the farm location. Displayed as live
context so the farmer sees both what is happening right now and what is
coming over the next week.

### 3. `/v1/trees/analyze` — Agroforestry Computer Vision
Accepts a multipart image upload (JPEG, PNG, or WEBP) from a drone, aerial,
or satellite source. Internally uses OpenCV for tree crown detection and
Gemini AI for agronomic context generation. Returns:

- Total tree count and density per acre
- Canopy coverage percentage
- Health breakdown: healthy / needs care / needs replacement
- Observations and agronomic recommendations

This endpoint is called only when the user uploads a farm image, keeping API
quota usage efficient for users on the free plan.



## ML Model: Features, Logic, Output

### Feature Engineering
Raw forecast fields alone are not informative enough for crop risk decisions.
The following derived features are computed from the daily forecast data before
the model sees any input:

Feature  Source Fields Agronomic Meaning 

 `heat_stress`  `temp_max > 32°C`  Triggers above this reduce photosynthesis in most crops 
`cold_stress`  `temp_min < 10°C`  Risk of chilling injury for tropical crops like maize 
 `drought_flag`  `precipitation_sum < 1mm`  Effectively a dry day — triggers irrigation need 
 `flood_risk_flag`  `precipitation_sum > 30mm`  Waterlogging and runoff risk 
 `high_rain_prob`  `precipitation_probability > 70%`  Advises against spraying pesticides or fertilizers 
 `high_wind`  `wind_max > 40 km/h`  Physical crop damage and spray drift risk 
 `crop_stress_score`  Weighted composite  0–100 index summarising overall daily stress level 
 `planting_window`  `stress_score < 20`| Binary flag marking ideal planting or top-dressing days 

### Model
Algorithm: Gradient Boosting Classifier (`sklearn.ensemble.GradientBoostingClassifier`)  
Parameters: `n_estimators=150`, `max_depth=4`, `random_state=42`  
Training data: 2,000 synthetically generated weather samples labeled
using agronomically validated thresholds  
Features used: `temp_max`, `temp_min`, `precipitation_sum`,
`precipitation_probability`, `wind_max`

### Output Classes
 Class  Label  Meaning 
 0 |Low Risk  Conditions are favorable for most farm activities |
 1  Moderate Risk  Some caution advised — monitor conditions closely |
 2  High Risk  Adverse conditions — delay sensitive farm operations |

The model runs independently on each of the 7 forecast days, producing a
per-day risk label and confidence score. The results feed into the advisory
generator which produces plain-language farming tips based on the pattern
of risk flags across the week.


## How to Run Locally

### Prerequisites
- Python 3.10 or higher
- Git
- A WeatherAI API key from [weather-ai.co](https://weather-ai.co)

### 1. Clone the repository
```bash
git clone https://github.com/terah254/crop-advisory-dashboard.git
cd crop-advisory-dashboard
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the project root:
WEATHER_AI_KEY=wai_your_key_here

### 5. Start the backend
```bash
uvicorn main:app --reload --port 8000
```
Test it is running:
http://127.0.0.1:8000/health

### 6. Start the frontend
Open a second terminal:
```bash
cd frontend
streamlit run app.py
```
Opens at:
http://localhost:8501

### 7. Run the Jupyter notebook (optional)
```bash
cd notebook
jupyter notebook
```
Open `exploration.ipynb` and run all cells to see the full data exploration,
feature engineering, model training, and WeatherAI API integration walkthrough.



## Live Demo

 Link 
 Frontend Dashboard | [huggingface.co/spaces/terah254/Crop_advisory_dashboard](https://huggingface.co/spaces/terah254/Crop_advisory_dashboard)
 Backend API (https://cropadvisorydashboard-production.up.railway.app)
 API Docs (https://cropadvisorydashboard-production.up.railway.app/docs)
 GitHub Repo (https://github.com/terah254/crop-advisory-dashboard) 



## Tech Stack

Layer - Technology 
Frontend - Streamlit, Plotly 
Backend - FastAPI, Uvicorn 
ML Model - scikit-learn, GradientBoostingClassifier 
Data - WeatherAI REST APIs 
Hosting (Frontend) - Hugging Face Spaces 
Hosting (Backend) - Railway 
Language - Python 3.11+ 


## Future Work

- Integrate NASA POWER satellite data for historical rainfall validation
- Replace synthetic training data with labeled agronomic field datasets
- Add SMS alert delivery using the WeatherAI Scale plan `/v1/sms/alert`
  endpoint to push risk warnings directly to farmers' phones
- Extend forecast window to 14 days using the `/v1/forecast14` Pro endpoint
- Build a farmer registration system modeled on the WeatherAI Bomet
  Agricultural Alert System for community-scale advisory delivery
