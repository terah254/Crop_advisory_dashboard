import streamlit as st
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime


st.set_page_config(
    page_title="Crop Advisory Dashboard",
    page_icon="🌾",
    layout="wide"
)


st.markdown("""
    <style>
        .main { background-color: #f5f7f2; }
        .block-container { padding-top: 2rem; }
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"   # change to Railway URL when deployed

st.title("🌾 Crop Advisory Intelligence Dashboard")
st.caption("Powered by WeatherAI Geo-Intelligence APIs + ML Risk Modeling")
st.markdown("---")

with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2454/2454285.png",
        width=80
    )
    st.header("📍 Farm Details")

    st.subheader("Location")
    lat = st.number_input("Latitude",  value=-1.2921, format="%.4f",
                          help="e.g. -1.2921 for Nairobi")
    lon = st.number_input("Longitude", value=36.8219, format="%.4f",
                          help="e.g. 36.8219 for Nairobi")

    st.subheader("Farm Info")
    county     = st.text_input("County / Region", value="Nairobi")
    land_acres = st.number_input("Farm Size (acres)", value=2.5, min_value=0.1)
    crop_type  = st.selectbox("Crop Type", [
        "Maize", "Tea", "Coffee", "Wheat",
        "Rice", "Beans", "Tomatoes", "Other"
    ])

    st.subheader("Canopy Analysis (Optional)")
    image_file = st.file_uploader(
        "Upload Farm Image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Aerial or drone image of your farm"
    )
    if image_file:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)

    st.markdown("---")
    run = st.button("🔍 Analyze Farm", type="primary", use_container_width=True)

def risk_color(label):
    return {
        "Low Risk":      "green",
        "Moderate Risk": "orange",
        "High Risk":     "red"
    }.get(label, "gray")

def risk_emoji(label):
    return {
        "Low Risk":      "✅",
        "Moderate Risk": "⚡",
        "High Risk":     "🚨"
    }.get(label, "❓")

def format_date(date_str):
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").strftime("%a %d %b")
    except:
        return date_str

if not run:
    st.markdown("### 👈 Enter your farm details in the sidebar and click Analyze Farm")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🌦 **7-Day Forecast**\nGet daily temperature, rainfall and wind forecasts for your exact farm location.")
    with col2:
        st.info("🤖 **ML Risk Scoring**\nMachine learning model scores each day's crop stress risk as Low, Moderate or High.")
    with col3:
        st.info("🌳 **Canopy Analysis**\nUpload a drone image to count trees and assess canopy health automatically.")

    st.markdown("---")
    st.markdown("#### How it works")
    steps = {
        "1️⃣ Enter location": "Input your farm's GPS coordinates and details in the sidebar.",
        "2️⃣ Fetch forecast":  "The app calls WeatherAI APIs to get your 7-day forecast.",
        "3️⃣ Run ML model":    "Features are engineered and passed to a GradientBoosting classifier.",
        "4️⃣ Get advisory":    "Actionable recommendations are generated based on forecast + canopy data."
    }
    for step, desc in steps.items():
        st.markdown(f"**{step}** — {desc}")

# MAin analysiis
if run:
    with st.spinner("🌍 Fetching forecast and running risk model..."):
        try:
            # Build request
            form_data = {
                "lat":        str(lat),
                "lon":        str(lon),
                "county":     county,
                "land_acres": str(land_acres)
            }
            files = {}
            if image_file:
                files["image"] = (image_file.name, image_file.read(), "image/jpeg")

            with httpx.Client(timeout=60) as client:
                response = client.post(
                    f"{API_URL}/analyze",
                    data=form_data,
                    files=files if files else None
                )
                response.raise_for_status()

            result = response.json()
            st.success("✅ Analysis complete!")

        except httpx.ConnectError:
            st.error("❌ Cannot connect to backend. Make sure uvicorn is running on port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()

    # Section 1
    st.markdown("---")
    st.subheader("📍 Current Conditions")

    current = result.get("current", {})
    # handle nested structure
    if "current" in current:
        current = current["current"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌡 Temperature",  f"{current.get('temp_c', current.get('temp', '—'))} °C")
    c2.metric("💧 Humidity",     f"{current.get('humidity', '—')} %")
    c3.metric("💨 Wind",         f"{current.get('wind_kph', current.get('wind_kmh', '—'))} km/h")
    c4.metric("🌂 Condition",    current.get("condition", current.get("description", "—")))
    c5.metric("📍 Location",     f"{county}")

    # Section 2: Risk summary banner 
    st.markdown("---")
    high_days = result["risk_summary"]["high_risk_days"]
    risk_labels = result["risk_summary"]["risk_labels"]

    if high_days >= 4:
        st.error(f"🚨 HIGH ALERT: Severe crop stress risk on {high_days}/7 days. Immediate action recommended.")
    elif high_days >= 2:
        st.warning(f"⚡ MODERATE ALERT: {high_days} high-risk days forecast this week. Monitor closely.")
    elif high_days == 1:
        st.warning(f"⚡ 1 high-risk day forecast. Stay alert.")
    else:
        st.success("✅ LOW RISK: Favorable conditions across the week. Good time for farm activity.")

    # Section 3: 7-day forecast chart 
    st.markdown("---")
    st.subheader("📊 7-Day Forecast + ML Risk Output")

    df = pd.DataFrame(result["forecast"])
    df["date_label"] = df["date"].apply(format_date)

    # Color each bar by risk label
    color_map = {
        "Low Risk":      "#2ecc71",
        "Moderate Risk": "#f39c12",
        "High Risk":     "#e74c3c"
    }
    df["bar_color"] = df["risk_label"].map(color_map)

    tab1, tab2, tab3 = st.tabs(["🌡 Temperature", "🌧 Rainfall & Risk", "📈 Stress Score"])

    with tab1:
        fig1 = go.Figure()
        fig1.add_scatter(
            x=df["date_label"], y=df["temp_max"],
            name="Max Temp (°C)", line=dict(color="tomato", width=2),
            mode="lines+markers"
        )
        fig1.add_scatter(
            x=df["date_label"], y=df["temp_min"],
            name="Min Temp (°C)", line=dict(color="skyblue", width=2, dash="dot"),
            mode="lines+markers"
        )
        fig1.update_layout(
            title="Daily Temperature Range",
            yaxis_title="Temperature (°C)",
            height=350, plot_bgcolor="white"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_bar(
            x=df["date_label"], y=df["precipitation_sum"],
            name="Rainfall (mm)",
            marker_color=df["bar_color"],
            text=df["risk_label"], textposition="outside"
        )
        fig2.add_scatter(
            x=df["date_label"], y=df["precipitation_probability"],
            name="Rain Probability (%)",
            line=dict(color="navy", width=2, dash="dot"),
            yaxis="y2", mode="lines+markers"
        )
        fig2.update_layout(
            title="Rainfall + Risk Label per Day",
            yaxis_title="Rainfall (mm)",
            yaxis2=dict(title="Probability (%)", overlaying="y", side="right"),
            height=350, plot_bgcolor="white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = go.Figure()
        fig3.add_scatter(
            x=df["date_label"], y=df["crop_stress_score"],
            name="Crop Stress Score",
            line=dict(color="purple", width=3),
            fill="tozeroy", fillcolor="rgba(128,0,128,0.1)",
            mode="lines+markers"
        )
        fig3.add_hline(y=20, line_dash="dash", line_color="green",
                       annotation_text="Low Risk Threshold")
        fig3.add_hline(y=50, line_dash="dash", line_color="orange",
                       annotation_text="Moderate Risk Threshold")
        fig3.update_layout(
            title="Crop Stress Score (0–100)",
            yaxis_title="Stress Score",
            yaxis_range=[0, 100],
            height=350, plot_bgcolor="white"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Section 4: Daily risk table 
    st.markdown("---")
    st.subheader("📋 Daily Risk Breakdown")

    display_df = df[[
        "date_label", "temp_max", "temp_min",
        "precipitation_sum", "precipitation_probability",
        "wind_max", "crop_stress_score", "risk_label"
    ]].rename(columns={
        "date_label":               "Date",
        "temp_max":                 "Max Temp (°C)",
        "temp_min":                 "Min Temp (°C)",
        "precipitation_sum":        "Rainfall (mm)",
        "precipitation_probability":"Rain Prob (%)",
        "wind_max":                 "Wind (km/h)",
        "crop_stress_score":        "Stress Score",
        "risk_label":               "Risk Level"
    })

    def highlight_risk(val):
        colors = {
            "Low Risk":      "background-color: #d5f5e3; color: green",
            "Moderate Risk": "background-color: #fef9e7; color: orange",
            "High Risk":     "background-color: #fadbd8; color: red"
        }
        return colors.get(val, "")

    st.dataframe(
        display_df.style.map(highlight_risk, subset=["Risk Level"]),
        use_container_width=True,
        hide_index=True
    )

    #  Section 5: Advisory 
    st.markdown("---")
    st.subheader("🧠 ML Advisory Recommendations")
    st.caption(f"Based on 7-day forecast for {county} — Crop: {crop_type}")

    for tip in result.get("advisory", []):
        st.info(f"💡 {tip}")

    #  Section 6: Tree & canopy analysis 
    if result.get("tree_analysis"):
        st.markdown("---")
        st.subheader("🌳 Canopy & Tree Health Analysis")

        ta = result["tree_analysis"]

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("🌳 Total Trees",     ta.get("total_tree_count", "—"))
        t2.metric("🌿 Canopy Coverage", f"{ta.get('canopy_coverage_pct', '—')} %")
        t3.metric("✅ Healthy",          ta.get("tree_health", {}).get("healthy", "—"))
        t4.metric("⚠️ Need Care",        ta.get("tree_health", {}).get("needs_care", "—"))

        col_left, col_right = st.columns(2)

        with col_left:
            health = ta.get("tree_health", {})
            if sum(health.values()) > 0:
                fig4 = px.pie(
                    values=list(health.values()),
                    names=list(health.keys()),
                    color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
                    title="Tree Health Breakdown"
                )
                st.plotly_chart(fig4, use_container_width=True)

        with col_right:
            if ta.get("overlay_image_url"):
                st.image(ta["overlay_image_url"], caption="Annotated Farm Canopy")

        if ta.get("observations"):
            st.markdown("**🔍 Observations:**")
            for obs in ta["observations"]:
                st.write(f"• {obs}")

        if ta.get("recommendations"):
            st.markdown("**✅ Agronomic Recommendations:**")
            for rec in ta["recommendations"]:
                st.success(f"• {rec}")

    #  Section 7: Raw JSON (for developers) 
    st.markdown("---")
    with st.expander("🔧 Raw API Response (Developer View)"):
        st.json(result)

    # Footer 
    st.markdown("---")
    st.caption("Built with WeatherAI APIs · FastAPI · Streamlit · scikit-learn")