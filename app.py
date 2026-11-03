import os
import base64
import streamlit.components.v1 as components
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Air Pollution Prediction System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "air_quality_model.pkl"
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "AirPollutionDB"
PREDICTIONS_COLLECTION = "air_quality"
HISTORICAL_COLLECTION = "historical_air_quality"

NAV_ITEMS = [
    "🏠 Home",
    "📊 City Analytics",
    "🔮 AQI Prediction",
    "🗂️ Prediction History",
    "ℹ️ About Project",
]

# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
    [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"] {
    font-size: 0 !important;
    /* Hide the unwanted Material icon text near Deploy */
[data-testid="stAppDeployButton"] span[data-testid="stIconMaterial"] {
    font-size: 0 !important;
}
}
    /* Sidebar navigation typography */
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-family: "Trebuchet MS", Arial, sans-serif !important;
        font-size: 17px !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px !important;
    }

    .stApp {
        background: linear-gradient(135deg, #08111f 0%, #0b1728 55%, #0d2030 100%);
        color: #e2e8f0;
    }
    .block-container {
    padding-top: 3.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 750 !important;
        letter-spacing: -0.4px;
    }
    h1 { font-size: 2.45rem !important; }
    p, label, [data-testid="stCaptionContainer"] { color: #cbd5e1; }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #132238, #102b35);
        border: 1px solid #244254;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #10b981;
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #34d399 !important;
        font-weight: 800 !important;
    }
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(100deg, #059669, #0891b2);
        color: #ffffff !important;
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        font-weight: 700;
        transition: all 0.2s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: linear-gradient(100deg, #047857, #0e7490);
        border-color: #34d399;
        box-shadow: 0 0 18px rgba(16, 185, 129, 0.18);
    }
    .stNumberInput input, [data-baseweb="select"] > div {
        background-color: #101e30 !important;
        color: #f1f5f9 !important;
        border-radius: 9px !important;
        border-color: #334155 !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"] {
        background-color: #132238 !important;
    }
    [data-testid="stForm"] {
        background: rgba(17, 34, 54, 0.75);
        border: 1px solid #294154;
        border-radius: 16px;
        padding: 22px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #294154;
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: #294154 !important; }
    [data-testid="stSidebar"] { background-color: #0a1423; }
    /* Bigger sidebar navigation */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px 4px !important;
    min-height: 42px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 1.08rem !important;
    font-weight: 650 !important;
    line-height: 1.35 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 8px;
}
    a { color: #67e8f9 !important; }


.hero {
    box-sizing: border-box;
    width: 100%;
    min-height: 340px;
    padding: 38px 36px;
    border-radius: 22px;
    overflow: visible;
    position: relative;
    box-sizing: border-box;
    width: 100%;
    max-width: 100%;
    overflow-wrap: anywhere;

    background:
        linear-gradient(
            90deg,
            rgba(7, 17, 31, 0.88),
            rgba(7, 17, 31, 0.58)
        ),
        url("hero_clouds.png");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;

    border: 1px solid #28505b;
    box-shadow: 0 16px 45px rgba(0, 0, 0, 0.22);
    margin-bottom: 22px;
}

    .hero-kicker {
        color: #67e8f9;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero h1 {
    margin: 0 0 16px 0;
    font-size: clamp(2rem, 3.2vw, 2.8rem) !important;
    line-height: 1.2;
    overflow-wrap: anywhere;
}
    .hero p {
    color: #cbd5e1;
    font-size: 1.05rem;
    max-width: 850px;
    line-height: 1.7;
    overflow-wrap: anywhere;
}
@media (max-width: 900px) {
    .hero {
        min-height: 0;
        padding: 26px 22px;
    }

    .hero h1 {
        font-size: 2rem !important;
    }
}
    .section-note {
        color: #94a3b8;
        font-size: 0.92rem;
    }
    .pill {
        display: inline-block;
        border: 1px solid #2c5961;
        color: #a7f3d0;
        background: rgba(16,185,129,0.08);
        border-radius: 999px;
        padding: 5px 10px;
        margin: 3px 5px 3px 0;
        font-size: 0.8rem;
    }
    
/* Global font across the entire dashboard */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html,
body,
[class*="st-"],
[data-testid="stApp"],
[data-testid="stSidebar"],
[data-testid="stMarkdownContainer"],
input,
textarea,
button,
select {
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar title: Air Quality */
[data-testid="stSidebar"] h2 {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.6px !important;
    color: #f1f5f9 !important;
}

/* Sidebar subtitle */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.2px !important;
}

/* Headings on every page */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 750 !important;
    letter-spacing: -0.5px !important;
}

/* Main page text */
p, label, li, [data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar navigation labels */
[data-testid="stSidebar"] .stButton button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA / DATABASE HELPERS
# ============================================================
@st.cache_resource
def get_mongo():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    return client


@st.cache_data(ttl=60)
def load_historical_data():
    client = get_mongo()
    collection = client[DB_NAME][HISTORICAL_COLLECTION]
    records = list(collection.find({}, {"_id": 0}))
    if not records:
        return pd.DataFrame()

    data = pd.DataFrame(records)
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    numeric_columns = [
        "AQI", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
        "SO2", "O3", "Benzene", "Toluene", "Xylene",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "City" in data.columns:
        data["City"] = data["City"].astype("string").str.strip()
    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_prediction_collection():
    client = get_mongo()
    return client[DB_NAME][PREDICTIONS_COLLECTION]


def style_plot(fig, height=420):
    fig.update_layout(
        paper_bgcolor="#0b1728",
        plot_bgcolor="#0b1728",
        font=dict(color="#cbd5e1"),
        height=height,
        margin=dict(l=20, r=20, t=65, b=25),
        hoverlabel=dict(
            bgcolor="#132238",
            bordercolor="#34d399",
            font_color="#f1f5f9",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        linecolor="#334155",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        linecolor="#334155",
        zeroline=False,
    )
    return fig


def aqi_category(value):
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


def aqi_color(value):
    if value <= 50:
        return "#34f5a1"
    if value <= 100:
        return "#a3e635"
    if value <= 200:
        return "#facc15"
    if value <= 300:
        return "#fb923c"
    if value <= 400:
        return "#f87171"
    return "#e879f9"


def prepare_city_summary(data):
    if data.empty or "City" not in data.columns:
        return pd.DataFrame()

    aggregations = {}
    if "AQI" in data.columns:
        aggregations["AQI"] = ["mean", "max", "count"]
    for pollutant in ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]:
        if pollutant in data.columns:
            aggregations[pollutant] = "mean"

    if not aggregations:
        return pd.DataFrame()

    summary = data.groupby("City", dropna=True).agg(aggregations)
    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary = summary.reset_index()

    rename_map = {
        "AQI_mean": "Average AQI",
        "AQI_max": "Highest AQI",
        "AQI_count": "AQI Records",
        "PM2.5_mean": "Average PM2.5",
        "PM10_mean": "Average PM10",
        "NO2_mean": "Average NO2",
        "CO_mean": "Average CO",
        "SO2_mean": "Average SO2",
        "O3_mean": "Average O3",
    }
    return summary.rename(columns=rename_map)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================


with st.sidebar:
    st.markdown("## Air Quality")
    st.caption("INDIA MONITORING SYSTEM")
    st.divider()

    nav_items = [
        ("🏠 Home", "Home", "⌂"),
        ("📊 City Analytics", "City Analytics", "▥"),
        ("🔮 AQI Prediction", "AQI Prediction", "⌁"),
        ("🗂️ Prediction History", "Prediction History", "◷"),
        ("ℹ️ About Project", "About Project", "ⓘ"),
    ]

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "🏠 Home"

    for page_value, label, icon in nav_items:
        if st.button(
            f"{icon}    {label}",
            key=f"nav_{label}",
            use_container_width=True,
        ):
            st.session_state["active_page"] = page_value

    page = st.session_state["active_page"]

    st.divider()
    st.caption("Project dashboard • Local MongoDB")


# ============================================================
# CONNECT TO MONGODB AND LOAD DATA
# ============================================================
mongo_connected = False
mongo_error = None
try:
    mongo_client = get_mongo()
    db = mongo_client[DB_NAME]
    predictions_collection = db[PREDICTIONS_COLLECTION]
    historical_collection = db[HISTORICAL_COLLECTION]
    mongo_connected = True
except Exception as exc:
    mongo_error = str(exc)
    predictions_collection = None
    historical_collection = None

if not mongo_connected:
    st.warning(
        "MongoDB is not connected. Start MongoDB and refresh the page to load "
        "historical data or save prediction records."
    )
    with st.expander("Connection details"):
        st.code(mongo_error or "Unknown connection error")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: `{MODEL_PATH}`. Keep `air_quality_model.pkl` "
        "in the same folder as `app.py`."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the prediction model: {exc}")
    st.stop()

try:
    df = load_historical_data() if mongo_connected else pd.DataFrame()
except Exception as exc:
    st.error(f"Could not load historical data from MongoDB: {exc}")
    df = pd.DataFrame()

if not df.empty:
    if "AQI" in df.columns:
        df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")


# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":

    image_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "image.png")

    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    st.markdown(
        f"""
        <div class="hero" style="
            background-image:
                linear-gradient(
                    90deg,
                    rgba(7, 17, 31, 0.82),
                    rgba(7, 17, 31, 0.48)
                ),
                url('data:image/png;base64,{image_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        ">
            <div class="hero-kicker">India • Air quality intelligence</div>
            <h1>Understand the air.<br>Make better decisions.</h1>
            <p>
                Explore historical air-quality patterns across Indian cities
                and estimate an Air Quality Index (AQI) from selected pollutant
                concentrations using a machine-learning model.
            </p>
            <span class="pill">Historical insights</span>
            <span class="pill">City comparisons</span>
            <span class="pill">AQI estimation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Dashboard overview")
    if df.empty:
        st.info(
            "Historical data is not available yet. Check that MongoDB is running "
            "and the `historical_air_quality` collection contains your dataset."
        )
    else:
        total_records = len(df)
        city_count = df["City"].nunique() if "City" in df.columns else 0
        pollutants = [
            col for col in
            ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
             "O3", "Benzene", "Toluene", "Xylene"]
            if col in df.columns
        ]
        valid_aqi = df["AQI"].dropna(
        ) if "AQI" in df.columns else pd.Series(dtype=float)
        avg_aqi = valid_aqi.mean() if not valid_aqi.empty else float("nan")
        start_date = df["Date"].min() if "Date" in df.columns else pd.NaT
        end_date = df["Date"].max() if "Date" in df.columns else pd.NaT

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Historical records", f"{total_records:,}")
        c2.metric("Cities covered", f"{city_count:,}")
        c3.metric("Pollutants available", f"{len(pollutants):,}")
        c4.metric("Overall average AQI",
                  f"{avg_aqi:.1f}" if pd.notna(avg_aqi) else "N/A")

        if pd.notna(start_date) and pd.notna(end_date):
            st.caption(
                f"Historical coverage: {start_date:%d %b %Y} – {end_date:%d %b %Y}. "
                "These are dataset records, not live monitoring readings."
            )

        st.divider()
        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Most polluted cities by average AQI")
            summary = prepare_city_summary(df)
            if not summary.empty and "Average AQI" in summary.columns:
                ranking = (
                    summary.dropna(subset=["Average AQI"])
                    .sort_values("Average AQI", ascending=False)
                    .head(10)
                    .sort_values("Average AQI", ascending=True)
                )
                fig = px.bar(
                    ranking,
                    x="Average AQI",
                    y="City",
                    orientation="h",
                    color="Average AQI",
                    color_continuous_scale=["#34d399",
                                            "#22d3ee", "#f97316", "#f87171"],
                    hover_data=[
                        "Highest AQI"] if "Highest AQI" in ranking.columns else None,
                )
                fig.update_layout(coloraxis_showscale=False,
                                  title="Top 10 cities")
                st.plotly_chart(style_plot(fig, 430), use_container_width=True)
            else:
                st.info("AQI city summaries are unavailable in this dataset.")

        with right:
            st.subheader("Explore a city")
            city_names = sorted(df["City"].dropna().unique()
                                ) if "City" in df.columns else []
            if city_names:
                home_city = st.selectbox(
                    "Choose a city", city_names, key="home_city")
                city_rows = df[df["City"] == home_city].copy()
                city_aqi = city_rows["AQI"].dropna(
                ) if "AQI" in city_rows.columns else pd.Series(dtype=float)
                m1, m2 = st.columns(2)
                m1.metric(
                    "Average AQI", f"{city_aqi.mean():.1f}" if not city_aqi.empty else "N/A")
                m2.metric(
                    "Highest AQI", f"{city_aqi.max():.1f}" if not city_aqi.empty else "N/A")
                if "Date" in city_rows.columns and not city_aqi.empty:
                    trend = city_rows.dropna(
                        subset=["Date", "AQI"]).sort_values("Date")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend["Date"], y=trend["AQI"], mode="lines",
                        line=dict(color="#34f5a1", width=2.5),
                        fill="tozeroy", fillcolor="rgba(52,245,161,0.07)",
                        hovertemplate="%{x|%d %b %Y}<br>AQI: %{y:.1f}<extra></extra>",
                    ))
                    fig.update_layout(
                        title=f"{home_city} • AQI trend", showlegend=False)
                    st.plotly_chart(style_plot(fig, 300),
                                    use_container_width=True)
            else:
                st.info("No city names found in the historical collection.")

    st.divider()
    st.subheader("Quick navigation")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown("**📊 City Analytics**")
        st.caption("Compare AQI and pollutant patterns between cities.")
    with q2:
        st.markdown("**🔮 AQI Prediction**")
        st.caption("Estimate AQI from four pollutant inputs.")
    with q3:
        st.markdown("**🗂️ Prediction History**")
        st.caption("Review predictions saved to your local database.")


# ============================================================
# CITY ANALYTICS PAGE
# ============================================================
elif page == "📊 City Analytics":
    st.title("📊 City Analytics")
    st.write("Compare historical air-quality records across Indian cities.")
    st.caption(
        "All values below are calculated from the dataset stored in MongoDB; they are not live readings.")

    if df.empty or "City" not in df.columns:
        st.warning(
            "No historical city data is available. Check MongoDB and the historical collection.")
    else:
        summary = prepare_city_summary(df)
        if summary.empty:
            st.warning(
                "The historical collection does not contain usable AQI or pollutant columns.")
        else:
            metric_options = [
                col for col in
                ["Average AQI", "Highest AQI", "Average PM2.5", "Average PM10",
                 "Average NO2", "Average CO", "Average SO2", "Average O3"]
                if col in summary.columns
            ]
            controls1, controls2 = st.columns([1, 1])
            with controls1:
                metric_choice = st.selectbox(
                    "Rank cities by", metric_options, index=0 if metric_options else None
                )
            with controls2:
                direction = st.radio(
                    "Ranking order", [
                        "Most polluted / highest first", "Lowest first"],
                    horizontal=True,
                )

            if metric_options:
                ranking = summary.dropna(subset=[metric_choice]).copy()
                ranking = ranking.sort_values(
                    metric_choice,
                    ascending=(direction == "Lowest first"),
                )
                top_n = st.slider(
                    "Number of cities to display", min_value=5,
                    max_value=max(5, min(30, len(ranking))),
                    value=min(10, max(5, len(ranking))),
                )
                chart_data = ranking.head(top_n).copy()
                fig = px.bar(
                    chart_data,
                    x=metric_choice,
                    y="City",
                    orientation="h",
                    color=metric_choice,
                    color_continuous_scale=["#34d399",
                                            "#22d3ee", "#f97316", "#f87171"],
                    hover_data=[
                        col for col in
                        ["Average AQI", "Highest AQI",
                            "Average PM2.5", "Average PM10"]
                        if col in chart_data.columns and col != metric_choice
                    ],
                )
                fig.update_layout(
                    title=f"City ranking • {metric_choice}",
                    yaxis={"categoryorder": "total ascending" if direction !=
                           "Lowest first" else "total descending"},
                    coloraxis_showscale=False,
                )
                st.plotly_chart(style_plot(
                    fig, max(400, top_n * 32)), use_container_width=True)

            st.divider()
            st.subheader("Compare selected cities")
            available_cities = sorted(summary["City"].dropna().unique())
            default_cities = available_cities[: min(5, len(available_cities))]
            selected_cities = st.multiselect(
                "Choose two or more cities",
                options=available_cities,
                default=default_cities,
            )
            compare_metric = st.selectbox(
                "Comparison metric",
                [col for col in metric_options if col in summary.columns],
                key="compare_metric",
            )
            if selected_cities:
                compare_df = summary[summary["City"].isin(
                    selected_cities)].dropna(subset=[compare_metric])
                compare_df = compare_df.sort_values(
                    compare_metric, ascending=False)
                fig = px.bar(
                    compare_df, x="City", y=compare_metric, color="City",
                    color_discrete_sequence=[
                        "#34f5a1", "#22d3ee", "#a78bfa", "#fb923c", "#f87171", "#a3e635"],
                    text_auto=".1f",
                )
                fig.update_layout(
                    title=f"Selected-city comparison • {compare_metric}", showlegend=False)
                st.plotly_chart(style_plot(fig, 400), use_container_width=True)
            else:
                st.info("Select at least one city to compare.")

            st.divider()
            st.subheader("Pollutant profile")
            pollutant_columns = [
                col for col in ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
                if col in df.columns
            ]
            if pollutant_columns:
                pollutant = st.selectbox(
                    "Choose a pollutant", pollutant_columns)
                pollutant_summary = (
                    df.groupby("City", dropna=True)[pollutant]
                    .mean().dropna().sort_values(ascending=False).head(15)
                    .reset_index()
                )
                pollutant_summary.columns = ["City", f"Average {pollutant}"]
                fig = px.bar(
                    pollutant_summary, x=f"Average {pollutant}", y="City",
                    orientation="h", color=f"Average {pollutant}",
                    color_continuous_scale=["#34d399", "#22d3ee", "#f97316"],
                )
                fig.update_layout(
                    title=f"Cities by average {pollutant}",
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                )
                st.plotly_chart(style_plot(fig, 450), use_container_width=True)
            else:
                st.info("No supported pollutant concentration columns were found.")

            st.divider()
            st.subheader("City summary table")
            display_columns = [
                col for col in
                ["City", "Average AQI", "Highest AQI", "AQI Records", "Average PM2.5",
                 "Average PM10", "Average NO2", "Average CO", "Average SO2", "Average O3"]
                if col in summary.columns
            ]
            st.dataframe(
                summary[display_columns].sort_values(
                    "Average AQI", ascending=False, na_position="last"
                ).round(2),
                use_container_width=True,
                hide_index=True,
            )
            csv = summary[display_columns].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download city summary as CSV",
                data=csv,
                file_name="india_city_air_quality_summary.csv",
                mime="text/csv",
            )


# ============================================================
# AQI PREDICTION PAGE
# ============================================================
elif page == "🔮 AQI Prediction":
    st.title("🔮 AQI Prediction")
    st.write(
        "Enter pollutant concentrations to estimate AQI using the trained Random Forest model.")
    st.caption("Use the same units and input ranges used by your training dataset. This is an estimate, not an official reading.")

    with st.form("prediction_form"):
        st.subheader("Pollutant inputs")
        left, right = st.columns(2)
        with left:
            pm25 = st.number_input(
                "PM2.5 concentration", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)
            pm10 = st.number_input(
                "PM10 concentration", min_value=0.0, max_value=1000.0, value=80.0, step=1.0)
        with right:
            no2 = st.number_input(
                "NO2 concentration", min_value=0.0, max_value=500.0, value=30.0, step=1.0)
            co = st.number_input(
                "CO concentration", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
        submitted = st.form_submit_button(
            "✨ Predict AQI", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame(
            [[pm25, pm10, no2, co]],
            columns=["PM2.5", "PM10", "NO2", "CO"],
        )
        try:
            prediction = float(model.predict(input_data)[0])
            if not pd.notna(prediction):
                raise ValueError("The model returned an invalid number.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        category = aqi_category(prediction)
        color = aqi_color(prediction)
        advisory_by_category = {
            "Good": "Air quality is good. Enjoy outdoor activities as usual.",
            "Satisfactory": "Air quality is generally acceptable. Sensitive people should pay attention to symptoms.",
            "Moderate": "Consider reducing prolonged outdoor exertion if you experience discomfort.",
            "Poor": "Limit prolonged outdoor activity, especially for sensitive groups.",
            "Very Poor": "Reduce exposure and avoid strenuous outdoor activity where possible.",
            "Severe": "Avoid outdoor exertion and follow local health guidance.",
        }
        advisory = advisory_by_category[category]

        st.divider()
        st.subheader("Prediction result")
        result_col, info_col = st.columns([1.1, 1])
        with result_col:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max(0, min(prediction, 500)),
                number={"font": {"size": 42, "color": color},
                        "valueformat": ".1f"},
                title={
                    "text": f"ESTIMATED AQI<br><span style='font-size:16px;color:#67e8f9'>{category}</span>",
                    "font": {"size": 18, "color": "#e2e8f0"},
                },
                gauge={
                    "shape": "angular",
                    "axis": {"range": [0, 500], "tickwidth": 1, "tickcolor": "#94a3b8"},
                    "bar": {"color": color, "thickness": 0.28},
                    "bgcolor": "#132238",
                    "borderwidth": 1,
                    "bordercolor": "#294154",
                    "steps": [
                        {"range": [0, 50], "color": "#123b35"},
                        {"range": [50, 100], "color": "#263b2a"},
                        {"range": [100, 200], "color": "#403b20"},
                        {"range": [200, 300], "color": "#472c20"},
                        {"range": [300, 400], "color": "#472326"},
                        {"range": [400, 500], "color": "#3d2347"},
                    ],
                    "threshold": {
                        "line": {"color": "#67e8f9", "width": 4},
                        "thickness": 0.85,
                        "value": max(0, min(prediction, 500)),
                    },
                },
            ))
            gauge.update_layout(
                paper_bgcolor="#0b1728", plot_bgcolor="#0b1728",
                font={"color": "#e2e8f0"}, height=350,
                margin={"l": 25, "r": 25, "t": 65, "b": 15},
            )
            st.plotly_chart(gauge, use_container_width=True)

        with info_col:
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#102238,#10332f);
                    border:1px solid {color};border-radius:16px;padding:24px;margin-top:20px;">
                    <div style="color:#67e8f9;font-weight:800;letter-spacing:1.5px;font-size:0.8rem;">
                        ESTIMATED AIR QUALITY
                    </div>
                    <div style="font-size:2rem;font-weight:800;color:{color};margin:8px 0;">
                        {category}
                    </div>
                    <div style="font-size:1.1rem;color:#f1f5f9;">
                        Predicted AQI: {prediction:.1f}
                    </div>
                    <hr style="border-color:#294154;">
                    <div style="color:#67e8f9;font-weight:700;margin-bottom:8px;">HEALTH ADVISORY</div>
                    <div style="color:#e2e8f0;line-height:1.6;">{advisory}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if mongo_connected:
            record = {
                "PM2.5": float(pm25),
                "PM10": float(pm10),
                "NO2": float(no2),
                "CO": float(co),
                "predicted_AQI": round(prediction, 2),
                "category": category,
                "timestamp": datetime.now(),
            }
            try:
                result = predictions_collection.insert_one(record)
                st.success("Prediction saved to MongoDB.")
            except Exception as exc:
                st.error(
                    f"Prediction was calculated, but could not be saved: {exc}")
        else:
            st.warning(
                "Prediction calculated, but MongoDB is unavailable, so it was not saved.")

        st.caption(
            "This machine-learning estimate is not an official AQI reading and should not replace local monitoring or medical advice.")


# ============================================================
# PREDICTION HISTORY PAGE
# ============================================================
elif page == "🗂️ Prediction History":
    st.title("🗂️ Prediction History")
    st.write("Review the most recent AQI estimates saved by this app.")
    if not mongo_connected:
        st.warning("Connect to MongoDB to view prediction history.")
    else:
        try:
            records = list(
                predictions_collection.find({}, {"_id": 0})
                .sort("timestamp", -1)
                .limit(500)
            )
            if records:
                history = pd.DataFrame(records)
                if "timestamp" in history.columns:
                    history["timestamp"] = pd.to_datetime(
                        history["timestamp"], errors="coerce"
                    )
                if "predicted_AQI" in history.columns:
                    history["predicted_AQI"] = pd.to_numeric(
                        history["predicted_AQI"], errors="coerce"
                    )
                    valid = history["predicted_AQI"].dropna()
                    a, b, c = st.columns(3)
                    a.metric("Saved predictions", f"{len(history):,}")
                    b.metric("Average predicted AQI",
                             f"{valid.mean():.1f}" if not valid.empty else "N/A")
                    c.metric("Highest predicted AQI",
                             f"{valid.max():.1f}" if not valid.empty else "N/A")

                if "timestamp" in history.columns:
                    history = history.sort_values(
                        "timestamp", ascending=False, na_position="last")
                st.dataframe(history, use_container_width=True,
                             hide_index=True)
                st.caption(
                    "Showing up to the 500 most recent saved predictions.")
                download_df = history.copy()
                if "timestamp" in download_df.columns:
                    download_df["timestamp"] = download_df["timestamp"].astype(
                        str)
                st.download_button(
                    "⬇️ Download prediction history as CSV",
                    data=download_df.to_csv(index=False).encode("utf-8"),
                    file_name="aqi_prediction_history.csv",
                    mime="text/csv",
                )
            else:
                st.info(
                    "No predictions have been saved yet. Open AQI Prediction to create one.")
        except Exception as exc:
            st.error(f"Could not load prediction history: {exc}")


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================
elif page == "ℹ️ About Project":
    st.title("ℹ️ About the Project")
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Project overview</div>
            <h1>Air Pollution Prediction System</h1>
            <p>
                A Streamlit dashboard that combines historical air-quality
                exploration with a machine-learning estimate of AQI based on
                selected pollutant concentrations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What this app does")
        st.markdown(
            """
            - Displays historical air-quality records stored in MongoDB.
            - Compares cities by average and highest recorded AQI.
            - Explores historical pollutant concentrations and trends.
            - Predicts an estimated AQI from PM2.5, PM10, NO2, and CO.
            - Stores predictions in a separate MongoDB collection.
            """
        )
    with col2:
        st.subheader("Technology stack")
        st.markdown(
            """
            - **Python** — application logic
            - **Streamlit** — interactive dashboard
            - **Pandas** — data preparation and analysis
            - **Plotly** — charts and AQI gauge
            - **Scikit-learn / Joblib** — trained model loading and inference
            - **MongoDB** — historical data and prediction records
            """
        )

    st.divider()
    st.subheader("Dataset and model")
    st.markdown(
        """
        The project is configured to use the historical `city_day.csv` air-quality
        dataset imported into MongoDB's `historical_air_quality` collection.
        The prediction model expects four input features in this order:
        **PM2.5, PM10, NO2, CO**.
        """
    )
    st.info(
        "Important limitations: historical records may contain missing values; "
        "city averages depend on the available observations; and model predictions "
        "are estimates rather than official or real-time AQI readings. AQI category "
        "thresholds used in this interface are simplified project thresholds."
    )

    if not df.empty:
        st.subheader("Data availability")
        if "Date" in df.columns and df["Date"].notna().any():
            st.write(
                f"Date coverage: **{df['Date'].min():%d %b %Y}** to "
                f"**{df['Date'].max():%d %b %Y}**"
            )
        if "City" in df.columns:
            st.write(f"Cities represented: **{df['City'].nunique()}**")
        st.write(f"Historical records loaded: **{len(df):,}**")


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "🌿 Air Pollution Prediction System • Historical analysis + machine-learning estimates")
