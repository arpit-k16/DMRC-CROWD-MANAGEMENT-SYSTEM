import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.model_loader import load_models, preprocess_sample, predict_models

# Page config
st.set_page_config(page_title="Bheed-Mitra", page_icon="🚇", layout="wide", initial_sidebar_state="expanded")

# ----------------------
# Enhanced Custom CSS
# ----------------------
LIGHT_CSS = '''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    /* Ensure all text is visible in light mode */
    .main, .block-container, .stApp {
        color: #262730 !important;
    }
    
    p, div, span, li, td, th, label, .stMarkdown, .stText {
        color: #262730 !important;
    }
    
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    
    .big-title { 
        font-size: 48px; 
        font-weight: 800; 
        color: #1a1a2e !important; 
        padding: 20px 0; 
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-title { 
        font-size: 28px; 
        font-weight: 700; 
        color: #0057A6 !important; 
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #0057A6;
    }
    
    .card { 
        background: white; 
        padding: 25px; 
        border-radius: 16px; 
        box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
        margin-bottom: 20px;
        border-left: 4px solid #0057A6;
        transition: transform 0.2s;
        color: #262730 !important;
    }
    
    .card p, .card li, .card div, .card span, .card ul {
        color: #262730 !important;
    }
    
    .card h3 {
        color: #0057A6 !important;
    }
    
    .card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.15); }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stat-card * {
        color: white !important;
    }
    
    .metric-container {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin: 10px 0;
        color: #262730 !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    h1, h2, h3, h4, h5, h6 { 
        color: #1a1a2e !important; 
    }
    
    .info-box {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 15px 0;
        color: #1a237e !important;
    }
    
    .info-box p, .info-box div, .info-box span {
        color: #1a237e !important;
    }
    
    /* Streamlit specific elements */
    .stSelectbox label, .stTextInput label, .stNumberInput label {
        color: #262730 !important;
    }
    
    .stDataFrame, .stTable {
        color: #262730 !important;
    }
    
    /* Sidebar text */
    .css-1d391kg, .css-1lcbmhc, .css-1y4p8pa {
        color: #262730 !important;
    }
    
    /* Ensure all Streamlit text elements are visible */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] div,
    [data-testid="stText"],
    [class*="stText"],
    [class*="element-container"] p,
    [class*="element-container"] div {
        color: #262730 !important;
    }
    
    /* Streamlit write/text output */
    .stMarkdown, .stText {
        color: #262730 !important;
    }
</style>
'''

DARK_CSS = '''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    /* Ensure all text is visible in dark mode */
    .main, .block-container, .stApp {
        color: #E6EEF8 !important;
    }
    
    p, div, span, li, td, th, label, .stMarkdown, .stText {
        color: #E6EEF8 !important;
    }
    
    .main { background: linear-gradient(135deg, #0f1724 0%, #1e293b 100%); }
    
    .big-title { 
        font-size: 48px; 
        font-weight: 800; 
        color: #E6EEF8 !important; 
        padding: 20px 0; 
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
    }
    
    .big-title-text {
        background: linear-gradient(135deg, #9BD1FF 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .logo-img {
        height: 60px;
        width: auto;
        vertical-align: middle;
        display: inline-block;
    }
    
    .section-title { 
        font-size: 28px; 
        font-weight: 700; 
        color: #9BD1FF !important; 
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #9BD1FF;
    }
    
    .card { 
        background: #0b1220; 
        padding: 25px; 
        border-radius: 16px; 
        box-shadow: 0 8px 32px rgba(0,0,0,0.3); 
        margin-bottom: 20px;
        border-left: 4px solid #1f6feb;
        color: #E6EEF8 !important;
    }
    
    .card p, .card li, .card div, .card span, .card ul {
        color: #E6EEF8 !important;
    }
    
    .card h3 {
        color: #9BD1FF !important;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #1f6feb 0%, #764ba2 100%);
        color: white !important;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(31, 111, 235, 0.4);
    }
    
    .stat-card * {
        color: white !important;
    }
    
    .metric-container {
        background: #0b1220;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        margin: 10px 0;
        color: #E6EEF8 !important;
    }
    
    h1, h2, h3, h4, h5, h6 { 
        color: #E6EEF8 !important; 
    }
    
    .info-box {
        background: #1e3a5f;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f6feb;
        margin: 15px 0;
        color: #E6EEF8 !important;
    }
    
    .info-box p, .info-box div, .info-box span, .info-box strong {
        color: #E6EEF8 !important;
    }
    
    /* Streamlit specific elements in dark mode */
    .stSelectbox label, .stTextInput label, .stNumberInput label {
        color: #E6EEF8 !important;
    }
    
    .stDataFrame, .stTable {
        color: #E6EEF8 !important;
    }
    
    /* Ensure all Streamlit text elements are visible in dark mode */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] div,
    [data-testid="stText"],
    [class*="stText"],
    [class*="element-container"] p,
    [class*="element-container"] div {
        color: #E6EEF8 !important;
    }
    
    /* Streamlit write/text output */
    .stMarkdown, .stText {
        color: #E6EEF8 !important;
    }
</style>
'''

# ----------------------
# Helper Functions for Visualizations
# ----------------------
def create_crowd_distribution(df, station_col="Platform Crowd Level at Boarding Station"):
    """Create distribution histogram"""
    fig = px.histogram(
        df, x=station_col, nbins=5,
        title="Crowd Level Distribution",
        labels={station_col: "Crowd Level (1-5)", "count": "Frequency"},
        color_discrete_sequence=['#667eea']
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        height=350
    )
    return fig

def create_time_trend(df, station=None):
    """Create time-based trend chart"""
    if station:
        sub_df = df[df["Source Station"] == station]
    else:
        sub_df = df
    
    cat_order = ["Morning Peak", "Mid Morning", "Afternoon", "Evening Peak", "Night"]
    trend_data = sub_df.groupby("Boarding Time Category")["Platform Crowd Level at Boarding Station"].mean().reindex(cat_order).fillna(0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_data.index,
        y=trend_data.values,
        mode='lines+markers',
        name='Avg Crowd Level',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color='#764ba2')
    ))
    fig.update_layout(
        title="Crowd Level by Time Category",
        xaxis_title="Time Category",
        yaxis_title="Average Crowd Level (1-5)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        font=dict(size=12)
    )
    return fig

def create_wait_time_scatter(df, station=None):
    """Create wait time vs crowd level scatter"""
    if station:
        sub_df = df[df["Source Station"] == station]
    else:
        sub_df = df
    
    fig = px.scatter(
        sub_df, 
        x="Wait Time (mins)", 
        y="Platform Crowd Level at Boarding Station",
        color="Boarding Time Category",
        title="Wait Time vs Crowd Level",
        labels={"Wait Time (mins)": "Wait Time (minutes)", "Platform Crowd Level at Boarding Station": "Crowd Level"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        font=dict(size=12)
    )
    return fig

def create_station_comparison(df, top_n=10):
    """Compare top stations by average crowd"""
    station_stats = df.groupby("Source Station")["Platform Crowd Level at Boarding Station"].agg(['mean', 'count']).reset_index()
    station_stats = station_stats[station_stats['count'] >= 10]  # Filter stations with enough data
    station_stats = station_stats.nlargest(top_n, 'mean')
    
    fig = px.bar(
        station_stats,
        x='mean',
        y='Source Station',
        orientation='h',
        title=f"Top {top_n} Stations by Average Crowd Level",
        labels={'mean': 'Average Crowd Level', 'Source Station': 'Station'},
        color='mean',
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        font=dict(size=12),
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap for numeric features"""
    numeric_cols = ["Journey Time (mins)", "Wait Time (mins)", "Number of Line Changes", 
                   "official_ridership_scaled", "Platform Crowd Level at Boarding Station"]
    corr_data = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_data.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        font=dict(size=12)
    )
    return fig

def create_line_comparison(df):
    """Compare crowd levels across metro lines"""
    line_stats = df.groupby("Metro Line Used (Primary)")["Platform Crowd Level at Boarding Station"].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=line_stats.index,
        y=line_stats.values,
        title="Average Crowd Level by Metro Line",
        labels={'x': 'Metro Line', 'y': 'Average Crowd Level'},
        color=line_stats.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        font=dict(size=12),
        xaxis_tickangle=-45
    )
    return fig

# ----------------------
# Load data & models
# ----------------------
@st.cache_data
def load_data():
    return pd.read_csv("delhi_metro_cleaned_no_unknown.csv")

df = load_data()

try:
    xgb_model, encoder, scaler = load_models()
    models_ok = True
except Exception as e:
    st.error(f"Model load error: {e}")
    models_ok = False

# ----------------------
# Apply Dark Mode Permanently
# ----------------------
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ----------------------
# Sidebar: Settings
# ----------------------
if "role" not in st.session_state:
    st.session_state.role = None
if "live" not in st.session_state:
    st.session_state.live = False
if "live_buffer" not in st.session_state:
    st.session_state.live_buffer = []
if "live_speed" not in st.session_state:
    st.session_state.live_speed = 1.0  # 1x speed (1 second)
if "live_crowd" not in st.session_state:
    st.session_state.live_crowd = None
if "live_time_seconds" not in st.session_state:
    st.session_state.live_time_seconds = 36000  # Default: 10:00:00 (10 AM)

with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("---")
    
    st.markdown("### Dataset Overview")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Unique Stations", len(set(df["Source Station"].unique()) | set(df["Destination Station"].unique())))
    st.metric("Avg Crowd Level", f"{df['Platform Crowd Level at Boarding Station'].mean():.2f}")
    
    st.markdown("---")
    
    # Speed control and time setting for live simulation (admin only)
    if st.session_state.role == "admin":
        st.markdown("###  Live Simulation Speed")
        speed_options = {"1× (1 sec)": 1.0, "2× (0.5 sec)": 0.5, "4× (0.25 sec)": 0.25}
        speed_keys = list(speed_options.keys())
        # Find current speed index
        try:
            current_idx = [i for i, v in enumerate(speed_options.values()) if abs(v - st.session_state.live_speed) < 0.01][0]
        except (IndexError, KeyError):
            current_idx = 0
        speed_label = st.selectbox("Simulation Speed", speed_keys, 
                                   index=current_idx,
                                   help="Control the update frequency of live simulation")
        st.session_state.live_speed = speed_options[speed_label]
        st.markdown("---")
        
        # Time setting for live simulation
        st.markdown("###  Simulation Time")
        def seconds_to_hms(seconds):
            """Convert seconds since midnight to (hours, minutes, seconds)"""
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return hours, minutes, secs
        
        def hms_to_seconds(h, m, s):
            """Convert (hours, minutes, seconds) to seconds since midnight"""
            return h * 3600 + m * 60 + s
        
        # Generate time options (every 15 minutes)
        time_options = []
        for h in range(24):
            for m in range(0, 60, 15):
                time_str = f"{h:02d}:{m:02d}:00"
                time_options.append((time_str, hms_to_seconds(h, m, 0)))
        
        # Find current time in options
        current_h, current_m, current_s = seconds_to_hms(st.session_state.live_time_seconds)
        # Round to nearest 15 minutes for dropdown
        current_m_rounded = (current_m // 15) * 15
        current_time_str = f"{current_h:02d}:{current_m_rounded:02d}:00"
        
        # Get time options as list of strings
        time_str_options = [opt[0] for opt in time_options]
        try:
            current_time_idx = time_str_options.index(current_time_str)
        except ValueError:
            current_time_idx = 0
        
        selected_time_str = st.selectbox("Set Time (HH:MM:SS)", time_str_options,
                                        index=current_time_idx,
                                        help="Set the simulation time (updates every 15 minutes)")
        
        # Update time in session state
        selected_seconds = dict(time_options)[selected_time_str]
        st.session_state.live_time_seconds = selected_seconds
        
        # Display current time
        h, m, s = seconds_to_hms(st.session_state.live_time_seconds)
        st.caption(f"Current: {h:02d}:{m:02d}:{s:02d}")
        st.markdown("---")
    
    if st.button("🏠 Reset to Home"):
        st.session_state.role = None
        st.rerun()

# ----------------------
# Home / Role selection
# ----------------------
if st.session_state.role is None:
    st.markdown('<div class="big-title"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTMJmyJpGsL5dPBwAQU0UVQ_joLPmvXH15DUw&s" class="logo-img" alt="Bheed-Mitra Logo"> <span class="big-title-text">Bheed-Mitra</span></div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; font-size:18px; color:#9BD1FF; margin-bottom:40px;">Advanced Dashboard for Metro Crowd Prediction & Analytics</p>', unsafe_allow_html=True)
    
    # Overview Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(" Total Records", f"{len(df):,}", help="Total data points in dataset")
    with col2:
        st.metric(" Stations", len(set(df["Source Station"].unique()) | set(df["Destination Station"].unique())), help="Unique metro stations")
    with col3:
        st.metric(" Avg Crowd", f"{df['Platform Crowd Level at Boarding Station'].mean():.2f}/5", help="Average platform crowd level")
    with col4:
        st.metric(" Avg Wait", f"{df['Wait Time (mins)'].mean():.1f} min", help="Average wait time")
    
    st.markdown("---")
    
    # Quick Visualizations on Home Page
    st.markdown('<div class="section-title"> Insights </div>', unsafe_allow_html=True)
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.plotly_chart(create_crowd_distribution(df), use_container_width=True)
    with col_v2:
        st.plotly_chart(create_time_trend(df), use_container_width=True)
    
    col_v3, col_v4 = st.columns(2)
    with col_v3:
        st.plotly_chart(create_station_comparison(df, top_n=10), use_container_width=True)
    with col_v4:
        st.plotly_chart(create_line_comparison(df), use_container_width=True)
    
    st.markdown("---")
    
    # Role Selection
    st.markdown('<div class="section-title"> Access Dashboard</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top:0;">👨‍💼 Station Admin</h3>
            <p>Access advanced analytics, live simulations, and station-level insights.</p>
            <ul>
                <li>Real-time crowd monitoring</li>
                <li>Historical trend analysis</li>
                <li>What-if scenario planning</li>
                <li>Station performance metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button(" Login as Station Admin", key="admin_btn", use_container_width=True):
            st.session_state.role = "admin"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top:0;">👤 Passenger</h3>
            <p>Get personalized crowd predictions and travel recommendations.</p>
            <ul>
                <li>Crowd level predictions</li>
                <li>Optimal travel time suggestions</li>
                <li>Station-specific insights</li>
                <li>Real-time recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Login as Passenger", key="pass_btn", use_container_width=True):
            st.session_state.role = "passenger"
            st.rerun()
    st.stop()

# ----------------------
# ADMIN VIEW
# ----------------------
if st.session_state.role == "admin":
    st.markdown('<div class="section-title">👨‍💼 Station Admin Dashboard</div>', unsafe_allow_html=True)
    
    stations = sorted(set(df["Source Station"].unique()).union(df["Destination Station"].unique()))
    
    # Control Panel
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        station = st.selectbox("Select Station", stations, help="Choose a station to analyze")
    with col2:
        window = st.selectbox("Forecast Window", ["Now", "Next 1 Hour", "Next 3 Hours"], help="Prediction time horizon")
    with col3:
        if not st.session_state.live:
            if st.button("▶️ Start Live Simulation", use_container_width=True):
                st.session_state.live = True
                st.rerun()
        else:
            if st.button("⏸️ Stop Live Simulation", use_container_width=True):
                st.session_state.live = False
                st.rerun()

    # Filter data for selected station
    sub = df[df["Source Station"] == station]
    
    if sub.empty:
        st.warning(f"No data available for {station}")
        st.stop()
    
    # Compute statistics
    avg_platform = sub["Platform Crowd Level at Boarding Station"].mean()
    median_wait = sub["Wait Time (mins)"].median()
    peak_crowd = sub["Platform Crowd Level at Boarding Station"].max()
    min_crowd = sub["Platform Crowd Level at Boarding Station"].min()
    std_crowd = sub["Platform Crowd Level at Boarding Station"].std()
    total_journeys = len(sub)
    avg_journey_time = sub["Journey Time (mins)"].mean()

    # Enhanced KPI Cards
    st.markdown("### Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Platform Crowd", f"{avg_platform:.2f}/5", f"±{std_crowd:.2f}", help="Average historical crowd level")
    k2.metric("Median Wait Time", f"{median_wait:.1f} min", help="Typical waiting time")
    k3.metric("Peak Crowd Level", f"{peak_crowd}/5", f"Min: {min_crowd}/5", help="Maximum observed crowd")
    k4.metric("Total Journeys", f"{total_journeys:,}", help="Historical data points")
    
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Avg Journey Time", f"{avg_journey_time:.1f} min", help="Average travel time")
    k6.metric("Std Deviation", f"{std_crowd:.2f}", help="Crowd level variability")
    k7.metric("Most Common Line", sub["Metro Line Used (Primary)"].mode().iloc[0] if not sub.empty else "N/A", help="Primary metro line")
    k8.metric("Peak Time Category", sub["Boarding Time Category"].mode().iloc[0] if not sub.empty else "N/A", help="Busiest time period")
    
    # Live Simulation Section
    st.markdown("### 🔴 Live Crowd Monitoring")
    live_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    def seconds_to_hms(seconds):
        """Convert seconds since midnight to (hours, minutes, seconds)"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return hours, minutes, secs
    
    def get_time_category_multiplier(time_seconds):
        """Determine time-based crowd multiplier based on actual time"""
        # Convert seconds to hours (with decimal precision)
        hour = time_seconds / 3600.0
        
        if 6 <= hour < 10:  # Morning Peak
            base_mult = 1.3 + 0.2 * np.sin((hour - 6) * np.pi / 4)  # Gradual rise with spikes
            volatility = 0.15
        elif 10 <= hour < 14:  # Mid Morning
            base_mult = 1.0 + 0.1 * np.sin(hour * np.pi / 4)
            volatility = 0.08
        elif 14 <= hour < 17:  # Afternoon
            base_mult = 0.9 + 0.1 * np.sin(hour * np.pi / 6)
            volatility = 0.1
        elif 17 <= hour < 21:  # Evening Peak
            base_mult = 1.4 + 0.3 * np.sin((hour - 17) * np.pi / 4)  # More volatility
            volatility = 0.2
        else:  # Night
            base_mult = 0.6 + 0.1 * np.sin(hour * np.pi / 12)  # Smooth drop
            volatility = 0.05
        
        return base_mult, volatility
    
    def compute_live_realistic(base, time_seconds):
        """Realistic dynamic simulation model"""
        # Get time-based multipliers
        base_mult, volatility = get_time_category_multiplier(time_seconds)
        
        # Base value with time adjustment
        adjusted_base = base * base_mult
        
        # Sinusoidal component for organic movement (based on time in minutes)
        minutes = time_seconds / 60.0
        sin_component = 0.2 * np.sin(minutes * 0.1)
        
        # Random variation with volatility
        random_var = np.random.normal(0, volatility)
        
        # Rare event spike (5% chance)
        event_spike = 0
        if np.random.random() < 0.05:
            event_spike = np.random.uniform(1.0, 2.0)  # 1-2 level increase
        
        # Combine all components
        raw_value = adjusted_base + sin_component + random_var + event_spike
        
        # Clamp to valid range [1, 5]
        crowd_level = max(1, min(5, round(raw_value, 1)))
        
        return crowd_level
    
    if st.session_state.live:
        # Initialize base crowd level (historical average)
        base = avg_platform if avg_platform > 0 else df["Platform Crowd Level at Boarding Station"].mean()
        if st.session_state.live_crowd is None:
            st.session_state.live_crowd = base
        
        # Get current time
        current_time_seconds = st.session_state.live_time_seconds
        h, m, s = seconds_to_hms(current_time_seconds)
        time_str = f"{h:02d}:{m:02d}:{s:02d}"
        
        # Compute new crowd level (use historical base, not smoothed value)
        instant_crowd = compute_live_realistic(base, current_time_seconds)
        
        # Add to buffer (keep last 20 points for chart)
        # Store as tuple: (time_string, crowd_level)
        st.session_state.live_buffer.append((time_str, instant_crowd))
        if len(st.session_state.live_buffer) > 20:
            st.session_state.live_buffer.pop(0)
        
        # Compute smoothed value (moving average of last 10 values)
        if len(st.session_state.live_buffer) >= 10:
            smoothed_crowd = np.mean([item[1] for item in st.session_state.live_buffer[-10:]])
        else:
            smoothed_crowd = np.mean([item[1] for item in st.session_state.live_buffer]) if st.session_state.live_buffer else instant_crowd
        
        # Display metrics with time
        delta_val = smoothed_crowd - avg_platform
        live_placeholder.markdown(f"""
        <div>
            <div style="text-align: center; margin-bottom: 15px;">
                <h3 style="color: #9BD1FF; margin: 0;">🕐 Current Time: {time_str}</h3>
            </div>
            <div style="display: flex; gap: 20px; align-items: center;">
                <div style="flex: 1;">
                    <h4>🔴 Live Crowd Level (Instantaneous)</h4>
                    <h2 style="color: {'#4caf50' if instant_crowd <= 2 else '#ff9800' if instant_crowd <= 3 else '#f44336'}; margin: 0;">
                        {instant_crowd:.1f}/5
                    </h2>
                </div>
                <div style="flex: 1;">
                    <h4> Smoothed Crowd Level (10-point MA)</h4>
                    <h2 style="color: {'#4caf50' if smoothed_crowd <= 2 else '#ff9800' if smoothed_crowd <= 3 else '#f44336'}; margin: 0;">
                        {smoothed_crowd:.2f}/5
                    </h2>
                    <p style="margin: 5px 0; color: {'green' if delta_val < 0 else 'red' if delta_val > 0 else 'gray'};">
                        {delta_val:+.2f} vs Historical Avg
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Live trend chart
        recent = sub.groupby("Boarding Time Category")["Platform Crowd Level at Boarding Station"].mean().reindex(["Morning Peak","Mid Morning","Afternoon","Evening Peak","Night"]).fillna(0)
        chart_placeholder.line_chart(recent)
        
        # Increment time by 1 second (or based on speed multiplier)
        # Each simulation step advances time by 1 second
        st.session_state.live_time_seconds = (st.session_state.live_time_seconds + 1) % 86400  # Wrap around at midnight
        
        # Respect speed control
        speed_delay = st.session_state.live_speed
        time.sleep(speed_delay)
        st.rerun()
    else:
        # Reset state when stopped (but keep time setting)
        if st.session_state.live_crowd is not None:
            st.session_state.live_crowd = None
            st.session_state.live_buffer = []
        
        live_placeholder.metric("⏸️ Live Crowd Level (Paused)", f"{int(round(avg_platform))}/5", help="Click 'Start Live Simulation' to enable")
        
        # Live trend chart
        recent = sub.groupby("Boarding Time Category")["Platform Crowd Level at Boarding Station"].mean().reindex(["Morning Peak","Mid Morning","Afternoon","Evening Peak","Night"]).fillna(0)
        chart_placeholder.line_chart(recent)

    st.markdown("---")
    
    # Advanced Visualizations
    st.markdown("### Analytics & Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs([" Trends & Patterns", " Distributions", " Correlations", " Station Comparison"])
    
    with tab1:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.plotly_chart(create_time_trend(sub, station), use_container_width=True)
        with col_t2:
            st.plotly_chart(create_wait_time_scatter(sub, station), use_container_width=True)
        
        # Additional trend: Day type comparison
        day_comparison = sub.groupby("Day-Type")["Platform Crowd Level at Boarding Station"].mean()
        fig_day = px.bar(
            x=day_comparison.index,
            y=day_comparison.values,
            title="Crowd Level: Weekday vs Weekend",
            labels={'x': 'Day Type', 'y': 'Average Crowd Level'},
            color=day_comparison.values,
            color_continuous_scale='Blues'
        )
        fig_day.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig_day, use_container_width=True)
    
    with tab2:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.plotly_chart(create_crowd_distribution(sub), use_container_width=True)
        with col_d2:
            # Wait time distribution
            fig_wait = px.histogram(
                sub, x="Wait Time (mins)", nbins=20,
                title="Wait Time Distribution",
                labels={"Wait Time (mins)": "Wait Time (minutes)", "count": "Frequency"},
                color_discrete_sequence=['#764ba2']
            )
            fig_wait.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=350)
            st.plotly_chart(fig_wait, use_container_width=True)
        
        # Statistical Summary
        st.markdown("#### Statistical Summary")
        st.dataframe(sub[["Platform Crowd Level at Boarding Station", "Wait Time (mins)", "Journey Time (mins)"]].describe(), use_container_width=True)
    
    with tab3:
        st.plotly_chart(create_correlation_heatmap(sub), use_container_width=True)
        
        # Feature importance visualization (simplified)
        st.markdown("#### Feature Impact Analysis")
        feature_impact = {
            "Boarding Time Category": sub.groupby("Boarding Time Category")["Platform Crowd Level at Boarding Station"].mean().std(),
            "Wait Time": sub["Wait Time (mins)"].corr(sub["Platform Crowd Level at Boarding Station"]),
            "Ridership": sub["official_ridership_scaled"].corr(sub["Platform Crowd Level at Boarding Station"]),
            "Journey Time": sub["Journey Time (mins)"].corr(sub["Platform Crowd Level at Boarding Station"])
        }
        fig_impact = px.bar(
            x=list(feature_impact.keys()),
            y=list(feature_impact.values()),
            title="Feature Impact on Crowd Level",
            labels={'x': 'Feature', 'y': 'Impact Score'},
            color=list(feature_impact.values()),
            color_continuous_scale='RdYlGn'
        )
        fig_impact.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig_impact, use_container_width=True)
    
    with tab4:
        st.plotly_chart(create_station_comparison(df, top_n=15), use_container_width=True)
        st.plotly_chart(create_line_comparison(df), use_container_width=True)

    st.markdown("---")
    
    # What-If Scenario Analysis
    st.markdown("### What-If Scenario Analysis")
    st.markdown("Test different conditions to predict crowd levels under various scenarios.")
    
    colx, coly, colz = st.columns(3)
    with colx:
        wf_time = st.selectbox(" Boarding Time", ["Morning Peak", "Mid Morning", "Afternoon", "Evening Peak", "Night"], key='wf_time')
    with coly:
        wf_event = st.selectbox(" Special Event Nearby?", ["No", "Yes"], key='wf_event')
    with colz:
        wf_weather = st.selectbox(" Weather Condition", ["Clear", "Cloudy", "Light Rain", "Heavy Rain"], key='wf_weather')
    
    if st.button(" Run Scenario Analysis", use_container_width=True):
        sample = {
            "Source Station": station,
            "Destination Station": station,
            "Metro Line Used (Primary)": sub["Metro Line Used (Primary)"].mode().iloc[0] if not sub.empty else "Unknown",
            "Boarding Time Category": wf_time,
            "Day-Type": "Weekday",
            "Weather Condition": wf_weather,
            "Purpose of Travel": "General",
            "Age Group": "25-34",
            "Frequency of Metro Usage": "Daily",
            "Journey Time (mins)": int(sub["Journey Time (mins)"].median() if not sub.empty else 15),
            "Wait Time (mins)": int(sub["Wait Time (mins)"].median() if not sub.empty else 3),
            "Number of Line Changes": int(sub["Number of Line Changes"].median() if not sub.empty else 0),
            "official_ridership_scaled": float(sub["official_ridership_scaled"].mean() if not sub.empty else 0.5),
            "Could You Get a Seat Immediately?": "No" if wf_event=="Yes" else "Yes",
            "Peak Crowd Point During Journey": "Boarding Station",
            "Special Event Nearby": wf_event
        }
        xgb, enc, scaler = load_models()
        X = preprocess_sample(sample, enc, scaler)
        preds = predict_models(X, xgb)
        
        crowd_level = preds['xgb']
        st.success(f" **Prediction:** {crowd_level}/5")
        
        # Recommendations
        if crowd_level >= 4:
            st.error("🚨 **High Alert:** Deploy additional staff, increase train frequency, and consider crowd management measures.")
        elif crowd_level == 3:
            st.warning("⚠️ **Moderate:** Monitor closely and prepare for potential increases.")
        else:
            st.success("✅ **Normal Operations:** Standard monitoring sufficient.")
 
    
# ----------------------
# Helper function to determine time category from hour
# ----------------------
def get_time_category_from_hour(hour):
    """Determine time category based on hour (0-23)"""
    if 6 <= hour < 10:
        return "Morning Peak"
    elif 10 <= hour < 14:
        return "Mid Morning"
    elif 14 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening Peak"
    else:  # 21-23 or 0-5
        return "Night"

def get_boarding_time_category(travel_time_option):
    """Convert travel time option to actual time category based on current time"""
    now = datetime.now()
    
    if travel_time_option == "Now":
        # Use current time
        hour = now.hour
        return get_time_category_from_hour(hour)
    elif travel_time_option == "In 30 minutes":
        # Calculate time 30 minutes from now
        future_time = now + timedelta(minutes=30)
        hour = future_time.hour
        return get_time_category_from_hour(hour)
    elif travel_time_option == "1 hour":
        # Calculate time 1 hour from now
        future_time = now + timedelta(hours=1)
        hour = future_time.hour
        return get_time_category_from_hour(hour)
    elif travel_time_option == "Morning Peak":
        return "Morning Peak"
    elif travel_time_option == "Evening Peak":
        return "Evening Peak"
    else:
        # Default fallback
        return "Afternoon"

# ----------------------
# PASSENGER VIEW
# ----------------------
if st.session_state.role == "passenger":
    st.markdown('<div class="section-title">👤 Passenger Journey Planner</div>', unsafe_allow_html=True)
    
    stations = sorted(set(df["Source Station"].unique()).union(df["Destination Station"].unique()))
    
    # Input Section
    st.markdown("### Plan Your Journey")
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        src = st.selectbox(" Select Your Station", stations, help="Choose your boarding station")
        dest = st.selectbox(" Destination Station", stations, help="Choose your destination", index=min(1, len(stations)-1))
    with col2:
        travel_time = st.selectbox(" When are you travelling?", ["Now", "In 30 minutes", "1 hour", "Morning Peak", "Evening Peak"], help="Select your travel time")
    with col3:
        weather = st.selectbox(" Weather", ["Clear", "Cloudy", "Light Rain", "Heavy Rain"], help="Current weather condition")

    # Auto-fill logic - Calculate actual time category based on current time
    boarding_time = get_boarding_time_category(travel_time)
    
    # Display calculated time info for user transparency
    if travel_time in ["Now", "In 30 minutes", "1 hour"]:
        now = datetime.now()
        if travel_time == "Now":
            display_time = now.strftime("%H:%M")
            time_info = f"Current time: {display_time}"
        elif travel_time == "In 30 minutes":
            future_time = now + timedelta(minutes=30)
            display_time = future_time.strftime("%H:%M")
            time_info = f"Travel time: {display_time} (30 minutes from now)"
        else:  # 1 hour
            future_time = now + timedelta(hours=1)
            display_time = future_time.strftime("%H:%M")
            time_info = f"Travel time: {display_time} (1 hour from now)"

        st.info(f" {time_info} → Time Category: **{boarding_time}**")
    
    ridership_map = dict(zip(df["Source Station"], df["official_ridership_scaled"]))
    
    sample = {
        "Source Station": src, 
        "Destination Station": dest,
        "Metro Line Used (Primary)": df[df["Source Station"]==src]["Metro Line Used (Primary)"].mode().iloc[0] if not df[df["Source Station"]==src].empty else "Unknown",
        "Boarding Time Category": boarding_time, 
        "Day-Type": "Weekday",
        "Weather Condition": weather, 
        "Purpose of Travel": "General",
        "Age Group": "25-34", 
        "Frequency of Metro Usage": "Daily",
        "Journey Time (mins)": 15, 
        "Wait Time (mins)": 3, 
        "Number of Line Changes": 0,
        "official_ridership_scaled": ridership_map.get(src, df["official_ridership_scaled"].mean()),
        "Could You Get a Seat Immediately?": "No", 
        "Peak Crowd Point During Journey": "Boarding Station", 
        "Special Event Nearby": "No"
    }

    if st.button("Check Crowd Level & Get Recommendations", key='pass_check', use_container_width=True):
        xgb, enc, scaler = load_models()
        X = preprocess_sample(sample, enc, scaler)
        preds = predict_models(X, xgb)
        crowd = preds["xgb"]

        # Enhanced Prediction Display
        emoji_map = {
            1: "🟢 Very Low",
            2: "🟡 Low", 
            3: "🟠 Moderate",
            4: "🔴 High",
            5: "🔴🔴 Very High"
        }
        
        st.markdown("---")
        st.markdown(f"""
        <div class="card" style="text-align:center; background: linear-gradient(135deg, {'#4caf50' if crowd <= 2 else '#ff9800' if crowd == 3 else '#f44336'} 0%, {'#81c784' if crowd <= 2 else '#ffb74d' if crowd == 3 else '#e57373'} 100%); color: white;">
            <h1 style="color: white; margin: 10px 0;">Predicted Crowd Level</h1>
            <h2 style="color: white; font-size: 72px; margin: 20px 0;">{crowd}/5</h2>
            <h3 style="color: white; margin: 10px 0;">{emoji_map[crowd]}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Station Statistics
        src_data = df[df["Source Station"] == src]
        if not src_data.empty:
            st.markdown("###  Station Statistics")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric("Historical Avg", f"{src_data['Platform Crowd Level at Boarding Station'].mean():.2f}/5")
            stat_col2.metric("Typical Wait", f"{src_data['Wait Time (mins)'].median():.1f} min")
            stat_col3.metric("Peak Level", f"{src_data['Platform Crowd Level at Boarding Station'].max()}/5")
            stat_col4.metric("Data Points", f"{len(src_data):,}")
        
        # Recommendations
        st.markdown("###  Recommendations")
        if crowd >= 4:
            st.error("""
            **🚨 High Crowd Expected:**
            - Consider delaying your journey by 30-60 minutes
            - Check alternate stations nearby
            - Allow extra time for boarding
            - Be prepared for longer wait times
            """)
        elif crowd == 3:
            st.warning("""
            **⚠️ Moderate Crowd:**
            - Current time is acceptable but not optimal
            - Consider travelling slightly earlier or later
            - Expect moderate wait times
            """)
        else:
            st.success("""
            **✅ Low Crowd - Great Time to Travel!**
            - Optimal travel conditions
            - Minimal wait times expected
            - Comfortable boarding experience
            """)
        
        # Visualizations for Passenger
        st.markdown("###  Station Insights")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            if not src_data.empty:
                st.plotly_chart(create_time_trend(src_data, src), use_container_width=True)
        with col_v2:
            if not src_data.empty:
                st.plotly_chart(create_crowd_distribution(src_data), use_container_width=True)
        
        # Best time to travel
        if not src_data.empty:
            best_times = src_data.groupby("Boarding Time Category")["Platform Crowd Level at Boarding Station"].mean().sort_values()
            st.markdown("#### ⏰ Best Times to Travel (Lowest Crowd)")
            best_time_df = pd.DataFrame({
                "Time Category": best_times.index,
                "Avg Crowd Level": best_times.values
            })
            st.dataframe(best_time_df, use_container_width=True, hide_index=True)
    
    else:
        # Show general insights before prediction
        st.markdown("### Quick Station Insights")
        if src in stations:
            src_data = df[df["Source Station"] == src]
            if not src_data.empty:
                col_insight1, col_insight2 = st.columns(2)
                with col_insight1:
                    st.plotly_chart(create_time_trend(src_data, src), use_container_width=True)
                with col_insight2:
                    st.plotly_chart(create_crowd_distribution(src_data), use_container_width=True)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1,1,1])
if footer_col1.button("🏠 Back to Home"):
    st.session_state.role = None
    st.rerun()
if footer_col2.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
if footer_col3.button("🚪 Exit App"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

