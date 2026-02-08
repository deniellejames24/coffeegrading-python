"""
Robusta Coffee Grading Analytics Dashboard
Based on Philippine National Standards (PNS) & ATI
Streamlit Interactive Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
import pickle
import io

# Page configuration
st.set_page_config(
    page_title="Robusta Coffee Grading Dashboard - PNS",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - EXACT SAME AS ORIGINAL
st.markdown("""
    <style>
    /* Force light theme for better visibility */
    .stApp {
        background-color: #D2DCB6;
    }
    
    /* Metric containers - ensure text is visible */
    [data-testid="stMetricContainer"] {
        background-color: #1e1e1e !important;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    [data-testid="stMetricContainer"] label {
        color: #fafafa !important;
    }
    
    [data-testid="stMetricContainer"] [data-testid="stMetricValue"] {
        color: #fafafa !important;
    }
    
    [data-testid="stMetricContainer"] [data-testid="stMetricDelta"] {
        color: #fafafa !important;
    }
    
    /* Info boxes - ensure text is visible */
    [data-baseweb="notification"] {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    
    .stAlert {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #1e1e1e !important;
    }
    
    .stDataFrame table {
        color: #fafafa !important;
    }
    
    .stDataFrame th {
        background-color: #2c5f2d !important;
        color: #ffffff !important;
    }
    
    .stDataFrame td {
        color: #fafafa !important;
    }
    
    /* General text elements */
    .main .block-container {
        color: #fafafa !important;
    }
    
    .stMarkdown {
        color: #fafafa !important;
    }
    
    .stMarkdown p {
        color: #fafafa !important;
    }
    
    .stMarkdown li {
        color: #fafafa !important;
    }
    
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    
    /* Headings */
    h1 {
        color: #2c5f2d !important;
    }
    
    h2 {
        color: #97bc62 !important;
    }
    
    h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2c5f2d;
        color: white;
    }
    
    /* Tables */
    table {
        color: #fafafa !important;
    }
    
    /* Sidebar text */
    .css-1d391kg {
        color: #fafafa !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #cccccc !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #fafafa !important;
    }
    
    /* Radio buttons and selectboxes */
    .stRadio label {
        color: #fafafa !important;
    }
    
    .stSelectbox label {
        color: #fafafa !important;
    }
    
    .stNumberInput label {
        color: #fafafa !important;
    }
    
    .stSlider label {
        color: #fafafa !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================
# FUNCTIONS
# =====================================

@st.cache_data
def load_data():
    """Load the coffee dataset"""
    try:
        df = pd.read_csv('robusta_coffee_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file 'robusta_coffee_dataset.csv' not found. Please ensure the file is in the same directory.")
        st.stop()

def calculate_pns_grade(total_defect_pct):
    """
    Calculate PNS grade based on total defect percentage for Robusta
    Grade 1: max 10%
    Grade 2: max 15%
    Grade 3: max 25%
    Grade 4: max 40%
    """
    if total_defect_pct <= 10:
        return 1
    elif total_defect_pct <= 15:
        return 2
    elif total_defect_pct <= 25:
        return 3
    elif total_defect_pct <= 40:
        return 4
    else:
        return 5  # Below standard

def calculate_fine_premium_grade(primary_defects, secondary_defects, bean_screen_size_mm):
    """
    Calculate Fine/Premium Robusta grade based on CQI/UCDA standards
    UPDATED: No cupping score required - based on defects and bean size only
    
    Fine Robusta: 0 primary defects, max 5 secondary defects, bean size >= 6.5mm
    Premium Robusta: max 12 combined defects, bean size >= 6.0mm
    Commercial: otherwise
    
    Note: Cupping score requirement removed as it's not available for beneficiaries
    Bean screen size is now a key quality indicator
    """
    if primary_defects == 0 and secondary_defects <= 5 and bean_screen_size_mm >= 6.5:
        return 'Fine'
    elif (primary_defects + secondary_defects) <= 12 and bean_screen_size_mm >= 6.0:
        return 'Premium'
    else:
        return 'Commercial'

def classify_bean_size(screen_size_mm):
    """
    Classify bean size for Robusta based on screen size
    Large: retained by 5.6mm screen (dry processed) or 7.5mm (wet processed)
    Small: passes through but retained by smaller screens
    """
    if screen_size_mm >= 7.5:
        return 'Large'
    elif screen_size_mm >= 6.5:
        return 'Medium'
    elif screen_size_mm >= 5.5:
        return 'Small'
    else:
        return 'Below Standard'

@st.cache_data
def engineer_features(df):
    """Create engineered features for Robusta grading"""
    df_eng = df.copy()
    
    # Ensure we're only working with Robusta variety
    if 'variety' in df_eng.columns:
        df_eng = df_eng[df_eng['variety'].str.lower().str.contains('robusta')].copy()
    
    # Convert bean screen size from inches to mm if needed
    if 'bean_screen_size_inches' in df_eng.columns:
        df_eng['bean_screen_size_mm'] = df_eng['bean_screen_size_inches'] * 25.4
    elif 'bean_screen_size_mm' not in df_eng.columns:
        # Simulate from plant characteristics if not available
        df_eng['bean_screen_size_mm'] = (
            (df_eng['plant_height_cm'] / 200) * 2 +
            (df_eng['trunk_diameter_cm'] / 15) * 3 +
            4.5
        ).clip(4.0, 9.0)
    
    # Age categories
    df_eng['age_category'] = pd.cut(
        df_eng['plant_age_months'],
        bins=[0, 24, 48, 72, 100],
        labels=['Young', 'Mature', 'Prime', 'Old']
    )
    
    # Handle defects - use actual data if available, otherwise simulate
    if 'primary_defects' not in df_eng.columns:
        # Simulate from quality score if defects not available
        df_eng['total_defect_pct'] = 50 - (df_eng['quality_score'] / 2)
        df_eng['total_defect_pct'] = df_eng['total_defect_pct'].clip(0, 50)
        df_eng['primary_defects'] = (df_eng['total_defect_pct'] * 0.3).round().astype(int)
        df_eng['secondary_defects'] = (df_eng['total_defect_pct'] * 0.7).round().astype(int)
    else:
        # Calculate total defect percentage from actual defects
        # Assume 350g sample, average bean weight ~0.15g per bean
        total_beans_sample = 350 / 0.15
        df_eng['total_defect_pct'] = ((df_eng['primary_defects'] + df_eng['secondary_defects']) / total_beans_sample) * 100
    
    # PNS Grade (1-4)
    df_eng['pns_grade'] = df_eng['total_defect_pct'].apply(calculate_pns_grade)
    
    # Fine/Premium/Commercial classification - now based on defects and bean size only
    df_eng['coffee_grade'] = df_eng.apply(
        lambda x: calculate_fine_premium_grade(
            x['primary_defects'], 
            x['secondary_defects'], 
            x['bean_screen_size_mm']
        ),
        axis=1
    )
    
    df_eng['bean_size_class'] = df_eng['bean_screen_size_mm'].apply(classify_bean_size)
    
    # Climate suitability for Robusta (elevation 600-1200 masl, temp 13-26°C)
    if 'elevation_masl' in df_eng.columns:
        elevation_score = 1 - abs(df_eng['elevation_masl'] - 900) / 600
        elevation_score = elevation_score.clip(0, 1)
    else:
        elevation_score = 0.8
    
    temp_score = 1 - abs(df_eng['monthly_temp_avg_c'] - 19.5) / 13
    temp_score = temp_score.clip(0, 1)
    
    # Robusta requires 200mm rainfall
    rainfall_score = (df_eng['monthly_rainfall_mm'] / 200).clip(0, 1.5)
    rainfall_score = rainfall_score.clip(0, 1)
    
    df_eng['climate_suitability_robusta'] = (
        temp_score * 0.4 + 
        rainfall_score * 0.4 + 
        elevation_score * 0.2
    )
    
    # Soil suitability for Robusta (pH 5.6-6.5)
    optimal_ph = 6.0
    df_eng['soil_suitability_robusta'] = 1 - (abs(df_eng['soil_pH'] - optimal_ph) / 1.5).clip(0, 1)
    
    # Moisture suitability
    df_eng['moisture_suitability'] = (df_eng['soil_moisture_pct'] / 35).clip(0, 1)
    
    # Overall quality index
    df_eng['overall_quality_index'] = (
        df_eng['climate_suitability_robusta'] * 0.3 +
        df_eng['soil_suitability_robusta'] * 0.3 +
        df_eng['moisture_suitability'] * 0.2 +
        (1 - df_eng['environmental_stress_index']) * 0.2
    )
    
    # Production ready (Robusta bears fruit at 36 months)
    df_eng['production_ready'] = (df_eng['plant_age_months'] >= 36).astype(int)
    
    return df_eng

def prepare_harvest_yield_data(df):
    """
    Prepare data for harvest yield prediction
    Handles seasonal patterns where harvest only occurs in Nov, Dec, Jan, Feb, Mar
    """
    # Create month columns if they exist in dataset
    harvest_months = ['nov_yield', 'dec_yield', 'jan_yield', 'feb_yield', 'mar_yield']
    
    # Stack the data to have one row per month observation
    records = []
    
    for idx, row in df.iterrows():
        # For November-December (Year 1)
        if 'nov_yield' in df.columns and pd.notna(row.get('nov_yield', np.nan)):
            records.append({
                'plant_age_months': row['plant_age_months'],
                'elevation_masl': row.get('elevation_masl', 900),
                'monthly_temp_avg_c': row['monthly_temp_avg_c'],
                'monthly_rainfall_mm': row['monthly_rainfall_mm'],
                'soil_pH': row['soil_pH'],
                'soil_moisture_pct': row['soil_moisture_pct'],
                'fertilization_freq': row.get('fertilization_frequency', 3),
                'pest_management_freq': row.get('pest_management_frequency', 3),
                'month': 11,  # November
                'harvest_yield_kg': row['nov_yield']
            })
        
        if 'dec_yield' in df.columns and pd.notna(row.get('dec_yield', np.nan)):
            records.append({
                'plant_age_months': row['plant_age_months'] + 1,
                'elevation_masl': row.get('elevation_masl', 900),
                'monthly_temp_avg_c': row['monthly_temp_avg_c'],
                'monthly_rainfall_mm': row['monthly_rainfall_mm'],
                'soil_pH': row['soil_pH'],
                'soil_moisture_pct': row['soil_moisture_pct'],
                'fertilization_freq': row.get('fertilization_frequency', 3),
                'pest_management_freq': row.get('pest_management_frequency', 3),
                'month': 12,  # December
                'harvest_yield_kg': row['dec_yield']
            })
        
        # For January-March (Year 2)
        if 'jan_yield' in df.columns and pd.notna(row.get('jan_yield', np.nan)):
            records.append({
                'plant_age_months': row['plant_age_months'] + 2,
                'elevation_masl': row.get('elevation_masl', 900),
                'monthly_temp_avg_c': row['monthly_temp_avg_c'],
                'monthly_rainfall_mm': row['monthly_rainfall_mm'],
                'soil_pH': row['soil_pH'],
                'soil_moisture_pct': row['soil_moisture_pct'],
                'fertilization_freq': row.get('fertilization_frequency', 3),
                'pest_management_freq': row.get('pest_management_frequency', 3),
                'month': 1,  # January
                'harvest_yield_kg': row['jan_yield']
            })
        
        if 'feb_yield' in df.columns and pd.notna(row.get('feb_yield', np.nan)):
            records.append({
                'plant_age_months': row['plant_age_months'] + 3,
                'elevation_masl': row.get('elevation_masl', 900),
                'monthly_temp_avg_c': row['monthly_temp_avg_c'],
                'monthly_rainfall_mm': row['monthly_rainfall_mm'],
                'soil_pH': row['soil_pH'],
                'soil_moisture_pct': row['soil_moisture_pct'],
                'fertilization_freq': row.get('fertilization_frequency', 3),
                'pest_management_freq': row.get('pest_management_frequency', 3),
                'month': 2,  # February
                'harvest_yield_kg': row['feb_yield']
            })
        
        if 'mar_yield' in df.columns and pd.notna(row.get('mar_yield', np.nan)):
            records.append({
                'plant_age_months': row['plant_age_months'] + 4,
                'elevation_masl': row.get('elevation_masl', 900),
                'monthly_temp_avg_c': row['monthly_temp_avg_c'],
                'monthly_rainfall_mm': row['monthly_rainfall_mm'],
                'soil_pH': row['soil_pH'],
                'soil_moisture_pct': row['soil_moisture_pct'],
                'fertilization_freq': row.get('fertilization_frequency', 3),
                'pest_management_freq': row.get('pest_management_frequency', 3),
                'month': 3,  # March
                'harvest_yield_kg': row['mar_yield']
            })
    
    harvest_df = pd.DataFrame(records)
    
    # Add derived features
    if len(harvest_df) > 0:
        # Climate suitability
        harvest_df['elevation_score'] = 1 - abs(harvest_df['elevation_masl'] - 900) / 600
        harvest_df['elevation_score'] = harvest_df['elevation_score'].clip(0, 1)
        
        harvest_df['temp_score'] = 1 - abs(harvest_df['monthly_temp_avg_c'] - 19.5) / 13
        harvest_df['temp_score'] = harvest_df['temp_score'].clip(0, 1)
        
        harvest_df['rainfall_score'] = (harvest_df['monthly_rainfall_mm'] / 200).clip(0, 1.5)
        harvest_df['rainfall_score'] = harvest_df['rainfall_score'].clip(0, 1)
        
        harvest_df['climate_suitability'] = (
            harvest_df['temp_score'] * 0.4 + 
            harvest_df['rainfall_score'] * 0.4 + 
            harvest_df['elevation_score'] * 0.2
        )
        
        # Soil suitability
        harvest_df['soil_suitability'] = 1 - (abs(harvest_df['soil_pH'] - 6.0) / 1.5).clip(0, 1)
        
        # Moisture suitability
        harvest_df['moisture_suitability'] = (harvest_df['soil_moisture_pct'] / 35).clip(0, 1)
        
        # Production readiness
        harvest_df['production_ready'] = (harvest_df['plant_age_months'] >= 36).astype(int)
        
        # Month cyclical encoding for seasonality
        harvest_df['month_sin'] = np.sin(2 * np.pi * harvest_df['month'] / 12)
        harvest_df['month_cos'] = np.cos(2 * np.pi * harvest_df['month'] / 12)
    
    return harvest_df

# CONTINUATION - Helper Functions

def plot_confusion_matrix(cm, labels, title):
    """Create confusion matrix plot using plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=500,
        font=dict(size=12)
    )
    
    return fig

def plot_feature_importance(importance_df, top_n=15):
    """Plot feature importance"""
    top_features = importance_df.head(top_n).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=top_features['Importance'],
        y=top_features['Feature'],
        orientation='h',
        marker=dict(color='#2c5f2d')
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Most Important Features",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=600,
        showlegend=False
    )
    
    return fig

def calculate_yield_forecast(plant_age_months, farm_area_ha, climate_suitability, soil_suitability, 
                            fertilization_type, fertilization_frequency, pest_management_frequency,
                            bean_screen_size, overall_quality_index, forecast_years=5):
    """
    Calculate yield forecast for Robusta coffee per hectare over specified years
    Returns yield per year and grade distribution probabilities
    """
    # Base yield parameters for Robusta (kg/ha/year)
    base_yield_per_ha = 1200  # Average Robusta yield
    max_yield_per_ha = 2500   # Maximum achievable yield
    
    # Age factor - Robusta production curve
    if plant_age_months < 36:
        age_factor = 0  # Not yet producing
    elif plant_age_months < 48:
        age_factor = 0.5  # Young production
    elif plant_age_months < 72:
        age_factor = 0.8  # Growing production
    elif plant_age_months < 120:
        age_factor = 1.0  # Prime production
    elif plant_age_months < 180:
        age_factor = 0.9  # Mature production
    elif plant_age_months < 240:
        age_factor = 0.7  # Declining production
    else:
        age_factor = 0.5  # Old trees
    
    # Fertilization factor
    if fertilization_type == "Organic":
        fert_base = 0.85
    else:  # Non-organic
        fert_base = 1.0
    
    # Frequency multiplier (1=Never to 5=Always)
    fert_freq_multiplier = 0.7 + (fertilization_frequency * 0.075)  # 0.7 to 1.075
    fertilization_factor = fert_base * fert_freq_multiplier
    
    # Pest management factor (1=Never to 5=Always)
    pest_factor = 0.6 + (pest_management_frequency * 0.1)  # 0.6 to 1.0
    
    # Environmental factors
    climate_factor = climate_suitability
    soil_factor = soil_suitability
    quality_factor = overall_quality_index
    
    # Calculate yearly yields
    yearly_data = []
    for year in range(1, forecast_years + 1):
        # Age progression
        future_age_months = plant_age_months + (year * 12)
        
        # Recalculate age factor for future years
        if future_age_months < 36:
            future_age_factor = 0
        elif future_age_months < 48:
            future_age_factor = 0.5
        elif future_age_months < 72:
            future_age_factor = 0.8
        elif future_age_months < 120:
            future_age_factor = 1.0
        elif future_age_months < 180:
            future_age_factor = 0.9
        elif future_age_months < 240:
            future_age_factor = 0.7
        else:
            future_age_factor = 0.5
        
        # Calculate yield for this year
        year_yield = (base_yield_per_ha * 
                     future_age_factor * 
                     fertilization_factor * 
                     pest_factor * 
                     climate_factor * 
                     soil_factor * 
                     quality_factor)
        
        # Cap at maximum
        year_yield = min(year_yield, max_yield_per_ha)
        
        # Calculate grade probabilities based on management and conditions
        quality_score = (fertilization_factor * 0.3 + 
                        pest_factor * 0.3 + 
                        climate_factor * 0.2 + 
                        soil_factor * 0.2)
        
        # Grade distribution probabilities
        if quality_score >= 0.85:
            fine_prob = 0.6
            premium_prob = 0.35
            commercial_prob = 0.05
        elif quality_score >= 0.75:
            fine_prob = 0.4
            premium_prob = 0.45
            commercial_prob = 0.15
        elif quality_score >= 0.65:
            fine_prob = 0.2
            premium_prob = 0.5
            commercial_prob = 0.3
        else:
            fine_prob = 0.1
            premium_prob = 0.3
            commercial_prob = 0.6
        
        yearly_data.append({
            'Year': year,
            'Age (months)': future_age_months,
            'Yield (kg/ha)': round(year_yield, 2),
            'Total Yield (kg)': round(year_yield * farm_area_ha, 2),
            'Fine Probability': fine_prob,
            'Premium Probability': premium_prob,
            'Commercial Probability': commercial_prob,
            'Fine Yield (kg/ha)': round(year_yield * fine_prob, 2),
            'Premium Yield (kg/ha)': round(year_yield * premium_prob, 2),
            'Commercial Yield (kg/ha)': round(year_yield * commercial_prob, 2)
        })
    
    return pd.DataFrame(yearly_data)

# =====================================
# LOAD DATA
# =====================================

df = load_data()
df_engineered = engineer_features(df)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Analysis Module:",
    [
        "🏠 Home & Overview",
        "📊 Exploratory Data Analysis",
        "🎯 Grade Classification",
        "📈 Defect Analysis",
        "🔮 Grade Prediction Tool",
        "🌾 Harvest Yield Prediction",
        "📅 Yield & Grade Forecasting",
        "💡 PNS Standards & Guidelines"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Robusta Coffee Grading Platform**\n\n"
    f"📊 Total Records: {len(df_engineered):,}\n\n"
    f"🌱 Robusta Plants Only\n\n"
    "Based on PNS Standards"
)

# =====================================
# PAGE 1: HOME & OVERVIEW
# =====================================

if page == "🏠 Home & Overview":
    st.title("Robusta Coffee Grading Analytics Dashboard")
    st.markdown("### Philippine National Standards (PNS) Compliant System")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Samples",
            value=f"{len(df_engineered):,}",
            delta="Robusta Only"
        )
    
    with col2:
        fine_count = (df_engineered['coffee_grade'] == 'Fine').sum()
        st.metric(
            label="⭐ Fine Robusta",
            value=f"{fine_count:,}",
            delta=f"{(fine_count/len(df_engineered)*100):.1f}%"
        )
    
    with col3:
        premium_count = (df_engineered['coffee_grade'] == 'Premium').sum()
        st.metric(
            label="🥇 Premium Robusta",
            value=f"{premium_count:,}",
            delta=f"{(premium_count/len(df_engineered)*100):.1f}%"
        )
    
    with col4:
        avg_defects = df_engineered['total_defect_pct'].mean()
        st.metric(
            label="📉 Avg Defects",
            value=f"{avg_defects:.1f}%",
            delta="Total Defects"
        )
    
    st.markdown("---")
    
    # Grade distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Coffee Grade Distribution")
        grade_dist = df_engineered['coffee_grade'].value_counts()
        fig = px.pie(
            values=grade_dist.values,
            names=grade_dist.index,
            title='Fine / Premium / Commercial Distribution',
            color_discrete_map={'Fine': '#2c5f2d', 'Premium': '#97bc62', 'Commercial': '#d3d3d3'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Defect Distribution")
        
        # Average defects by grade
        defect_by_grade = df_engineered.groupby('coffee_grade')['total_defect_pct'].mean().sort_values()
        fig = px.bar(
            x=defect_by_grade.index,
            y=defect_by_grade.values,
            title='Average Defect % by Grade',
            labels={'x': 'Grade', 'y': 'Avg Defect (%)'},
            color=defect_by_grade.values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Overview sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Grading Standards")
        st.markdown("""
        **CQI/UCDA Fine Robusta Classification:**
        
        **⭐ Fine Robusta:**
        - 0 primary defects
        - ≤5 secondary defects
        - Bean size ≥6.5mm
        
        **🥇 Premium Robusta:**
        - ≤12 combined defects
        - Bean size ≥6.0mm
        
        **📦 Commercial:**
        - Does not meet Fine/Premium standards
        
        *Based on 350g sample*
        
        *Note: Cupping score optional when assessment facilities available*
        """)
    
    with col2:
        st.markdown("### 📋 Key Requirements")
        
        stats_df = pd.DataFrame({
            'Parameter': [
                'Elevation Range',
                'Temperature Range',
                'Rainfall (monthly)',
                'Soil pH Range',
                'Production Age',
                'Bean Screen Size'
            ],
            'Robusta Standard': [
                '600-1,200 masl',
                '13-26°C',
                '200 mm',
                '5.6-6.5',
                '36+ months',
                '≥5.6 mm (dry) / ≥7.5 mm (wet)'
            ]
        })
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Quick insights
    st.markdown("### 🔍 Quick Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fine_count = (df_engineered['coffee_grade'] == 'Fine').sum()
        st.success(
            f"**⭐ Fine Robusta:** {fine_count:,}\n\n"
            f"**Percentage:** {(fine_count/len(df_engineered)*100):.1f}%\n\n"
            f"**Highest Quality**"
        )
    
    with col2:
        large_beans = (df_engineered['bean_size_class'] == 'Large').sum()
        st.info(
            f"**📏 Large Beans:** {large_beans:,}\n\n"
            f"**Percentage:** {(large_beans/len(df_engineered)*100):.1f}%\n\n"
            f"**Screen Size ≥7.5mm**"
        )
    
    with col3:
        mature_plants = (df_engineered['plant_age_months'] >= 36).sum()
        st.warning(
            f"**🌱 Production Ready:** {mature_plants:,}\n\n"
            f"**Percentage:** {(mature_plants/len(df_engineered)*100):.1f}%\n\n"
            f"**Age ≥36 months**"
        )

# =====================================
# PAGE 2: EXPLORATORY DATA ANALYSIS
# =====================================

elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "📋 Data Overview"])
    
    with tab1:
        st.markdown("### Key Variable Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df_engineered, x='total_defect_pct', nbins=50,
                             title='Total Defect Percentage Distribution',
                             labels={'total_defect_pct': 'Defect (%)'},
                             color_discrete_sequence=['#d62728'])
            fig.add_vline(x=10, line_dash="dash", line_color="green", annotation_text="Grade 1")
            fig.add_vline(x=15, line_dash="dash", line_color="blue", annotation_text="Grade 2")
            fig.add_vline(x=25, line_dash="dash", line_color="orange", annotation_text="Grade 3")
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.histogram(df_engineered, x='plant_age_months', nbins=50,
                             title='Plant Age Distribution',
                             labels={'plant_age_months': 'Age (months)'},
                             color_discrete_sequence=['#ff7f0e'])
            fig.add_vline(x=36, line_dash="dash", line_color="red", annotation_text="Production Age")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df_engineered, x='bean_screen_size_mm', nbins=30,
                             title='Bean Screen Size Distribution',
                             labels={'bean_screen_size_mm': 'Screen Size (mm)'},
                             color_discrete_sequence=['#2c5f2d'])
            fig.add_vline(x=7.5, line_dash="dash", line_color="green", annotation_text="Large")
            fig.add_vline(x=6.5, line_dash="dash", line_color="blue", annotation_text="Medium")
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.histogram(df_engineered, x='monthly_temp_avg_c', nbins=40,
                             title='Temperature Distribution',
                             labels={'monthly_temp_avg_c': 'Temperature (°C)'},
                             color_discrete_sequence=['#ff6347'])
            fig.add_vline(x=13, line_dash="dash", line_color="blue", annotation_text="Min")
            fig.add_vline(x=26, line_dash="dash", line_color="red", annotation_text="Max")
            st.plotly_chart(fig, use_container_width=True)
        
        # Coffee grade by age
        st.markdown("### Coffee Grade by Plant Age")
        fig = px.box(df_engineered, x='age_category', y='total_defect_pct',
                    color='coffee_grade',
                    title='Defect Percentage by Age Category and Grade',
                    labels={'total_defect_pct': 'Defect (%)', 'age_category': 'Age Category'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Correlation Analysis")
        
        key_vars = [
            'plant_age_months', 'bean_screen_size_mm', 'total_defect_pct',
            'primary_defects', 'secondary_defects', 'monthly_temp_avg_c',
            'monthly_rainfall_mm', 'soil_pH', 'soil_moisture_pct',
            'climate_suitability_robusta', 'overall_quality_index'
        ]
        
        corr_matrix = df_engineered[key_vars].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Matrix - Grading Features",
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations with defects
        st.markdown("### Top Correlations with Total Defects")
        defect_corr = corr_matrix['total_defect_pct'].abs().sort_values(ascending=False)[1:11]
        
        fig = px.bar(x=defect_corr.values, y=defect_corr.index, orientation='h',
                    labels={'x': 'Absolute Correlation', 'y': 'Feature'},
                    color=defect_corr.values,
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Dataset Statistics")
            st.dataframe(df_engineered[key_vars].describe(), use_container_width=True)
        
        with col2:
            st.markdown("#### Grade Distribution Summary")
            summary_df = pd.DataFrame({
                'Coffee Grade': df_engineered['coffee_grade'].value_counts().index,
                'Count': df_engineered['coffee_grade'].value_counts().values,
                'Percentage': (df_engineered['coffee_grade'].value_counts().values / len(df_engineered) * 100).round(2)
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            st.markdown("#### PNS Grade Summary")
            pns_summary = pd.DataFrame({
                'PNS Grade': [f"Grade {g}" for g in df_engineered['pns_grade'].value_counts().sort_index().index],
                'Count': df_engineered['pns_grade'].value_counts().sort_index().values,
                'Percentage': (df_engineered['pns_grade'].value_counts().sort_index().values / len(df_engineered) * 100).round(2)
            })
            st.dataframe(pns_summary, use_container_width=True, hide_index=True)
        
        st.markdown("#### Sample Data")
        display_cols = ['plant_age_months', 'bean_screen_size_mm', 'total_defect_pct',
                       'primary_defects', 'secondary_defects', 'coffee_grade', 'pns_grade',
                       'monthly_temp_avg_c', 'monthly_rainfall_mm', 'soil_pH']
        st.dataframe(df_engineered[display_cols].head(20), use_container_width=True)

# =====================================
# PAGE 3: GRADE CLASSIFICATION
# =====================================

elif page == "🎯 Grade Classification":
    st.title("🎯 Coffee Grade Classification")
    
    st.markdown("### CQI/UCDA Fine Robusta Classification")
    st.info("""
    **Fine Robusta:** 0 primary defects, ≤5 secondary defects, bean size ≥6.5mm
    
    **Premium Robusta:** ≤12 combined defects, bean size ≥6.0mm
    
    **Commercial:** All others
    
    *Note: Cupping score optional when assessment facilities available*
    """)
    
    if st.button("🚀 Train Classification Model", key='grade_class'):
        with st.spinner("Training classification models..."):
            # Prepare features for grading
            feature_cols = [
                'plant_age_months', 'bean_screen_size_mm',
                'monthly_temp_avg_c', 'monthly_rainfall_mm',
                'soil_pH', 'soil_moisture_pct',
                'climate_suitability_robusta', 'overall_quality_index',
                'environmental_stress_index'
            ]
            
            X = df_engineered[feature_cols]
            y = df_engineered['coffee_grade']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
                'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
            }
            
            results = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred
                }
            
            # Display results
            st.success("✅ Models trained successfully!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Performance")
                perf_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Accuracy': [results[m]['accuracy'] for m in results.keys()]
                })
                st.dataframe(perf_df, use_container_width=True, hide_index=True)
            
            with col2:
                fig = px.bar(perf_df, x='Model', y='Accuracy',
                           title='Model Comparison',
                           color='Accuracy',
                           color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
            
            # Best model analysis
            best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            best_pred = results[best_model_name]['predictions']
            
            st.markdown(f"#### Best Model: {best_model_name}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, best_pred)
            labels = sorted(y.unique())
            fig = plot_confusion_matrix(cm, labels, f"Confusion Matrix - {best_model_name}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification report
            st.markdown("#### Detailed Classification Report")
            report = classification_report(y_test, best_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # Feature importance
            if hasattr(results[best_model_name]['model'], 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': results[best_model_name]['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = plot_feature_importance(importance_df, top_n=9)
                st.plotly_chart(fig, use_container_width=True)

# =====================================
# PAGE 4: DEFECT ANALYSIS
# =====================================

elif page == "📈 Defect Analysis":
    st.title("📈 Defect Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Defect Distribution", "🔍 Factor Analysis", "📉 Defect Prediction"])
    
    with tab1:
        st.markdown("### Defect Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Primary vs Secondary defects
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df_engineered['primary_defects'], name='Primary Defects',
                                      marker_color='red', opacity=0.7))
            fig.add_trace(go.Histogram(x=df_engineered['secondary_defects'], name='Secondary Defects',
                                      marker_color='orange', opacity=0.7))
            fig.update_layout(title='Primary vs Secondary Defects Distribution',
                            barmode='overlay', xaxis_title='Number of Defects')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Defect percentage by grade
            grade_defects = df_engineered.groupby('coffee_grade')['total_defect_pct'].mean().sort_values()
            fig = px.bar(x=grade_defects.index, y=grade_defects.values,
                        title='Average Defect % by Coffee Grade',
                        labels={'x': 'Grade', 'y': 'Avg Defect (%)'},
                        color=grade_defects.values,
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        
        # Defects by bean size
        st.markdown("### Defects by Bean Screen Size")
        fig = px.box(df_engineered, x='bean_size_class', y='total_defect_pct',
                    color='bean_size_class',
                    title='Defect Distribution by Bean Size Class',
                    labels={'total_defect_pct': 'Defect (%)', 'bean_size_class': 'Bean Size'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Defects by plant age
        st.markdown("### Defects by Plant Age")
        fig = px.scatter(df_engineered, x='plant_age_months', y='total_defect_pct',
                        color='coffee_grade', size='bean_screen_size_mm',
                        title='Plant Age vs Defects',
                        labels={'plant_age_months': 'Age (months)', 'total_defect_pct': 'Defect (%)'})
        fig.add_hline(y=10, line_dash="dash", line_color="green", annotation_text="Grade 1 Limit")
        fig.add_hline(y=15, line_dash="dash", line_color="blue", annotation_text="Grade 2 Limit")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Environmental & Soil Factors Impact on Defects")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature impact
            fig = px.scatter(df_engineered, x='monthly_temp_avg_c', y='total_defect_pct',
                           color='coffee_grade',
                           title='Temperature vs Defects',
                           labels={'monthly_temp_avg_c': 'Temperature (°C)', 'total_defect_pct': 'Defect (%)'})
            fig.add_vline(x=13, line_dash="dash", line_color="blue", annotation_text="Min Optimal")
            fig.add_vline(x=26, line_dash="dash", line_color="red", annotation_text="Max Optimal")
            st.plotly_chart(fig, use_container_width=True)
            
            # Soil pH impact
            fig = px.scatter(df_engineered, x='soil_pH', y='total_defect_pct',
                           color='coffee_grade',
                           title='Soil pH vs Defects',
                           labels={'soil_pH': 'Soil pH', 'total_defect_pct': 'Defect (%)'})
            fig.add_vline(x=5.6, line_dash="dash", line_color="green", annotation_text="Min Optimal")
            fig.add_vline(x=6.5, line_dash="dash", line_color="green", annotation_text="Max Optimal")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rainfall impact
            fig = px.scatter(df_engineered, x='monthly_rainfall_mm', y='total_defect_pct',
                           color='coffee_grade',
                           title='Rainfall vs Defects',
                           labels={'monthly_rainfall_mm': 'Rainfall (mm)', 'total_defect_pct': 'Defect (%)'})
            fig.add_vline(x=200, line_dash="dash", line_color="green", annotation_text="Optimal")
            st.plotly_chart(fig, use_container_width=True)
            
            # Soil moisture impact
            fig = px.scatter(df_engineered, x='soil_moisture_pct', y='total_defect_pct',
                           color='coffee_grade',
                           title='Soil Moisture vs Defects',
                           labels={'soil_moisture_pct': 'Soil Moisture (%)', 'total_defect_pct': 'Defect (%)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Climate suitability impact
        st.markdown("### Climate & Soil Suitability Impact")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df_engineered, x='climate_suitability_robusta', y='total_defect_pct',
                           color='coffee_grade', size='bean_screen_size_mm',
                           title='Climate Suitability vs Defects',
                           labels={'climate_suitability_robusta': 'Climate Suitability', 'total_defect_pct': 'Defect (%)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df_engineered, x='overall_quality_index', y='total_defect_pct',
                           color='coffee_grade', size='bean_screen_size_mm',
                           title='Overall Quality Index vs Defects',
                           labels={'overall_quality_index': 'Quality Index', 'total_defect_pct': 'Defect (%)'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Defect Percentage Prediction")
        st.info("Predict total defect percentage based on environmental and plant factors")
        
        if st.button("🚀 Train Defect Prediction Model", key='defect_pred'):
            with st.spinner("Training regression model..."):
                feature_cols = [
                    'plant_age_months', 'bean_screen_size_mm',
                    'monthly_temp_avg_c', 'monthly_rainfall_mm',
                    'soil_pH', 'soil_moisture_pct',
                    'environmental_stress_index',
                    'climate_suitability_robusta',
                    'soil_suitability_robusta',
                    'overall_quality_index'
                ]
                
                X = df_engineered[feature_cols]
                y = df_engineered['total_defect_pct']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train models
                models = {
                    'Random Forest': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
                    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=42)
                }
                
                results = {}
                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[name] = {
                        'model': model,
                        'predictions': y_pred,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2
                    }
                
                st.success("✅ Models trained successfully!")
                
                # Performance comparison
                st.markdown("#### Model Performance Comparison")
                perf_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'RMSE': [results[m]['rmse'] for m in results.keys()],
                    'MAE': [results[m]['mae'] for m in results.keys()],
                    'R²': [results[m]['r2'] for m in results.keys()]
                })
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(perf_df, use_container_width=True, hide_index=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='R²',
                        x=perf_df['Model'],
                        y=perf_df['R²'],
                        marker_color='#2c5f2d'
                    ))
                    fig.update_layout(title='Model R² Comparison', height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Best model visualization
                best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
                best_pred = results[best_model_name]['predictions']
                
                st.markdown(f"#### Best Model: {best_model_name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{results[best_model_name]['r2']:.4f}")
                with col2:
                    st.metric("RMSE", f"{results[best_model_name]['rmse']:.4f}%")
                with col3:
                    st.metric("MAE", f"{results[best_model_name]['mae']:.4f}%")
                
                # Actual vs Predicted
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Actual vs Predicted Defects', 'Residual Plot')
                )
                
                fig.add_trace(
                    go.Scatter(x=y_test, y=best_pred, mode='markers',
                              marker=dict(color='#2c5f2d', opacity=0.5),
                              name='Predictions'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=[y_test.min(), y_test.max()],
                              y=[y_test.min(), y_test.max()],
                              mode='lines',
                              line=dict(color='red', dash='dash'),
                              name='Perfect Fit'),
                    row=1, col=1
                )
                
                residuals = y_test - best_pred
                fig.add_trace(
                    go.Scatter(x=best_pred, y=residuals, mode='markers',
                              marker=dict(color='#1f77b4', opacity=0.5),
                              name='Residuals'),
                    row=1, col=2
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
                
                fig.update_xaxes(title_text="Actual Defect (%)", row=1, col=1)
                fig.update_yaxes(title_text="Predicted Defect (%)", row=1, col=1)
                fig.update_xaxes(title_text="Predicted Defect (%)", row=1, col=2)
                fig.update_yaxes(title_text="Residuals", row=1, col=2)
                
                fig.update_layout(height=500, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                if hasattr(results[best_model_name]['model'], 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': results[best_model_name]['model'].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = plot_feature_importance(importance_df, top_n=10)
                    st.plotly_chart(fig, use_container_width=True)

# =====================================
# PAGE 5: GRADE PREDICTION TOOL (UPDATED - NO CUPPING SCORE)
# =====================================

elif page == "🔮 Grade Prediction Tool":
    st.title("🔮 Interactive Grade Prediction Tool")
    st.markdown("### Input parameters to predict coffee grade")
    
    st.info("Adjust the parameters below to see real-time predictions for Robusta coffee grade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌱 Plant & Bean Characteristics")
        plant_age = st.number_input("Plant Age (months)", min_value=0, max_value=300, value=48, step=1,
                                    help="Robusta production starts at 36 months, optimal lifespan 20-25 years (240-300 months)")
        bean_screen = st.slider("Bean Screen Size (mm)", min_value=4.0, max_value=9.0, value=6.5, step=0.1,
                               help="Standard sizes: >7.0, >6.5-7.0, >6.0-6.5, >5.5-6.0, >5.0-5.5, 4.0-5.0 mm")
        
        st.markdown("#### 🔴 Defect Counts")
        primary_defects = st.number_input("Primary Defects", min_value=0, max_value=50, value=0, step=1,
                                         help="Category 1 defects (black, moldy, sour, insect-damaged beans)")
        secondary_defects = st.number_input("Secondary Defects", min_value=0, max_value=50, value=3, step=1,
                                           help="Category 2 defects (broken, immature, faded beans)")
        
        st.markdown("#### 🌦️ Climate Conditions")
        elevation = st.slider("Elevation (masl)", 0, 3000, 900, 50,
                            help="Optimal range for Robusta: 600-1,200 masl (meters above sea level)")
        temp_avg = st.slider("Average Temperature (°C)", -10.0, 50.0, 19.5, 0.5,
                            help="Optimal range for Robusta: 13-26°C")
        rainfall = st.slider("Monthly Rainfall (mm)", 0, 1000, 200, 10,
                            help="Optimal for Robusta: 200mm/month, can range 0-1000mm")
        humidity = st.slider("Relative Humidity (%)", 0, 100, 75, 5,
                            help="Optimal range for Robusta: 75-85%")
    
    with col2:
        st.markdown("#### 🌿 Soil Properties")
        soil_ph = st.slider("Soil pH", 0.0, 14.0, 6.0, 0.1,
                           help="Standard pH scale 0-14. Optimal for Robusta: 5.6-6.5")
        soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 25, 5,
                                 help="Optimal soil moisture for coffee cultivation")
        
        st.markdown("#### 📊 Calculated Indices")
        st.info("These are automatically calculated based on your inputs")
        
        # Calculate total defects percentage
        # Assume 350g sample, average bean weight ~0.15g per bean
        total_beans_sample = 350 / 0.15  # approximately 2333 beans
        total_defect_count = primary_defects + secondary_defects
        total_defect_pct = (total_defect_count / total_beans_sample) * 100
        
        st.metric("Total Defect %", f"{total_defect_pct:.2f}%")
        
        # Calculate derived indices
        elevation_score = 1 - abs(elevation - 900) / 600
        elevation_score = max(0, min(1, elevation_score))
        
        temp_score = 1 - abs(temp_avg - 19.5) / 13
        temp_score = max(0, min(1, temp_score))
        
        rainfall_score = min(rainfall / 200, 1.5)
        rainfall_score = max(0, min(1, rainfall_score))
        
        climate_suitability = (temp_score * 0.3 + rainfall_score * 0.3 + elevation_score * 0.4)
        
        soil_suitability = 1 - abs(soil_ph - 6.0) / 1.5
        soil_suitability = max(0, min(1, soil_suitability))
        
        moisture_suitability = soil_moisture / 35
        moisture_suitability = max(0, min(1, moisture_suitability))
        
        # Calculate environmental stress based on deviations from optimal
        temp_stress = abs(temp_avg - 19.5) / 13
        rainfall_stress = abs(rainfall - 200) / 200
        ph_stress = abs(soil_ph - 6.0) / 1.5
        elevation_stress = abs(elevation - 900) / 300
        env_stress = (temp_stress + rainfall_stress + ph_stress + elevation_stress) / 4
        env_stress = max(0, min(1, env_stress))
        
        overall_quality = (
            climate_suitability * 0.3 +
            soil_suitability * 0.3 +
            moisture_suitability * 0.2 +
            (1 - env_stress) * 0.2
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Climate Suitability", f"{climate_suitability:.2f}")
            st.metric("Soil Suitability", f"{soil_suitability:.2f}")
            st.metric("Elevation Score", f"{elevation_score:.2f}")
        with col_b:
            st.metric("Overall Quality Index", f"{overall_quality:.2f}")
            st.metric("Bean Size Class", classify_bean_size(bean_screen))
            st.metric("Elevation Category", 
                     "Optimal" if 600 <= elevation <= 1200 else "Sub-optimal")
    
    st.markdown("---")
    
    if st.button("🔮 Predict Coffee Grade", key='predict_grade', type='primary'):
        with st.spinner("Calculating predictions..."):
            # UPDATED: Determine grade based on bean size and defects only (NO cupping score)
            predicted_grade = calculate_fine_premium_grade(primary_defects, secondary_defects, bean_screen)
            
            # Calculate PNS grade for reference
            pns_grade = calculate_pns_grade(total_defect_pct)
            
            # Bean size class
            bean_size_class = classify_bean_size(bean_screen)
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                grade_color = "green" if predicted_grade == "Fine" else "blue" if predicted_grade == "Premium" else "orange"
                st.markdown(f"### :{grade_color}[{predicted_grade} Robusta]")
                st.metric(
                    label="Coffee Grade",
                    value=predicted_grade,
                    delta=f"Bean: {bean_screen:.1f}mm"
                )
            
            with col2:
                st.metric(
                    label="📊 PNS Reference",
                    value=f"Grade {pns_grade}",
                    delta=f"Defects: {total_defect_pct:.2f}%",
                    help="Philippine National Standard grade for reference"
                )
            
            with col3:
                st.metric(
                    label="📏 Bean Size",
                    value=bean_size_class,
                    delta=f"{bean_screen:.1f}mm"
                )
            
            # Detailed defect breakdown
            st.markdown("### 📉 Defect Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Primary Defects", f"{primary_defects}")
            with col2:
                st.metric("Secondary Defects", f"{secondary_defects}")
            with col3:
                st.metric("Total Defects", f"{total_defect_count}")
            with col4:
                st.metric("Defect %", f"{total_defect_pct:.2f}%")
            
            # Grade requirements visualization
            st.markdown("### 📊 Grade Requirements Comparison")
            
            requirements = pd.DataFrame({
                'Grade': ['Fine', 'Premium', 'Commercial', 'Your Sample'],
                'Primary Defects': [0, '0-12*', '>12*', primary_defects],
                'Secondary Defects': ['≤5', '0-12*', '>12*', secondary_defects],
                'Combined Defects': ['≤5', '≤12', '>12', total_defect_count],
                'Bean Size (mm)': ['≥6.5', '≥6.0', '<6.0', f"{bean_screen:.1f}"]
            })
            
            st.dataframe(requirements, use_container_width=True, hide_index=True)
            st.caption("*Combined primary and secondary defects. Note: Cupping score optional when facilities available.")
            
            # Gauge charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Defect gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=total_defect_count,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Total Defect Count"},
                    delta={'reference': 12, 'decreasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [None, 50]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 5], 'color': "#90ee90"},
                            {'range': [5, 12], 'color': "#add8e6"},
                            {'range': [12, 25], 'color': "#ffffcc"},
                            {'range': [25, 50], 'color': "#ffcccb"}
                        ],
                        'threshold': {
                            'line': {'color': "green", 'width': 4},
                            'thickness': 0.75,
                            'value': 5
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bean size gauge (UPDATED - replaces cupping score gauge)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=bean_screen,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Bean Screen Size (mm)"},
                    gauge={
                        'axis': {'range': [4, 9]},
                        'bar': {'color': "#2c5f2d"},
                        'steps': [
                            {'range': [4, 5.5], 'color': "#ffcccb"},
                            {'range': [5.5, 6.0], 'color': "#ffffcc"},
                            {'range': [6.0, 6.5], 'color': "#add8e6"},
                            {'range': [6.5, 7.5], 'color': "#90ee90"},
                            {'range': [7.5, 9], 'color': "#228b22"}
                        ],
                        'threshold': {
                            'line': {'color': "green", 'width': 4},
                            'thickness': 0.75,
                            'value': 6.5
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Personalized Recommendations")
            
            if predicted_grade == 'Commercial':
                st.error("⚠️ **Coffee graded as Commercial** - Below Fine/Premium standards")
                st.markdown("""
                **Critical Improvement Actions:**
                - **Reduce defects** through better harvesting (selective picking only)
                - **Improve processing**: Proper fermentation (18-24hrs), clean water, timely drying
                - **Better sorting**: Remove all defective beans before final processing
                - **Increase bean size**: Better nutrition and water management
                - **Quality control**: Regular inspection and grading throughout process
                """)
            
            if primary_defects > 0:
                st.error("🔴 **Primary defects detected!** These are critical quality issues.")
                st.markdown("""
                **Actions to eliminate primary defects:**
                - Check for mold during storage (control humidity <60%)
                - Prevent over-fermentation (max 24 hours)
                - Avoid harvesting overripe or ground cherries
                - Implement pest control for coffee berry borer
                """)
            
            if secondary_defects > 5:
                st.warning("🟡 **High secondary defects** - Exceeds Fine Robusta standards")
                st.markdown("""
                **Actions to reduce secondary defects:**
                - Harvest only ripe cherries (avoid immature/green)
                - Careful handling to prevent breakage
                - Proper drying (avoid over/under drying)
                - Improve soil fertility and plant nutrition
                """)
            
            if bean_screen < 6.5:
                st.warning("📏 **Bean size below Fine Robusta standard (6.5mm)**")
                st.markdown("""
                **Actions to increase bean size:**
                - Apply complete fertilizer (14-14-14) regularly
                - Ensure adequate water during cherry development
                - Proper spacing (3m x 2m) for better growth
                - Prune to reduce excessive fruiting load
                """)
            
            if temp_avg < 13 or temp_avg > 26:
                st.warning(f"🌡️ **Temperature ({temp_avg}°C) outside optimal range (13-26°C)**")
                st.markdown("- Implement shade management (30-40% coverage)")
                st.markdown("- Consider windbreaks for temperature moderation")
            
            if elevation < 600:
                st.warning(f"⛰️ **Elevation ({elevation}m) below optimal range (600-1,200 masl)**")
                st.markdown("- Robusta performs best at 600-1,200 masl")
                st.markdown("- Lower elevations may result in lower quality beans")
                st.markdown("- Consider improved agronomic practices to compensate")
            elif elevation > 1200:
                st.warning(f"⛰️ **Elevation ({elevation}m) above optimal range (600-1,200 masl)**")
                st.markdown("- Robusta may experience stress at higher elevations")
                st.markdown("- Consider switching to Arabica for elevations >900 masl")
                st.markdown("- Implement cold protection measures if needed")
            
            if rainfall < 150:
                st.warning(f"💧 **Low rainfall ({rainfall}mm) - Below optimal 200mm**")
                st.markdown("- Implement drip irrigation during dry periods")
                st.markdown("- Apply mulching to retain soil moisture")
            
            if soil_ph < 5.6:
                st.warning(f"🌿 **Soil pH ({soil_ph}) too acidic**")
                st.markdown("- Apply agricultural lime to increase pH to 5.6-6.5 range")
            elif soil_ph > 6.5:
                st.warning(f"🌿 **Soil pH ({soil_ph}) too alkaline**")
                st.markdown("- Apply sulfur or organic matter to decrease pH")
            
            if plant_age < 36:
                st.info("⏳ **Plant not yet mature** - Robusta production starts at 36 months")
                st.markdown("- Continue vegetative growth management")
                st.markdown("- Focus on pruning and desuckering")
            
            if predicted_grade in ['Fine', 'Premium']:
                st.success(f"✅ **Excellent! Meets {predicted_grade} Robusta standards**")
                st.markdown("""
                **Maintain quality by:**
                - Continuing current best practices
                - Regular monitoring of all parameters
                - Consistent quality control procedures
                - Proper post-harvest handling and storage
                """)
                
                if predicted_grade == 'Fine':
                    st.balloons()
                    st.success("🏆 **Premium Market Ready!** Your coffee qualifies for specialty markets.")

# =====================================
# PAGE 6: HARVEST YIELD PREDICTION (NEW MODULE)
# =====================================

elif page == "🌾 Harvest Yield Prediction":
    st.title("🌾 Harvest Yield Prediction")
    st.markdown("### Machine Learning-Based Seasonal Harvest Forecasting")
    
    st.info("""
    **Harvest Season for Robusta:** November, December, January, February, March
    
    This module predicts harvest yield based on:
    - Plant age, elevation, temperature, rainfall
    - Soil pH and moisture
    - Fertilization and pest management frequency
    - Seasonal patterns (month-specific)
    """)
    
    # Check if harvest data is available
    harvest_columns = ['nov_yield', 'dec_yield', 'jan_yield', 'feb_yield', 'mar_yield']
    has_harvest_data = any(col in df.columns for col in harvest_columns)
    
    if not has_harvest_data:
        st.warning("""
        ⚠️ **Harvest yield data not found in dataset**
        
        Expected columns: `nov_yield`, `dec_yield`, `jan_yield`, `feb_yield`, `mar_yield`
        
        This module requires actual harvest data from November through March to train prediction models.
        """)
        
        st.markdown("### 📋 Dataset Requirements")
        st.markdown("""
        Please ensure your dataset includes:
        - `nov_yield`: November harvest (kg)
        - `dec_yield`: December harvest (kg)  
        - `jan_yield`: January harvest (kg)
        - `feb_yield`: February harvest (kg)
        - `mar_yield`: March harvest (kg)
        - `elevation_masl`: Elevation (meters above sea level)
        - `monthly_temp_avg_c`: Average temperature (°C)
        - `monthly_rainfall_mm`: Monthly rainfall (mm)
        - `soil_pH`: Soil pH level
        - `soil_moisture_pct`: Soil moisture (%)
        - `fertilization_frequency`: Fertilization frequency (1-5 scale)
        - `pest_management_frequency`: Pest management frequency (1-5 scale)
        - `plant_age_months`: Plant age in months
        """)
        
        st.stop()
    
    # Prepare harvest data
    with st.spinner("Preparing harvest data..."):
        harvest_df = prepare_harvest_yield_data(df)
    
    if len(harvest_df) == 0:
        st.error("❌ No valid harvest records found. Please check your data.")
        st.stop()
    
    st.success(f"✅ Loaded {len(harvest_df):,} harvest records across {len(df):,} plants")
    
    # Data overview
    st.markdown("### 📊 Harvest Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(harvest_df):,}")
    with col2:
        st.metric("Avg Yield (kg)", f"{harvest_df['harvest_yield_kg'].mean():.2f}")
    with col3:
        st.metric("Max Yield (kg)", f"{harvest_df['harvest_yield_kg'].max():.2f}")
    with col4:
        st.metric("Min Yield (kg)", f"{harvest_df['harvest_yield_kg'].min():.2f}")
    
    # Monthly distribution
    st.markdown("### 📅 Harvest Distribution by Month")
    
    month_names = {11: 'November', 12: 'December', 1: 'January', 2: 'February', 3: 'March'}
    harvest_df['month_name'] = harvest_df['month'].map(month_names)
    
    monthly_stats = harvest_df.groupby('month_name')['harvest_yield_kg'].agg(['count', 'mean', 'std', 'min', 'max'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=monthly_stats.index,
            y=monthly_stats['mean'],
            error_y=monthly_stats['std'],
            title='Average Harvest Yield by Month',
            labels={'x': 'Month', 'y': 'Average Yield (kg)'},
            color=monthly_stats['mean'],
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            harvest_df,
            x='month_name',
            y='harvest_yield_kg',
            title='Harvest Yield Distribution by Month',
            labels={'month_name': 'Month', 'harvest_yield_kg': 'Yield (kg)'},
            color='month_name'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(monthly_stats, use_container_width=True)
    
    # Model Training Section
    st.markdown("---")
    st.markdown("## 🤖 Train Prediction Models")
    
    if st.button("🚀 Train Harvest Yield Models", key='train_harvest', type='primary'):
        with st.spinner("Training Random Forest and Gradient Boosting models..."):
            
            # Feature selection
            feature_cols = [
                'plant_age_months', 'elevation_masl',
                'monthly_temp_avg_c', 'monthly_rainfall_mm',
                'soil_pH', 'soil_moisture_pct',
                'fertilization_freq', 'pest_management_freq',
                'climate_suitability', 'soil_suitability',
                'moisture_suitability', 'production_ready',
                'month_sin', 'month_cos'  # Cyclical encoding for seasonality
            ]
            
            X = harvest_df[feature_cols]
            y = harvest_df['harvest_yield_kg']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            models = {
                'Random Forest': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            }
            
            results = {}
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
            
            st.success("✅ Models trained successfully!")
            
            # Performance comparison
            st.markdown("### 📊 Model Performance Comparison")
            
            perf_df = pd.DataFrame({
                'Model': list(results.keys()),
                'RMSE (kg)': [f"{results[m]['rmse']:.3f}" for m in results.keys()],
                'MAE (kg)': [f"{results[m]['mae']:.3f}" for m in results.keys()],
                'R² Score': [f"{results[m]['r2']:.4f}" for m in results.keys()]
            })
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(perf_df, use_container_width=True, hide_index=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='R² Score',
                    x=[results[m]['r2'] for m in results.keys()],
                    y=list(results.keys()),
                    orientation='h',
                    marker_color=['#2c5f2d', '#97bc62']
                ))
                fig.update_layout(
                    title='Model R² Comparison',
                    xaxis_title='R² Score',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Best model analysis
            best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
            best_pred = results[best_model_name]['predictions']
            
            st.markdown(f"### 🏆 Best Model: {best_model_name}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{results[best_model_name]['r2']:.4f}")
            with col2:
                st.metric("RMSE", f"{results[best_model_name]['rmse']:.3f} kg")
            with col3:
                st.metric("MAE", f"{results[best_model_name]['mae']:.3f} kg")
            
            # Visualizations
            st.markdown("### 📈 Model Performance Visualizations")
            
            # Actual vs Predicted
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Actual vs Predicted Yield', 'Residual Plot')
            )
            
            fig.add_trace(
                go.Scatter(
                    x=y_test,
                    y=best_pred,
                    mode='markers',
                    marker=dict(color='#2c5f2d', opacity=0.6, size=8),
                    name='Predictions'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    name='Perfect Fit'
                ),
                row=1, col=1
            )
            
            residuals = y_test - best_pred
            fig.add_trace(
                go.Scatter(
                    x=best_pred,
                    y=residuals,
                    mode='markers',
                    marker=dict(color='#1f77b4', opacity=0.6, size=8),
                    name='Residuals'
                ),
                row=1, col=2
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
            
            fig.update_xaxes(title_text="Actual Yield (kg)", row=1, col=1)
            fig.update_yaxes(title_text="Predicted Yield (kg)", row=1, col=1)
            fig.update_xaxes(title_text="Predicted Yield (kg)", row=1, col=2)
            fig.update_yaxes(title_text="Residuals (kg)", row=1, col=2)
            
            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if hasattr(results[best_model_name]['model'], 'feature_importances_'):
                st.markdown("### 🔍 Feature Importance Analysis")
                
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': results[best_model_name]['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = plot_feature_importance(importance_df, top_n=min(14, len(feature_cols)))
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Top 5 Most Important Features")
                top5 = importance_df.head(5)
                for idx, row in top5.iterrows():
                    st.markdown(f"**{idx+1}. {row['Feature']}**: {row['Importance']:.4f}")
            
            # Prediction by month
            st.markdown("### 📅 Predictions by Harvest Month")
            
            # Create month mapping for predictions
            test_months = []
            for idx in X_test.index:
                month_val = harvest_df.loc[idx, 'month']
                test_months.append(month_val)
            
            pred_by_month = pd.DataFrame({
                'month': test_months,
                'actual': y_test.values,
                'predicted': best_pred
            })
            
            pred_by_month['month_name'] = pred_by_month['month'].map(month_names)
            
            monthly_comparison = pred_by_month.groupby('month_name')[['actual', 'predicted']].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly_comparison.index,
                y=monthly_comparison['actual'],
                name='Actual',
                marker_color='#2c5f2d'
            ))
            fig.add_trace(go.Bar(
                x=monthly_comparison.index,
                y=monthly_comparison['predicted'],
                name='Predicted',
                marker_color='#97bc62'
            ))
            
            fig.update_layout(
                title='Actual vs Predicted Yield by Month',
                xaxis_title='Month',
                yaxis_title='Average Yield (kg)',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Save model option
            st.markdown("---")
            st.markdown("### 💾 Save Model")
            
            if st.button("Download Best Model", key='download_model'):
                # Serialize model and scaler
                model_data = {
                    'model': results[best_model_name]['model'],
                    'scaler': scaler,
                    'feature_cols': feature_cols,
                    'model_name': best_model_name,
                    'performance': {
                        'r2': results[best_model_name]['r2'],
                        'rmse': results[best_model_name]['rmse'],
                        'mae': results[best_model_name]['mae']
                    }
                }
                
                buffer = io.BytesIO()
                pickle.dump(model_data, buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Model File",
                    data=buffer,
                    file_name=f"harvest_yield_model_{best_model_name.lower().replace(' ', '_')}.pkl",
                    mime="application/octet-stream"
                )
                
                st.success("✅ Model ready for download!")

# =====================================
# PAGE 7: YIELD & GRADE FORECASTING
# =====================================

elif page == "📅 Yield & Grade Forecasting":
    st.title("📅 Yield & Grade Forecasting")
    st.markdown("### Predict Future Coffee Yield and Grade Distribution per Hectare")
    
    st.info("This tool forecasts coffee yield and grade classifications based on management practices and environmental conditions.")
    
    # Input Section
    st.markdown("---")
    st.markdown("## 📝 Farm & Management Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌱 Farm Characteristics")
        farm_area_ha = st.number_input(
            "Farm Area (hectares)", 
            min_value=0.1, 
            max_value=1000.0, 
            value=1.0, 
            step=0.1,
            help="Total farm area for yield calculation"
        )
        
        plant_age_forecast = st.number_input(
            "Current Plant Age (months)", 
            min_value=0, 
            max_value=300, 
            value=48, 
            step=1,
            help="Current age of coffee plants. Production starts at 36 months."
        )
        
        forecast_years = st.slider(
            "Forecast Period (years)", 
            min_value=1, 
            max_value=10, 
            value=5, 
            step=1,
            help="Number of years to forecast"
        )
        
        st.markdown("#### 🌦️ Environmental Conditions")
        elevation_forecast = st.slider(
            "Elevation (masl)", 
            min_value=0, 
            max_value=3000, 
            value=900, 
            step=50,
            help="Optimal range for Robusta: 600-1,200 masl (meters above sea level)"
        )
        
        temp_avg_forecast = st.slider(
            "Average Temperature (°C)", 
            min_value=-10.0, 
            max_value=50.0, 
            value=19.5, 
            step=0.5,
            help="Optimal range for Robusta: 13-26°C"
        )
        
        rainfall_forecast = st.slider(
            "Monthly Rainfall (mm)", 
            min_value=0, 
            max_value=1000, 
            value=200, 
            step=10,
            help="Optimal for Robusta: 200mm/month"
        )
        
        soil_ph_forecast = st.slider(
            "Soil pH", 
            min_value=4.0, 
            max_value=8.0, 
            value=6.0, 
            step=0.1,
            help="Optimal for Robusta: 5.6-6.5"
        )
        
        soil_moisture_forecast = st.slider(
            "Soil Moisture (%)", 
            min_value=0, 
            max_value=100, 
            value=25, 
            step=5,
            help="Optimal soil moisture for coffee cultivation"
        )
    
    with col2:
        st.markdown("#### 🌾 Fertilization Program")
        
        fertilization_type = st.radio(
            "Fertilization Type",
            options=["Organic", "Non-Organic"],
            help="Type of fertilizer used affects yield potential"
        )
        
        st.markdown("**Fertilization Application Frequency:**")
        fertilization_frequency = st.select_slider(
            "How often do you apply fertilizer?",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: {
                1: "1 - Never",
                2: "2 - Rarely",
                3: "3 - Sometimes",
                4: "4 - Often",
                5: "5 - Always"
            }[x],
            help="Frequency of fertilizer application (Likert scale 1-5)"
        )
        
        st.markdown("#### 🛡️ Pest Management Program")
        
        st.markdown("**Pest Management Application Frequency:**")
        pest_management_frequency = st.select_slider(
            "How often do you apply pest management?",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: {
                1: "1 - Never",
                2: "2 - Rarely",
                3: "3 - Sometimes",
                4: "4 - Often",
                5: "5 - Always"
            }[x],
            help="Frequency of pest management practices (Likert scale 1-5)"
        )
        
        st.markdown("---")
        st.markdown("#### 📊 Calculated Suitability Indices")
        
        # Calculate suitability scores with elevation
        elevation_optimal_center = 900  # midpoint of 600-1200
        elevation_optimal_range = 300  # half of the range
        elevation_score_fc = 1 - abs(elevation_forecast - elevation_optimal_center) / elevation_optimal_range
        elevation_score_fc = max(0, min(1, elevation_score_fc))
        
        temp_score_fc = 1 - abs(temp_avg_forecast - 19.5) / 13
        temp_score_fc = max(0, min(1, temp_score_fc))
        
        rainfall_score_fc = min(rainfall_forecast / 200, 1.5)
        rainfall_score_fc = max(0, min(1, rainfall_score_fc))
        
        climate_suitability_fc = (temp_score_fc * 0.3 + rainfall_score_fc * 0.3 + elevation_score_fc * 0.4)
        
        soil_suitability_fc = 1 - abs(soil_ph_forecast - 6.0) / 1.5
        soil_suitability_fc = max(0, min(1, soil_suitability_fc))
        
        moisture_suitability_fc = soil_moisture_forecast / 35
        moisture_suitability_fc = max(0, min(1, moisture_suitability_fc))
        
        # Calculate environmental stress
        temp_stress_fc = abs(temp_avg_forecast - 19.5) / 13
        rainfall_stress_fc = abs(rainfall_forecast - 200) / 200
        ph_stress_fc = abs(soil_ph_forecast - 6.0) / 1.5
        elevation_stress_fc = abs(elevation_forecast - 900) / 300
        env_stress_fc = (temp_stress_fc + rainfall_stress_fc + ph_stress_fc + elevation_stress_fc) / 4
        env_stress_fc = max(0, min(1, env_stress_fc))
        
        overall_quality_fc = (
            climate_suitability_fc * 0.3 +
            soil_suitability_fc * 0.3 +
            moisture_suitability_fc * 0.2 +
            (1 - env_stress_fc) * 0.2
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Climate Suitability", f"{climate_suitability_fc:.2f}")
            st.metric("Soil Suitability", f"{soil_suitability_fc:.2f}")
            st.metric("Elevation Score", f"{elevation_score_fc:.2f}")
        with col_b:
            st.metric("Overall Quality", f"{overall_quality_fc:.2f}")
            st.metric("Env. Stress", f"{env_stress_fc:.2f}")
            st.metric("Elevation Category", 
                     "Optimal" if 600 <= elevation_forecast <= 1200 else "Sub-optimal")
    
    st.markdown("---")
    
    # Generate Forecast Button
    if st.button("🚀 Generate Yield & Grade Forecast", key='yield_forecast', type='primary'):
        with st.spinner("Calculating yield and grade forecasts..."):
            
            # Calculate forecast
            forecast_df = calculate_yield_forecast(
                plant_age_months=plant_age_forecast,
                farm_area_ha=farm_area_ha,
                climate_suitability=climate_suitability_fc,
                soil_suitability=soil_suitability_fc,
                fertilization_type=fertilization_type,
                fertilization_frequency=fertilization_frequency,
                pest_management_frequency=pest_management_frequency,
                bean_screen_size=6.5,
                overall_quality_index=overall_quality_fc,
                forecast_years=forecast_years
            )
            
            st.success("✅ Forecast generated successfully!")
            
            # Display Results
            st.markdown("---")
            st.markdown("## 📊 Forecast Results")
            
            # Summary Metrics
            st.markdown("### 📈 Key Forecast Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            total_yield_period = forecast_df['Total Yield (kg)'].sum()
            avg_yield_per_year = forecast_df['Yield (kg/ha)'].mean()
            avg_fine_prob = forecast_df['Fine Probability'].mean()
            avg_premium_prob = forecast_df['Premium Probability'].mean()
            
            with col1:
                st.metric(
                    label=f"Total Yield ({forecast_years} years)",
                    value=f"{total_yield_period:,.0f} kg",
                    delta=f"{farm_area_ha} ha"
                )
            
            with col2:
                st.metric(
                    label="Avg Yield/Year",
                    value=f"{avg_yield_per_year:,.0f} kg/ha",
                    delta="per hectare"
                )
            
            with col3:
                st.metric(
                    label="Avg Fine Probability",
                    value=f"{avg_fine_prob*100:.1f}%",
                    delta="Quality grade"
                )
            
            with col4:
                st.metric(
                    label="Avg Premium Probability",
                    value=f"{avg_premium_prob*100:.1f}%",
                    delta="Quality grade"
                )
            
            st.markdown("---")
            
            # Yield Projection Over Time
            st.markdown("### 📈 Yield Projection Over Time")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=forecast_df['Year'],
                y=forecast_df['Yield (kg/ha)'],
                mode='lines+markers',
                name='Yield per Hectare',
                line=dict(color='#2c5f2d', width=3),
                marker=dict(size=10)
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['Year'],
                y=forecast_df['Total Yield (kg)'],
                mode='lines+markers',
                name='Total Farm Yield',
                line=dict(color='#97bc62', width=3, dash='dash'),
                marker=dict(size=10),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=f"Projected Coffee Yield Over {forecast_years} Years",
                xaxis_title="Year",
                yaxis_title="Yield (kg/ha)",
                yaxis2=dict(
                    title="Total Yield (kg)",
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                height=500,
                showlegend=True,
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Grade Distribution Over Time
            st.markdown("### 🏆 Grade Distribution Forecast")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=forecast_df['Year'],
                y=forecast_df['Fine Yield (kg/ha)'],
                name='Fine Robusta',
                marker_color='#2c5f2d'
            ))
            
            fig.add_trace(go.Bar(
                x=forecast_df['Year'],
                y=forecast_df['Premium Yield (kg/ha)'],
                name='Premium Robusta',
                marker_color='#97bc62'
            ))
            
            fig.add_trace(go.Bar(
                x=forecast_df['Year'],
                y=forecast_df['Commercial Yield (kg/ha)'],
                name='Commercial',
                marker_color='#d3d3d3'
            ))
            
            fig.update_layout(
                title="Projected Yield by Grade Classification",
                xaxis_title="Year",
                yaxis_title="Yield (kg/ha)",
                barmode='stack',
                height=500,
                showlegend=True,
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)

# CONTINUATION OF YIELD FORECASTING PAGE (Second Half)
            
            # Grade Probability Trends
            st.markdown("### 📊 Grade Probability Trends")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=forecast_df['Year'],
                y=forecast_df['Fine Probability'] * 100,
                mode='lines+markers',
                name='Fine Probability',
                line=dict(color='#2c5f2d', width=3),
                marker=dict(size=10),
                fill='tonexty'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['Year'],
                y=forecast_df['Premium Probability'] * 100,
                mode='lines+markers',
                name='Premium Probability',
                line=dict(color='#97bc62', width=3),
                marker=dict(size=10)
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['Year'],
                y=forecast_df['Commercial Probability'] * 100,
                mode='lines+markers',
                name='Commercial Probability',
                line=dict(color='#d3d3d3', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="Grade Classification Probability Over Time",
                xaxis_title="Year",
                yaxis_title="Probability (%)",
                height=500,
                hovermode='x unified',
                showlegend=True,
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Forecast Table
            st.markdown("### 📋 Detailed Forecast Data")
            
            # Format the dataframe for display
            display_df = forecast_df.copy()
            display_df['Fine Probability'] = (display_df['Fine Probability'] * 100).round(1).astype(str) + '%'
            display_df['Premium Probability'] = (display_df['Premium Probability'] * 100).round(1).astype(str) + '%'
            display_df['Commercial Probability'] = (display_df['Commercial Probability'] * 100).round(1).astype(str) + '%'
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Management Recommendations
            st.markdown("---")
            st.markdown("### 💡 Management Recommendations")
            
            # Analyze forecast trends
            yield_trend = forecast_df['Yield (kg/ha)'].iloc[-1] - forecast_df['Yield (kg/ha)'].iloc[0]
            fine_trend = forecast_df['Fine Probability'].iloc[-1] - forecast_df['Fine Probability'].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Yield Optimization")
                
                if plant_age_forecast < 36:
                    st.warning("⏳ **Plants not yet producing** - First harvest expected in " + 
                             f"{36 - plant_age_forecast} months")
                    st.markdown("""
                    **Pre-production focus:**
                    - Establish strong root systems
                    - Implement proper pruning and training
                    - Build soil fertility with organic matter
                    - Control weeds and maintain mulch
                    """)
                elif yield_trend > 0:
                    st.success("📈 **Increasing yield trend** - Good management trajectory")
                    st.markdown("""
                    **Continue current practices:**
                    - Maintain fertilization schedule
                    - Keep up pest management routines
                    - Monitor soil health regularly
                    - Ensure consistent water management
                    """)
                else:
                    st.warning("📉 **Declining yield trend** - Intervention recommended")
                    st.markdown("""
                    **Improvement actions:**
                    - Increase fertilization frequency
                    - Enhance pest control measures
                    - Consider soil amendment
                    - Evaluate pruning practices
                    """)
                
                if fertilization_frequency < 3:
                    st.info("🌾 **Increase fertilization** to boost yields")
                    st.markdown("- Target 3-4 applications per year")
                    st.markdown("- Use complete fertilizer (14-14-14)")
                    st.markdown("- Apply during active growth periods")
                
                if pest_management_frequency < 3:
                    st.info("🛡️ **Improve pest management** to reduce losses")
                    st.markdown("- Regular monitoring and scouting")
                    st.markdown("- Integrated pest management approach")
                    st.markdown("- Control coffee berry borer promptly")
            
            with col2:
                st.markdown("#### 🏆 Quality Improvement")
                
                avg_fine = forecast_df['Fine Probability'].mean()
                
                if avg_fine > 0.5:
                    st.success("⭐ **High Fine Robusta potential** - Excellent quality trajectory")
                    st.markdown("""
                    **Maintain premium quality:**
                    - Continue selective harvesting
                    - Strict post-harvest processing
                    - Proper fermentation control (18-24hrs)
                    - Optimal drying conditions
                    """)
                elif avg_fine > 0.3:
                    st.info("🥇 **Good Premium potential** - Room for quality improvement")
                    st.markdown("""
                    **Achieve Fine grade:**
                    - Eliminate all primary defects
                    - Reduce secondary defects to ≤5
                    - Improve harvesting selectivity
                    - Enhance processing consistency
                    """)
                else:
                    st.warning("📦 **Commercial grade focus** - Significant quality improvements needed")
                    st.markdown("""
                    **Quality enhancement priorities:**
                    - Harvest only ripe cherries
                    - Prevent over-fermentation
                    - Improve drying methods
                    - Better sorting and grading
                    """)
                
                if climate_suitability_fc < 0.7:
                    st.warning("🌦️ **Climate conditions sub-optimal**")
                    st.markdown("- Implement shade management")
                    st.markdown("- Improve water management")
                    st.markdown("- Consider microclimate modification")
                
                if elevation_forecast < 600:
                    st.warning(f"⛰️ **Elevation ({elevation_forecast}m) below optimal range**")
                    st.markdown("- Robusta performs best at 600-1,200 masl")
                    st.markdown("- Consider improved agronomic practices")
                elif elevation_forecast > 1200:
                    st.warning(f"⛰️ **Elevation ({elevation_forecast}m) above optimal range**")
                    st.markdown("- Robusta may experience stress at higher elevations")
                    st.markdown("- Consider switching to Arabica for elevations >900 masl")
                
                if soil_suitability_fc < 0.7:
                    st.warning("🌿 **Soil conditions need improvement**")
                    st.markdown(f"- Adjust pH to 5.6-6.5 range (current: {soil_ph_forecast})")
                    st.markdown("- Increase organic matter content")
                    st.markdown("- Regular soil testing recommended")

# =====================================
# PAGE 8: PNS STANDARDS & GUIDELINES (SAME AS ORIGINAL, KEEPING ALL CONTENT)
# =====================================

elif page == "💡 PNS Standards & Guidelines":
    st.title("💡 Coffee Grading Standards & Guidelines")
    st.markdown("### CQI/UCDA Fine Robusta Classification System")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Grading Standards", "🌱 Cultivation Requirements", "🔬 Defect Types", "📊 Size Classification"])
    
    with tab1:
        st.markdown("## CQI/UCDA Fine Robusta Grading Standards")
        
        st.info("""
        **International Standard for Robusta Coffee Quality**
        
        Based on Coffee Quality Institute (CQI) and Uganda Coffee Development Authority (UCDA) 
        Fine Robusta classification system.
        """)
        
        # Fine Robusta standards
        st.markdown("### Coffee Grade Classifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### ⭐ Fine Robusta
            
            **Requirements:**
            - ✅ Zero (0) primary (Category 1) defects
            - ✅ Maximum five (5) secondary (Category 2) defects
            - ✅ Bean size ≥6.5mm
            - ✅ Maximum three (3) quakers in 100g roasted sample
            - ✅ Cupping score ≥ 80.00 (when facilities available)
            - ✅ Free of all foreign (non-coffee) odors
            
            **Sample Size:** 350g green coffee
            
            **Market:** Specialty/Premium markets
            """)
            
            st.info("""
            ### 🥇 Premium Robusta
            
            **Requirements:**
            - ✅ Maximum twelve (12) combined primary and/or secondary defects
            - ✅ Bean size ≥6.0mm
            - ✅ Maximum five (5) quakers in 100g roasted sample
            - ✅ Cupping score ≥ 80.00 (when facilities available)
            - ✅ Free of all foreign (non-coffee) odors
            
            **Sample Size:** 350g green coffee
            
            **Market:** Premium/Export markets
            """)
        
        with col2:
            st.warning("""
            ### 📦 Commercial Robusta
            
            **Definition:**
            Coffees not complying with Fine Robusta or Premium Robusta grade specifications 
            are considered commodity or commercial coffee.
            
            **Characteristics:**
            - More than 12 combined defects, OR
            - Bean size <6.0mm, OR
            - Cupping score below 80 (when assessed), OR
            - Presence of foreign odors
            
            **Market:** Commodity/Instant coffee markets
            """)
            
            st.markdown("---")
            
            st.markdown("### 🎯 Grading Principles")
            st.markdown("""
            **Key Rules:**
            1. Only **full equivalent defects** are counted
            2. Defects must be recorded in **whole numbers**
            3. In beans with multiple imperfections, only the **most severe** is recorded
            4. Imperfections must match criteria in the Fine Robusta defect handbook
            5. **Primary defects** are more severe than secondary defects
            6. Both green (350g) and roasted (100g) samples are evaluated
            
            **Note:** Cupping score is optional when assessment facilities are not available.
            In such cases, grading is based on defect counts and bean size.
            """)
        
        st.markdown("---")
        
        # PNS Reference table
        st.markdown("### 📊 Philippine National Standard (PNS) Reference")
        st.caption("*PNS grading is used as reference for defect percentage benchmarks*")
        
        pns_reference = pd.DataFrame({
            'PNS Grade': ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
            'Max Defects (%)': ['10%', '15%', '25%', '40%'],
            'Equivalent CQI/UCDA': [
                'Fine/Premium quality range',
                'Premium quality range', 
                'Commercial quality range',
                'Below commercial standards'
            ],
            'Target Market': [
                'Specialty/Premium',
                'Premium/Export',
                'Commercial/Commodity',
                'Instant coffee/Blends'
            ]
        })
        st.dataframe(pns_reference, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Defect breakdown reference
        st.markdown("### 📉 Maximum Defect Percentages by Type (PNS Reference)")
        st.caption("*These PNS limits help guide defect control for achieving Fine/Premium grades*")
        
        defect_limits = pd.DataFrame({
            'Defect Type': [
                'Black beans', 'Infested beans', 'Broken beans', 'Immature beans',
                'Husk beans', 'Fermented/sour beans', 'Foreign matter', 'Admixture'
            ],
            'Category': [
                'Primary', 'Primary', 'Secondary', 'Secondary',
                'Secondary', 'Primary', 'Secondary', 'Secondary'
            ],
            'PNS Grade 1 (%)': [4, 4, 3, 2, 1, 1, 1, 0.5],
            'PNS Grade 2 (%)': [6, 5, 5, 3, 1.5, 1.5, 1, 0.5]
        })
        st.dataframe(defect_limits, use_container_width=True, hide_index=True)
    
# CONTINUATION OF PNS STANDARDS PAGE (Remaining Tabs)
    
    with tab2:
        st.markdown("## Robusta Cultivation Requirements")
        
        st.info("""
        **Physical Characteristics and Requirements for Robusta Cultivation**
        
        Based on Philippine Coffee Technoguide standards
        """)
        
        # Cultivation requirements table
        cult_reqs = pd.DataFrame({
            'Parameter': [
                'Elevation (masl)',
                'Temperature (°C)',
                'Sunshine Requirements',
                'Wind Requirements',
                'Relative Humidity (%)',
                'Rainfall (mm/month)',
                'Soil pH',
                'Soil Depth (m)',
                'Organic Matter'
            ],
            'Robusta Requirements': [
                '600 - 1,200',
                '13 - 26',
                '50%',
                'Slight',
                '75 - 85',
                '200',
                '5.6 - 6.5',
                '1.5',
                'Rich in OM'
            ]
        })
        
        st.dataframe(cult_reqs, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌱 Plant Characteristics")
            st.markdown("""
            **Robusta (Coffea robusta)**
            - Also known as "kapeng manipis"
            - Strong taste with 2-2.5% caffeine content
            - High yielding, pest and disease resistant
            - Tree height: Large, umbrella-shaped, ~4.5 meters
            - Root system: Shallow
            - Flowers: White with 5-6 petals, cross pollination
            - Leaves: Thin with wavy margins
            """)
        
        with col2:
            st.markdown("### 📅 Production Timeline")
            st.markdown("""
            **Growth Stages:**
            - **Planting:** Rainy season recommended
            - **Bearing of fruits:** Commences 3rd year from transplanting
            - **Prime production:** 4-6 years (48-72 months)
            - **Rejuvenation:** 10-30 years depending on vigor
            
            **Spacing:**
            - Monocropping: 3m x 2m
            - Intercropping: 3m x 2.5m
            """)
        
        st.markdown("---")
        
        st.markdown("### 🌾 Soil and Fertilization")
        st.markdown("""
        **Basal Application (at planting):**
        - Complete Fertilizer (14-14-14): 75 grams/tree or 124 kg/ha
        
        **Soil Requirements:**
        - pH: 5.6 - 6.5 (slightly acidic)
        - Organic Matter: Rich (>3%)
        - Depth: Minimum 1.5 meters
        - Nitrogen: 0.15-0.25%
        - Phosphorus: >7 mg/100g
        - Good drainage but moisture retentive
        """)
    
    with tab3:
        st.markdown("## Common Coffee Bean Defects")
        
        st.warning("""
        **Important:** Understanding defects is crucial for quality control and grading.
        
        Defects are classified into two categories:
        - **Primary (Category 1):** More severe defects
        - **Secondary (Category 2):** Less severe defects
        """)
        
        # Create defect reference
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔴 Primary Defects (Category 1)")
            
            with st.expander("1. Black Beans"):
                st.markdown("""
                **Description:**
                - Brown to black beans
                - Shrunken and wrinkled
                - Flat faced, crack too open
                
                **Causes:**
                - Lack of water during cherry development
                - Over fermentation
                - Overripe cherries picked from ground
                """)
            
            with st.expander("2. Moldy Beans"):
                st.markdown("""
                **Description:**
                - Coffee beans infested by mold
                - Yellowish or reddish spores present
                
                **Causes:**
                - Over fermentation
                - Long interruptions during drying
                - Storage with high moisture content
                """)
            
            with st.expander("3. Sour/Partial Sour Beans"):
                st.markdown("""
                **Description:**
                - Light brown to dark brown
                - Crack free of tegument
                - Silver skin can be reddish-brown
                
                **Causes:**
                - Delay between picking and depulping
                - Overextended fermentation
                - Use of dirty water
                - High moisture storage
                """)
            
            with st.expander("4. Insect-Damaged Beans"):
                st.markdown("""
                **Description:**
                - Beans with small holes caused by insects
                
                **Causes:**
                - Attack on cherries by Hypothenemus haempei (coffee berry borer)
                - Attack during storage by Araecerus Fasciculatus
                - Inadequate storage conditions
                """)
        
        with col2:
            st.markdown("### 🟡 Secondary Defects (Category 2)")
            
            with st.expander("5. Broken/Pressed Beans"):
                st.markdown("""
                **Description:**
                - Bruised beans with partial fractures
                
                **Causes:**
                - Damage during depulping
                - Improper drying process
                - Milling issues
                - Milling parchment with high moisture
                """)
            
            with st.expander("6. Immature Beans"):
                st.markdown("""
                **Description:**
                - Green or light grey beans
                - Very adherent silver skin
                - Smaller than normal size
                - Withered surface
                
                **Causes:**
                - Cherries picked before ripeness
                - Lack of fertilizer
                - Drought and rust disease
                """)
            
            with st.expander("7. Faded/Discolored Beans"):
                st.markdown("""
                **Types:**
                - **Streaked:** Irregular greenish color (uneven drying)
                - **Oldish:** White to brown (long/poor storage)
                - **Amber/Buttery:** Yellow, semi-transparent (iron deficiency)
                - **Overdried:** Amber to yellow (excessive drying)
                
                **Prevention:**
                - Proper drying procedures
                - Good storage conditions (25-30°C, 50-60% humidity)
                - Adequate soil nutrition
                """)
            
            with st.expander("8. Other Defects"):
                st.markdown("""
                **Crystallized Beans:**
                - Grayish/bluish, brittle
                - Cause: Dried above 60°C
                
                **Shrunk Beans:**
                - Wrinkled appearance
                - Causes: Drought, lack of fertilization
                
                **Wet/Undried Beans:**
                - Dark green, soft texture
                - Causes: High humidity, incomplete drying
                """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Defect Prevention Best Practices")
        st.success("""
        1. **Harvesting:**
           - Select only ripe cherries (selective picking/priming)
           - Avoid green, dried, and overripe cherries
           - Use clean containers
           - Avoid ground contact
        
        2. **Processing:**
           - Start drying within 12 hours after harvest
           - Proper fermentation time (18-24 hours for wet method)
           - Use clean water throughout
           - Maintain proper drying temperature (35-60°C gradually)
        
        3. **Storage:**
           - Maintain 12-14% moisture content
           - Store at 25-30°C with 50-60% humidity
           - Protect from pests
           - Regular monitoring
        
        4. **Sorting:**
           - Remove defective beans before processing
           - Regular quality checks
           - Proper grading procedures
        """)
    
    with tab4:
        st.markdown("## Bean Size Classification")
        
        st.info("""
        **Size classification for Robusta green coffee beans**
        
        Based on screen size and weight measurements
        """)
        
        # Size classification table
        size_class = pd.DataFrame({
            'Class': ['Large', 'Medium', 'Small', 'Below Standard'],
            'Dry Processed': [
                'Retained by 5.6mm × 5.6mm screen (max 1% pass through)',
                'Not specified',
                'Not specified',
                'Below 5.6mm'
            ],
            'Wet Processed': [
                'Retained by 7.5mm diameter holes (max 2.5% pass through)',
                'Pass 7.5mm, retained by 6.5mm (max 2.5% pass through)',
                'Pass 6.5mm, retained by 5.5mm (max 2.5% pass through)',
                'Below 5.5mm'
            ]
        })
        
        st.dataframe(size_class, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📏 Screen Size Standards")
            st.markdown("""
            **Robusta Bean Sizing:**
            - **Large:** ≥7.5mm (wet) or ≥5.6mm (dry)
            - **Medium:** 6.5-7.5mm (wet processed)
            - **Small:** 5.5-6.5mm (wet processed)
            
            **Quality Impact:**
            - Larger beans generally indicate better growing conditions
            - More uniform size = better roasting consistency
            - Size affects pricing and market grade
            """)
        
        with col2:
            st.markdown("### ⚖️ Weight Classification")
            st.markdown("""
            **Alternative Measurement:**
            - Can also classify by bean count per 25g
            - Fewer beans per 25g = larger individual beans
            - More consistent sizing = premium grade
            
            **Sorting Methods:**
            - Screen sieves (most common)
            - Gravity separation
            - Electronic sorters for high-volume operations
            """)
        
        st.markdown("---")
        
        st.markdown("### ✅ Good Quality Coffee Beans Characteristics")
        st.success("""
        **A good quality Robusta green bean should have:**
        
        1. **Appearance:**
           - Uniform in sizes and shapes
           - Hard and not spongy texture
           - Glossy and smooth surface
           - Greenish to deep green color with fresh background hue
        
        2. **Purity:**
           - Free from molds
           - No foreign bodies or insects
           - No imperfections or defects
           - Free from undesirable or rancid odor
        
        3. **Processing:**
           - Properly dried (12-14% moisture)
           - Well sorted and graded
           - Clean and free from debris
           - Properly stored before sale
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Grading Process Summary")
        
        grading_process = pd.DataFrame({
            'Step': ['1. Sampling', '2. Weighing', '3. Sorting', '4. Defect Count', '5. Size Classification', '6. Final Grade'],
            'Description': [
                'Take 300g (PNS) or 350g (Fine Robusta) sample',
                'Verify exact sample weight',
                'Manually or mechanically sort beans',
                'Count and classify all defects',
                'Determine size class using screens',
                'Assign grade based on standards'
            ],
            'Standard': [
                'Random representative sample',
                'Calibrated scale',
                'Remove foreign matter first',
                'Use defect handbook',
                'Multiple screen sizes',
                'PNS or CQI/UCDA standards'
            ]
        })
        
        st.dataframe(grading_process, use_container_width=True, hide_index=True)

# =====================================
# FOOTER
# =====================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Robusta Coffee Grading Dashboard v6.0 - PNS Compliant</strong></p>
        <p>Based on Philippine National Standards (PNS) & CQI/UCDA Fine Robusta Standards</p>
        <p>☕ Empowering Philippine coffee farmers with standards-based grading insights</p>
        <p><em>Standards Reference: PNS Green Coffee Beans & Coffee Technoguide</em></p>
        <p><strong>Updates:</strong> Bean size + defect grading (cupping optional) | ML-based harvest yield prediction</p>
    </div>
    """, unsafe_allow_html=True)