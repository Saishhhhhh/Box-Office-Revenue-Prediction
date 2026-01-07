import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Box Office Revenue Prediction",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    """Load the trained models"""
    try:
        model1 = joblib.load('model_india_net_rf.joblib')
        model2 = joblib.load('model_worldwide_residual_rf.joblib')
        return model1, model2
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()

model1, model2 = load_models()

# Title and description
st.title("🎬 Box Office Revenue Prediction")
st.markdown("### Predict India Net Collection for Indian Movies")
st.warning("⚠️ **Disclaimer**: These are predictive estimates and may contain errors. Actual results may vary.")

# Sidebar for inputs
with st.sidebar:
    st.header("📝 Movie Details")
    
    # Release Date
    release_date = st.date_input(
        "Release Date",
        value=datetime(2024, 1, 1),
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2030, 12, 31),
        help="Select the movie release date"
    )
    
    # Industry (Single Selection)
    industry = st.selectbox(
        "Industry",
        options=["Hindi", "Tamil", "Telugu", "Kannada", "Malayalam"],
        help="Select the primary industry that made the movie"
    )
    
    # Languages (Multiple Selection)
    st.markdown("**Languages**")
    languages = st.multiselect(
        "Select all languages",
        options=["English", "Hindi", "Tamil", "Telugu", "Kannada", "Malayalam", "Other_Language"],
        default=[],
        help="Select all languages the movie is available in. If 4+ languages selected, it will be marked as Pan-India."
    )
    
    # Genres (Multiple Selection)
    st.markdown("**Genres**")
    genres = st.multiselect(
        "Select genres",
        options=["Action", "Adventure", "Comedy", "Crime", "Drama", "Family", 
                 "Horror", "Mystery", "Romantic", "Thriller", "Other_Genre"],
        default=[],
        help="Select all genres that apply to the movie"
    )
    
    # Week1 Collection
    week1_collection = st.number_input(
        "Week 1 Collection (₹ Crores)",
        min_value=0.0,
        value=10.0,
        step=0.1,
        format="%.2f",
        help="Enter the combined Week 1 NET collection of all languages in Crores. Add up the Week 1 net collections from all language versions of the movie."
    )
    st.caption("ℹ️ **Note**: Enter the total Week 1 NET collection combining all language versions. The collection values must be net collections (not gross).")

# Main content area
st.header("🎯 India Net Collection Prediction")

# Prepare features for Model 1
def prepare_features(release_date, industry, languages, genres, week1_collection):
    """Prepare feature vector for model prediction"""
    # Extract date features
    release_year = release_date.year
    rel_month = release_date.month
    rel_day = release_date.day
    release_weekday = release_date.weekday()  # 0=Monday, 6=Sunday
    
    # Industry one-hot encoding
    industry_features = {
        'industry_Hindi': 1 if industry == 'Hindi' else 0,
        'industry_Tamil': 1 if industry == 'Tamil' else 0,
        'industry_Telugu': 1 if industry == 'Telugu' else 0,
        'industry_Kannada': 1 if industry == 'Kannada' else 0,
        'industry_Malayalam': 1 if industry == 'Malayalam' else 0,
    }
    
    # Language features
    num_languages = len(languages)
    is_pan_india = 1 if num_languages >= 4 else 0
    
    language_features = {
        'English': 1 if 'English' in languages else 0,
        'Hindi': 1 if 'Hindi' in languages else 0,
        'Tamil': 1 if 'Tamil' in languages else 0,
        'Telugu': 1 if 'Telugu' in languages else 0,
        'Kannada': 1 if 'Kannada' in languages else 0,
        'Malayalam': 1 if 'Malayalam' in languages else 0,
        'Other_Language': 1 if 'Other_Language' in languages else 0,
    }
    
    # Genre features
    genre_features = {
        'Action': 1 if 'Action' in genres else 0,
        'Adventure': 1 if 'Adventure' in genres else 0,
        'Comedy': 1 if 'Comedy' in genres else 0,
        'Crime': 1 if 'Crime' in genres else 0,
        'Drama': 1 if 'Drama' in genres else 0,
        'Family': 1 if 'Family' in genres else 0,
        'Horror': 1 if 'Horror' in genres else 0,
        'Mystery': 1 if 'Mystery' in genres else 0,
        'Romantic': 1 if 'Romantic' in genres else 0,
        'Thriller': 1 if 'Thriller' in genres else 0,
        'Other_Genre': 1 if 'Other_Genre' in genres else 0,
    }
    
    # Log transformation for week1 collection
    log_weekend1 = np.log1p(week1_collection)
    
    # Combine all features in the exact order expected by the model
    features = {
        'release_year': release_year,
        'rel_month': rel_month,
        'rel_day': rel_day,
        'release_weekday': release_weekday,
        **industry_features,
        'is_pan_india': is_pan_india,
        'num_languages': num_languages,
        **language_features,
        **genre_features,
        'log_weekend1': log_weekend1,
    }
    
    # Convert to DataFrame with correct column order
    feature_order = [
        'release_year', 'rel_month', 'rel_day', 'release_weekday',
        'industry_Hindi', 'industry_Tamil', 'industry_Telugu', 
        'industry_Kannada', 'industry_Malayalam',
        'is_pan_india', 'num_languages',
        'English', 'Hindi', 'Tamil', 'Telugu', 'Kannada', 'Malayalam', 'Other_Language',
        'Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Family',
        'Horror', 'Mystery', 'Romantic', 'Thriller', 'Other_Genre',
        'log_weekend1'
    ]
    
    return pd.DataFrame([features])[feature_order]

# Prediction button
predict_clicked = st.button("🚀 Predict Collection", type="primary", use_container_width=True)

# Handle new prediction
if predict_clicked:
    # Validate inputs
    if not languages:
        st.warning("⚠️ Please select at least one language")
    elif not genres:
        st.warning("⚠️ Please select at least one genre")
    else:
        with st.spinner("Predicting..."):
            # Prepare features
            X1 = prepare_features(release_date, industry, languages, genres, week1_collection)
            
            # Model 1 prediction
            log_india_net = model1.predict(X1)[0]
            india_net = np.expm1(log_india_net)
            
            # Store prediction in session state
            st.session_state['india_net_predicted'] = True
            st.session_state['india_net'] = india_net
            st.session_state['log_india_net'] = log_india_net
            st.session_state['X1'] = X1
            st.session_state['get_worldwide'] = False  # Reset worldwide checkbox
            st.rerun()

# Show prediction results if available
if st.session_state.get('india_net_predicted', False):
    india_net = st.session_state['india_net']
    
    # Display India Net prediction
    st.success("✅ Prediction Complete!")
    st.markdown("---")
    st.metric(
        label="🇮🇳 **Predicted India Net Collection**",
        value=f"₹ {india_net:.2f} Cr",
        delta=None
    )
    
    # Option to get worldwide prediction
    st.markdown("---")
    get_worldwide = st.checkbox(
        "🌍 Get Worldwide Collection Prediction",
        value=st.session_state.get('get_worldwide', False),
        key='get_worldwide',
        help="Check this to get worldwide collection prediction (lower accuracy)"
    )
    
    if get_worldwide:
        st.info("ℹ️ Note: Worldwide prediction accuracy is lower than India Net prediction")
        
        # Get stored values
        log_india_net = st.session_state['log_india_net']
        X1 = st.session_state['X1']
        
        # Prepare Model 2 features
        X2 = X1.copy()
        X2['predicted_log_india_net'] = log_india_net
        
        # Ensure correct feature order for Model 2
        model2_feature_order = list(model2.feature_names_in_)
        missing_features = set(model2_feature_order) - set(X2.columns)
        if missing_features:
            st.error(f"Missing features for Model 2: {missing_features}")
        else:
            X2 = X2[model2_feature_order]
            
            # Model 2 prediction
            log_residual = model2.predict(X2)[0]
            log_worldwide = log_india_net + log_residual
            worldwide = np.expm1(log_worldwide)
            
            # Display Worldwide prediction
            st.metric(
                label="🌐 **Predicted Worldwide Collection**",
                value=f"₹ {worldwide:.2f} Cr",
                delta=None
            )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Box Office Revenue Prediction System | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
