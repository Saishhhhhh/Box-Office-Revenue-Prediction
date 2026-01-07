<div align="center">

# 🎬 Box Office Revenue Prediction

### *Forecasting Indian Movie Box Office with Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![RandomForest](https://img.shields.io/badge/Model-Random%20Forest-green.svg)](https://scikit-learn.org/stable/modules/ensemble.html)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Box Office Revenue Prediction** is a data-driven machine learning system that estimates  
**India Net** and **Worldwide** box office collections for Indian films using a realistic, two-stage forecasting pipeline.

</div>


## ✨ Features

- **Model 1 (India Net)**: High-accuracy prediction model (R² Score: 0.95)
- **Model 2 (Worldwide)**: Optional worldwide collection prediction (R² Score: 0.86)
- **Interactive Web App**: User-friendly Streamlit interface
- **Pan-India Detection**: Automatically detects Pan-India movies (4+ languages)
- **Log Transformation**: Models work in log space for better accuracy

## 📊 Data Source

The dataset was **self-collected by scraping publicly available box office data** from the Sacnilk website.

The raw data was cleaned, aggregated, and transformed before modeling.  
This project is **not affiliated with or endorsed by Sacnilk**.


## 📋 Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- streamlit>=1.28.0
- pandas>=1.5.0
- numpy>=1.23.0
- scikit-learn>=1.2.0
- joblib>=1.2.0

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Box-Office-Revenue-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📱 Using the App

### Input Fields

1. **Release Date**: Select the movie release date
2. **Industry**: Choose the primary industry (Hindi, Tamil, Telugu, Kannada, Malayalam)
3. **Languages**: Select all languages the movie is available in (multiple selection)
   - If 4+ languages are selected, the movie is automatically marked as Pan-India
4. **Genres**: Select all applicable genres (multiple selection)
5. **Week 1 Collection**: Enter the combined Week 1 NET collection of all language versions in Crores (₹)
   - **Important**: This should be the total NET collection (not gross) combining all language versions

### Prediction Flow

1. Fill in all the required fields in the sidebar
2. Click **"🚀 Predict Collection"** button
3. View the **India Net Collection** prediction
4. (Optional) Check **"🌍 Get Worldwide Collection Prediction"** to see worldwide collection

### Important Notes

- ⚠️ **Disclaimer**: These are predictive estimates and may contain errors. Actual results may vary.
- The Week 1 Collection must be **NET collection** (not gross)
- Add up Week 1 collections from all language versions of the movie
- Worldwide prediction accuracy is lower than India Net prediction

## 📊 Model Information

### Model 1: India Net Collection

- **Algorithm**: Random Forest Regressor
- **Features**: 30 features including:
  - Release date components (year, month, day, weekday)
  - Industry (one-hot encoded)
  - Languages (one-hot encoded)
  - Genres (one-hot encoded)
  - Pan-India indicator
  - Number of languages
  - Log-transformed Week 1 collection
- **Performance**: R² Score: 0.95, MAE: ~₹7.9 Cr

### Model 2: Worldwide Collection

- **Algorithm**: Random Forest Regressor
- **Features**: 31 features (all Model 1 features + predicted_log_india_net)
- **Approach**: Predicts residual to add to India Net prediction
- **Performance**: R² Score: 0.56, MAE: ~₹218 Cr

## 📁 Project Structure

```
Box-Office-Revenue-Prediction/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/
│   ├── model_india_net_rf.joblib          # Model 1 (India Net)
│   └── model_worldwide_residual_rf.joblib # Model 2 (Worldwide)
│
├── notebooks/
│   ├── 01_Dataset_Cleaning.ipynb   # Data cleaning
│   ├── 02_Preprocessing.ipynb      # Feature engineering
│   ├── 03_Model_1_Training.ipynb   # Model training
│   └── 04_Model_Packaging.ipynb    # Model packaging
│
└── data/
    ├── movies.csv                  # Raw data
    ├── movies_cleaned.csv          # Cleaned data
    └── movies_cleaned_2.csv        # Preprocessed data
```

## 🔧 Development

### Training the Models

The models were trained using Jupyter notebooks:
1. `01_Dataset_Cleaning.ipynb`: Clean and prepare the dataset
2. `02_Preprocessing.ipynb`: Feature engineering and log transformations
3. `03_Model_1_Training.ipynb`: Train Model 1 and Model 2
4. `04_Model_Packaging.ipynb`: Save models as joblib files

### Feature Engineering

- Date features extracted from release date
- One-hot encoding for categorical variables (industry, languages, genres)
- Pan-India detection (4+ languages)
- Log transformation: `log1p()` for inputs, `expm1()` for outputs

## 📝 License

See LICENSE file for details.

---

**Note**: This is a predictive model and results should be used as estimates only. Actual box office performance may vary based on many factors not captured in the model.

<div align="center">

**Made with ❤️ by Saish**

⭐ Star this repo if you find it useful!

</div>