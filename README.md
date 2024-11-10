# Amazon Alexa Reviews Sentiment Analysis and Prediction

This project analyzes Amazon Alexa reviews to predict customer sentiment and provides a web interface for real-time sentiment prediction. The project consists of two parts:
1. **Data Analysis and Model Building**: Using machine learning to classify reviews as positive or negative.
2. **Web Application**: A Flask-based app that provides an interface for users to predict sentiment on individual reviews or upload CSV files for batch predictions.

## Overview
- **Goal**: Predict customer sentiment from review text and provide an accessible interface for predictions.
- **Dataset**: Amazon Alexa product reviews with attributes like rating, review text, and feedback label (positive or negative).

## Project Structure
1. **Data Analysis and Modeling**: Jupyter notebook for data exploration, preprocessing, and model training.
2. **Web Application**: Flask app with endpoints to serve predictions.

### 1. Data Analysis and Model Building
The `notebook.ipynb` notebook contains:
- **Exploratory Data Analysis**: Visualizes ratings, feedback distribution, review length, and word clouds for positive and negative reviews.
- **Text Preprocessing**: Clean text data, remove stopwords, stem words, and vectorize using `CountVectorizer`.
- **Modeling**: 
  - Models: Random Forest, XGBoost, and Decision Tree classifiers.
  - Results: Random Forest and XGBoost showed the highest accuracy.
  - Saved Models: The best-performing models are saved as pickle files for use in the app.

### 2. Web Application
The Flask app (`api.py`) provides two main endpoints:
- **Home Page** (`/`): Displays an introductory page with a sentiment analysis tool description.
- **Prediction Endpoint** (`/predict`): Accepts text or CSV files and returns sentiment predictions.

The app can:
- **Single Prediction**: Predict the sentiment of a single text input.
- **Batch Prediction**: Upload a CSV file with reviews for batch predictions. Returns a downloadable CSV with predictions.

### Files
- `notebook.ipynb`: Jupyter notebook for EDA, preprocessing, and model training.
- `api.py`: Flask API for handling predictions.
- `main.py`: Streamlit script for a simple web interface.
- `models/`: Folder containing saved model files (XGBoost, Random Forest, etc.)
- `templates/`: Contains HTML templates for the Flask app interface.

## Key Features
1. **Data Exploration and Visualization**:
   - Bar plots and pie charts for rating and feedback distribution.
   - Word clouds for common positive and negative words.

2. **Machine Learning Models**:
   - Trained Random Forest, XGBoost, and Decision Tree models.
   - Best-performing models saved for use in the app.

3. **Web Interface**:
   - Flask-based web app for real-time text sentiment prediction.
   - Option to upload CSV files for batch sentiment analysis.

## Results
- **Model Performance**:
  - **Random Forest**: 94.5% test accuracy
  - **XGBoost**: 94.2% test accuracy
  - **Decision Tree**: 92.3% test accuracy
- **Conclusion**: Random Forest and XGBoost are the best-performing models, providing valuable insights into customer sentiment for Amazon Alexa products.

## Conclusion
This project demonstrates effective sentiment analysis on product reviews using machine learning. With an accessible web interface, users can analyze sentiment for individual reviews or batches, providing valuable insights into customer opinions for Amazon Alexa products.
