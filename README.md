# **Sentiment Analysis of SimCity BuildIt App Reviews**

## Project Overview
This project aims to analyze user reviews for the *SimCity BuildIt* app on the Google Play Store to gain insights into user sentiment, satisfaction, and common issues. By leveraging sentiment analysis, we aim to uncover patterns in user feedback that can help improve the app experience, enhance customer support, and identify areas for potential app updates.

## Goal
The primary goal of this project is to perform sentiment analysis on user reviews to classify them as positive, negative, or neutral. This classification will help highlight frequently mentioned issues and areas that users appreciate, providing insights into user engagement and satisfaction.

## Key Components

### 1. Data Collection
- Reviews are scraped directly from the Google Play Store using the `google-play-scraper` library, which provides up-to-date and relevant feedback from users.

### 2. Data Preprocessing
- Text data is cleaned and preprocessed to prepare it for sentiment analysis. This includes removing unnecessary characters, stopwords, and performing tokenization.

### 3. Feature Extraction
- **TF-IDF**: Converts text into numerical data, representing word frequency and importance.
- **Word2Vec**: Embedding technique used to capture semantic relationships between words.

### 4. Sentiment Classification
- Three machine learning and deep learning models are employed for sentiment analysis:
  - **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
  - **Decision Tree**: A tree-based classifier for learning decision rules.
  - **LSTM (Long Short-Term Memory)**: A neural network model well-suited for analyzing sequential data.

### 5. Data Splitting
- Reviews are split into training and testing sets to evaluate model performance:
  - **80/20 split**: 80% training, 20% testing
  - **70/30 split**: 70% training, 30% testing
- Different combinations of feature extraction methods, models, and splits are tested to identify the most effective configuration.

### 6. Model Evaluation
- Models are evaluated on performance metrics such as accuracy, F1-score, precision, and recall to determine which model best classifies the sentiment of reviews.

## Project Structure
```
.
├── data/                   # Contains the scraped dataset (not provided for privacy)
├── notebooks/              # Jupyter notebooks for data preprocessing and analysis
├── models/                 # Scripts and trained models
├── src/
│   ├── data_collection.py  # Script for scraping reviews from Google Play Store
│   ├── preprocessing.py    # Script for text preprocessing and feature extraction
│   ├── train_model.py      # Script for training models
│   └── evaluate_model.py   # Script for model evaluation
├── README.md               # Project overview and instructions
└── requirements.txt        # Required libraries and dependencies
```

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sayid-alt/simcity-reviews-sentiment-analysis.git
   cd simcity-reviews-sentiment-analysis```
