# Sentiment Analysis of SimCity BuildIt App Reviews

## Project Overview
This project aims to analyze the sentiment of user reviews for the *SimCity BuildIt* app on the Google Play Store. By categorizing reviews as either positive or negative, the analysis provides insights into user satisfaction and common themes in feedback. The study employs both machine learning and deep learning approaches to enhance sentiment classification accuracy.

## Data Collection
- The dataset consists of user reviews scraped from the Google Play Store using the `google-play-scraper` Python library.
- The dataset is stored in CSV format, containing textual review data along with relevant metadata.

## Methodology

### Sentiment Classification Models
The analysis leverages three different classification models:
1. **Support Vector Machine (SVM)**
2. **Random Forest Classifier (RFC)**
3. **Long Short-Term Memory (LSTM) Neural Network**

### Feature Extraction Techniques
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Converts textual data into numerical representations based on word importance.
- **Word2Vec Word Embeddings**: Captures semantic relationships between words for deep learning applications.

### Data Splitting Strategy
To evaluate model performance, the dataset is divided into training and testing sets using different split ratios:
- 80% training / 20% testing
- 70% training / 30% testing

### Model Training Configurations
Multiple configurations are tested to determine optimal performance:
- `Random Forest` with `TF-IDF` (80/20 split)
- `Gradient Boosting` with `TF-IDF` (70/30 split)
- `LSTM` with `Word2Vec` (80/20 split)

## Implementation Steps
### 1. Importing Libraries
The project utilizes several Python libraries, including:
- `numpy`, `pandas`, `matplotlib`, `seaborn` for data manipulation and visualization.
- `nltk`, `gensim`, `sklearn` for text preprocessing and feature extraction.
- `tensorflow` and `keras` for deep learning model implementation.

### 2. Data Preprocessing
- Tokenization using `nltk`
- Stopword removal
- Text normalization (removing punctuation, converting to lowercase)
- Vectorization using TF-IDF and Word2Vec

### 3. Model Training and Evaluation
- Machine learning models are trained using `sklearn`
- LSTM network is built and trained using `tensorflow.keras`
- Model evaluation metrics include `accuracy_score`

## Expected Outcomes
The project aims to:
- Identify common positive and negative themes in user reviews.
- Provide insights that can help improve app development and user satisfaction.
- Compare machine learning and deep learning approaches for sentiment analysis.


## Model Performances
## Model Training Results

| Model                         | Training Accuracy | Testing Accuracy | Time Consumed |
|--------------------------------|------------------|------------------|--------------|
| Random Forest (TF-IDF, 80/20)  | 99.7%            | 93.6%            | 216.463 sec  |
| Gradient Boosting (TF-IDF, 70/30) | 94.2%        | 92.8%            | 135.819 sec  |
| LSTM (Word2Vec, 80/20)         | 98.3%            | 98.24% (val)     | N/A          |

<img src="https://raw.githubusercontent.com/sayid-alt/simcity-reviews-sentiment-analysis/refs/heads/main/img/training-chart.jpeg" />

## Conclusion
This sentiment analysis study helps in understanding user feedback trends for *SimCity BuildIt*, offering valuable insights to developers and stakeholders. Future work can include fine-tuning deep learning architectures or incorporating additional NLP techniques for enhanced accuracy.

