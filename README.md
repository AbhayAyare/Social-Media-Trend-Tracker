# 📊 Social Media Trend Tracker

A comprehensive data analytics project that analyzes social media trends, sentiment, and emerging topics using **NLP** and **machine learning**. Track Twitter and Reddit data to gain real-time insights into public opinion and trending topics.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Twitter API](https://img.shields.io/badge/Twitter-API%20v2-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![NLP](https://img.shields.io/badge/Analysis-NLP%2BML-orange)

🎥 Video Demonstration
Watch the complete project walkthrough and setup guide:

Recording 2025-11-14 235757.mp4
Click above to watch the complete project demonstration

## 🎯 Project Overview

This project tracks and analyzes social media trends from Twitter and Reddit to:

- 🔍 **Identify emerging topics** and themes in real-time
- ❤️ **Analyze public sentiment** and emotional trends over time  
- 📈 **Visualize trend evolution** with interactive dashboards
- 🚨 **Detect viral content** and predict trending topics
- 📊 **Provide actionable insights** for businesses and researchers

## 🏗️ Project Architecture

```
Data Sources → Collection → Processing → Analysis → Visualization → Insights
    ↓            ↓           ↓           ↓           ↓             ↓
  Twitter     Tweepy     Cleaning    Sentiment   Streamlit    Trend Reports
   Reddit     API/PS     Pipeline    Topic ML    Dashboard    Alert System
```

## 📁 Project Structure

```
social_media_trend_tracker/
│
├── data/                          # Data storage
│   ├── raw/                       # Raw collected data
│   └── processed/                 # Cleaned and processed data
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_nlp_analysis.ipynb
│   └── 04_visualization.ipynb
│
├── src/                           # Source code modules
│   ├── data_collection.py        # Twitter/Reddit data collection
│   ├── data_cleaning.py          # Text preprocessing pipeline
│   ├── nlp_analysis.py           # Sentiment & topic modeling
│   └── visualization.py          # Plotting and charts
│
├── dashboard/                     # Interactive web dashboard
│   ├── app.py                    # Streamlit main application
│   └── requirements.txt          # Dashboard dependencies
│
├── config/                        # Configuration files
│   ├── settings.py               # Project settings
│   └── twitter_config.ini        # API credentials template
│
├── results/                       # Generated outputs
│   ├── charts/                   # Saved visualizations
│   └── reports/                  # Analysis reports
│
├── main.py                       # Main execution script
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Twitter Developer Account (for API access)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/social-media-trend-tracker.git
cd social-media-trend-tracker
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment (recommended)
python -m venv trend_env

# Activate environment
# On Windows:
trend_env\Scripts\activate
# On Mac/Linux:
source trend_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Twitter API

1. **Get Twitter API Credentials:**
   - Apply at [developer.twitter.com](https://developer.twitter.com)
   - Create a new Project and App
   - Generate API keys

2. **Set Up Configuration:**
   ```bash
   # Edit the config file
   nano config/twitter_config.ini
   ```

3. **Add your credentials:**
   ```ini
   [twitter]
   # For API v2 (Recommended)
   bearer_token = your_bearer_token_here
   
   # OR for API v1.1 (Alternative)
   # consumer_key = your_consumer_key
   # consumer_secret = your_consumer_secret  
   # access_token = your_access_token
   # access_token_secret = your_access_token_secret
   ```

### Step 4: Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Step 5: Run the Project

#### Option A: Complete Analysis Pipeline
```bash
python main.py
```

#### Option B: Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

#### Option C: Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## 🛠️ Features

### 📊 Data Collection
- **Twitter API Integration**: Official Tweepy integration
- **Reddit Scraping**: Pushshift API for Reddit posts
- **Real-time Streaming**: Live data collection
- **Historical Data**: Backfill historical trends

### 🧹 Data Processing
- **Text Cleaning**: URL removal, tokenization, lemmatization
- **Noise Reduction**: Stopword removal, special character handling
- **Feature Extraction**: Hashtag analysis, mention tracking
- **Quality Metrics**: Text statistics and quality assessment

### 🧠 NLP & Machine Learning
- **Sentiment Analysis**: VADER and Transformers-based sentiment detection
- **Topic Modeling**: LDA and NMF for theme discovery
- **Emotion Detection**: Joy, anger, sadness, fear, surprise classification
- **Trend Detection**: Identify emerging topics

### 📈 Visualization
- **Interactive Dashboards**: Real-time Streamlit interface
- **Time Series Analysis**: Sentiment trends over time
- **Topic Evolution**: Track theme popularity changes
- **Word Clouds**: Visual representation of frequent terms
- **Engagement Metrics**: Likes, retweets, comments analysis

## 🎮 Usage Examples

### Basic Twitter Analysis
```python
from src.data_collection import DataCollector
from src.nlp_analysis import NLPAnalyzer

# Collect data
collector = DataCollector()
tweets = collector.scrape_twitter_data(
    queries=["AI", "machine learning"],
    days_back=7,
    max_tweets_per_query=200
)

# Analyze sentiment
analyzer = NLPAnalyzer()
sentiment_results = analyzer.analyze_sentiment_vader(tweets['text'])
```

### Custom Dashboard
Modify `dashboard/app.py` to add custom filters and visualizations.

## 📊 Sample Outputs

### Generated Visualizations
- **Sentiment Timeline**: Daily sentiment scores over time
- **Topic Distribution**: Pie charts of theme prevalence
- **Engagement Heatmaps**: Activity patterns by time and topic
- **Word Clouds**: Visual frequency analysis
- **Correlation Matrices**: Relationship between metrics

### Analysis Reports
- Weekly trend summaries
- Sentiment analysis reports
- Topic evolution tracking
- Engagement insights

## 🔧 Configuration

### Search Queries
Edit `config/settings.py` to modify search terms:
```python
TWITTER_QUERIES = [
    "AI OR ChatGPT OR OpenAI",
    "machine learning OR deep learning", 
    "data science OR analytics",
    "tech OR technology",
    "programming OR coding"
]
```

### Analysis Parameters
```python
NLP_SETTINGS = {
    'sentiment_threshold': 0.05,
    'num_topics': 5,
    'max_features': 1000
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Twitter API Limits**
   ```bash
   # Error: Rate limit exceeded
   # Solution: Wait 15 minutes or upgrade API tier
   ```

2. **Missing Dependencies**
   ```bash
   # Error: Module not found
   pip install -r requirements.txt --upgrade
   ```

3. **Configuration Issues**
   ```bash
   # Error: Invalid API credentials
   # Solution: Verify credentials in config/twitter_config.ini
   ```




