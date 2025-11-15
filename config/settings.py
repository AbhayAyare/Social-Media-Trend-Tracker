# Configuration settings for the project

# Data Collection Settings
TWITTER_QUERIES = [
    "AI OR ChatGPT OR OpenAI",
    "machine learning OR deep learning",
    "data science OR analytics",
    "tech OR technology",
    "programming OR coding"
]

REDDIT_SUBREDDITS = [
    "technology",
    "artificial",
    "MachineLearning",
    "datascience",
    "programming"
]

# NLP Settings
NLTK_PACKAGES = [
    'stopwords',
    'vader_lexicon',
    'punkt',
    'wordnet',
    'averaged_perceptron_tagger'
]

# Model Settings
TOPIC_MODELING = {
    'num_topics': 5,
    'random_state': 42
}

SENTIMENT_ANALYSIS = {
    'positive_threshold': 0.05,
    'negative_threshold': -0.05
}

# Visualization Settings
COLORS = {
    'positive': '#2E8B57',
    'negative': '#DC143C',
    'neutral': '#4682B4',
    'background': '#F5F5F5'
}