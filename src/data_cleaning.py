import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from textblob import TextBlob
import os

class DataCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.processed_dir = "data/processed"
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Extended stop words
        self.extra_stop_words = {
            'http', 'https', 'com', 'www', 'html', 'like', 'get', 'would',
            'one', 'us', 'could', 'also', 'may', 'might', 'since', 'upon'
        }
        self.stop_words.update(self.extra_stop_words)
    
    def load_data(self, twitter_path=None, reddit_path=None):
        """Load raw data from CSV files"""
        data = {}
        
        if twitter_path and os.path.exists(twitter_path):
            data['twitter'] = pd.read_csv(twitter_path, parse_dates=['Date'])
            print(f"✅ Loaded {len(data['twitter'])} tweets")
        
        if reddit_path and os.path.exists(reddit_path):
            data['reddit'] = pd.read_csv(reddit_path, parse_dates=['Date'])
            print(f"✅ Loaded {len(data['reddit'])} Reddit posts")
        
        return data
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (but keep the text)
        text = re.sub(r'@\w+|#', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens, then lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(cleaned_tokens)
    
    def extract_hashtags(self, text):
        """Extract hashtags from text"""
        if pd.isna(text):
            return []
        hashtags = re.findall(r'#(\w+)', str(text))
        return [tag.lower() for tag in hashtags]
    
    def calculate_text_metrics(self, text):
        """Calculate various text metrics"""
        if pd.isna(text) or text == "":
            return 0, 0, 0
        
        blob = TextBlob(str(text))
        word_count = len(str(text).split())
        char_count = len(str(text))
        sentence_count = len(str(text).split('.'))
        
        return word_count, char_count, sentence_count
    
    def clean_twitter_data(self, df):
        """Clean Twitter-specific data"""
        print("🧹 Cleaning Twitter data...")
        
        # Basic cleaning
        df_clean = df.copy()
        df_clean = df_clean.drop_duplicates(subset=['Content'])
        df_clean = df_clean[df_clean['Content'].notna()]
        
        # Text cleaning
        df_clean['Clean_Content'] = df_clean['Content'].apply(self.clean_text)
        
        # Extract hashtags
        df_clean['Extracted_Hashtags'] = df_clean['Content'].apply(self.extract_hashtags)
        
        # Calculate metrics
        metrics = df_clean['Content'].apply(
            lambda x: pd.Series(self.calculate_text_metrics(x), 
                              index=['Word_Count', 'Char_Count', 'Sentence_Count'])
        )
        df_clean = pd.concat([df_clean, metrics], axis=1)
        
        # Filter out very short texts
        df_clean = df_clean[df_clean['Word_Count'] >= 3]
        
        print(f"✅ Cleaned {len(df_clean)} tweets")
        return df_clean
    
    def clean_reddit_data(self, df):
        """Clean Reddit-specific data"""
        print("🧹 Cleaning Reddit data...")
        
        # Basic cleaning
        df_clean = df.copy()
        df_clean = df_clean.drop_duplicates(subset=['Title', 'Content'])
        df_clean = df_clean[df_clean['Title'].notna()]
        
        # Combine title and content for analysis
        df_clean['Full_Text'] = df_clean['Title'] + ' ' + df_clean['Content'].fillna('')
        
        # Text cleaning
        df_clean['Clean_Text'] = df_clean['Full_Text'].apply(self.clean_text)
        
        # Calculate metrics
        metrics = df_clean['Full_Text'].apply(
            lambda x: pd.Series(self.calculate_text_metrics(x), 
                              index=['Word_Count', 'Char_Count', 'Sentence_Count'])
        )
        df_clean = pd.concat([df_clean, metrics], axis=1)
        
        # Filter out very short texts
        df_clean = df_clean[df_clean['Word_Count'] >= 5]
        
        print(f"✅ Cleaned {len(df_clean)} Reddit posts")
        return df_clean
    
    def save_cleaned_data(self, twitter_df=None, reddit_df=None):
        """Save cleaned data to processed directory"""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        if twitter_df is not None:
            twitter_path = f"{self.processed_dir}/cleaned_twitter_{timestamp}.csv"
            twitter_df.to_csv(twitter_path, index=False)
            print(f"💾 Saved cleaned Twitter data to {twitter_path}")
        
        if reddit_df is not None:
            reddit_path = f"{self.processed_dir}/cleaned_reddit_{timestamp}.csv"
            reddit_df.to_csv(reddit_path, index=False)
            print(f"💾 Saved cleaned Reddit data to {reddit_path}")

# Example usage
if __name__ == "__main__":
    cleaner = DataCleaner()
    
    # Load and clean data
    data = cleaner.load_data(
        twitter_path="data/raw/tweets_20241219.csv",
        reddit_path="data/raw/reddit_posts_20241219.csv"
    )
    
    if 'twitter' in data:
        cleaned_twitter = cleaner.clean_twitter_data(data['twitter'])
        cleaner.save_cleaned_data(twitter_df=cleaned_twitter)
    
    if 'reddit' in data:
        cleaned_reddit = cleaner.clean_reddit_data(data['reddit'])
        cleaner.save_cleaned_data(reddit_df=cleaned_reddit)