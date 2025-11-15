import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from tqdm import tqdm
import os

class NLPAnalyzer:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        if pd.isna(text) or text == "":
            return 0.0, 'neutral'
        
        scores = self.sid.polarity_scores(str(text))
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return compound, sentiment
    
    def analyze_sentiment_transformers(self, texts, batch_size=32):
        """Analyze sentiment using transformers (more accurate but slower)"""
        print("🤖 Analyzing sentiment with transformers...")
        
        sentiments = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            batch = [str(text) for text in batch if str(text).strip()]
            
            if not batch:
                sentiments.extend([{'label': 'NEUTRAL', 'score': 0.5}] * len(batch))
                continue
                
            try:
                results = self.sentiment_pipeline(batch)
                sentiments.extend(results)
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                sentiments.extend([{'label': 'NEUTRAL', 'score': 0.5}] * len(batch))
        
        return sentiments
    
    def perform_topic_modeling_lda(self, texts, num_topics=5, max_features=1000):
        """Perform topic modeling using LDA"""
        print("🔍 Performing LDA topic modeling...")
        
        # Vectorize texts
        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        X = vectorizer.fit_transform(texts)
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            learning_method='online'
        )
        lda.fit(X)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Display topics
        topics = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_features = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topics[f"Topic_{topic_idx + 1}"] = top_features
            print(f"Topic {topic_idx + 1}: {', '.join(top_features)}")
        
        # Assign dominant topic to each document
        topic_distribution = lda.transform(X)
        dominant_topics = topic_distribution.argmax(axis=1)
        
        return topics, dominant_topics, lda, vectorizer
    
    def perform_topic_modeling_nmf(self, texts, num_topics=5, max_features=1000):
        """Perform topic modeling using NMF"""
        print("🔍 Performing NMF topic modeling...")
        
        # Vectorize texts with TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        X = vectorizer.fit_transform(texts)
        
        # Apply NMF
        nmf = NMF(
            n_components=num_topics,
            random_state=42,
            alpha=.1,
            l1_ratio=.5
        )
        nmf.fit(X)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Display topics
        topics = {}
        for topic_idx, topic in enumerate(nmf.components_):
            top_features = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topics[f"Topic_{topic_idx + 1}"] = top_features
            print(f"Topic {topic_idx + 1}: {', '.join(top_features)}")
        
        # Assign dominant topic to each document
        topic_distribution = nmf.transform(X)
        dominant_topics = topic_distribution.argmax(axis=1)
        
        return topics, dominant_topics, nmf, vectorizer
    
    def extract_key_phrases(self, texts, max_phrases=20):
        """Extract key phrases using frequency analysis"""
        print("📊 Extracting key phrases...")
        
        all_words = []
        for text in texts:
            if pd.isna(text):
                continue
            words = str(text).split()
            all_words.extend([word for word in words if len(word) > 3])
        
        word_freq = Counter(all_words)
        return word_freq.most_common(max_phrases)
    
    def analyze_emotions(self, texts):
        """Basic emotion analysis using keyword matching"""
        print("😊 Analyzing emotions...")
        
        emotion_keywords = {
            'joy': ['happy', 'excited', 'great', 'amazing', 'wonderful', 'love', 'best'],
            'anger': ['angry', 'mad', 'hate', 'terrible', 'awful', 'horrible', 'stupid'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'sorry', 'bad'],
            'fear': ['scared', 'afraid', 'worried', 'nervous', 'anxious', 'fear'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected', 'wow']
        }
        
        emotions = []
        for text in texts:
            if pd.isna(text):
                emotions.append('neutral')
                continue
                
            text_lower = str(text).lower()
            emotion_scores = {emotion: 0 for emotion in emotion_keywords}
            
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        emotion_scores[emotion] += 1
            
            # Get emotion with highest score
            if sum(emotion_scores.values()) == 0:
                emotions.append('neutral')
            else:
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                emotions.append(dominant_emotion)
        
        return emotions
    
    def create_word_cloud(self, texts, save_path=None):
        """Create word cloud from texts"""
        print("☁️ Creating word cloud...")
        
        all_text = ' '.join([str(text) for text in texts if pd.notna(text)])
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Social Media Content', fontsize=16, pad=20)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"💾 Word cloud saved to {save_path}")
        
        plt.show()
        return wordcloud

# Example usage
if __name__ == "__main__":
    analyzer = NLPAnalyzer()
    
    # Sample data
    sample_texts = [
        "I love this new AI technology! It's amazing!",
        "This is terrible and I hate it.",
        "The weather is nice today.",
        "Machine learning is transforming industries.",
        "I'm worried about privacy issues with AI."
    ]
    
    # Analyze sentiment
    for text in sample_texts:
        score, sentiment = analyzer.analyze_sentiment_vader(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment} (Score: {score:.2f})\n")
    
    # Topic modeling
    topics, dominant_topics, model, vectorizer = analyzer.perform_topic_modeling_lda(sample_texts, num_topics=2)
    
    # Key phrases
    key_phrases = analyzer.extract_key_phrases(sample_texts)
    print("Key phrases:", key_phrases)