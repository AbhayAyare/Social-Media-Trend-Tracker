#!/usr/bin/env python3
"""
Social Media Trend Tracker - Main Execution File
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add src to path
sys.path.append('./src')

from data_collection import DataCollector
from data_cleaning import DataCleaner
from nlp_analysis import NLPAnalyzer
from visualization import TrendVisualizer

def main():
    """Main execution function"""
    print("🚀 Starting Social Media Trend Tracker...")
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Data Collection
    print("\n" + "="*50)
    print("📊 STEP 1: Data Collection")
    print("="*50)
    
    collector = DataCollector("config/twitter_config.ini")
    
    # Collect Twitter data using Tweepy
    twitter_queries = [
        "AI OR ChatGPT OR OpenAI",
        "machine learning OR deep learning",
        "data science OR analytics"
    ]
    
    print("🐦 Collecting Twitter data using Tweepy...")
    twitter_df = collector.scrape_twitter_data(
        queries=twitter_queries,
        days_back=3,
        max_tweets_per_query=100
    )
    
    # Collect Reddit data  
    print("👾 Collecting Reddit data...")
    reddit_df = collector.scrape_reddit_data(
        subreddits=["technology", "MachineLearning"],
        days_back=3,
        max_posts=200
    )
    
    # Step 2: Data Cleaning
    print("\n" + "="*50)
    print("🧹 STEP 2: Data Cleaning")
    print("="*50)
    
    cleaner = DataCleaner()
    
    # Clean Twitter data
    if not twitter_df.empty:
        # Rename columns to match expected format
        if 'created_at' in twitter_df.columns:
            twitter_df = twitter_df.rename(columns={'created_at': 'Date', 'text': 'Content'})
        
        cleaned_twitter = cleaner.clean_twitter_data(twitter_df)
    else:
        print("❌ No Twitter data to clean")
        cleaned_twitter = pd.DataFrame()
    
    # Clean Reddit data
    if not reddit_df.empty:
        cleaned_reddit = cleaner.clean_reddit_data(reddit_df)
    else:
        print("❌ No Reddit data to clean") 
        cleaned_reddit = pd.DataFrame()
    
    # Save cleaned data
    cleaner.save_cleaned_data(
        twitter_df=cleaned_twitter,
        reddit_df=cleaned_reddit
    )
    
    # Step 3: NLP Analysis
    print("\n" + "="*50)
    print("🧠 STEP 3: NLP Analysis")
    print("="*50)
    
    analyzer = NLPAnalyzer()
    
    # Analyze Twitter data
    if not cleaned_twitter.empty:
        print("\n📊 Analyzing Twitter data...")
        
        # Sentiment analysis
        cleaned_twitter['Sentiment_Score'] = cleaned_twitter['Clean_Content'].apply(
            lambda x: analyzer.analyze_sentiment_vader(x)[0]
        )
        cleaned_twitter['Sentiment_Label'] = cleaned_twitter['Clean_Content'].apply(
            lambda x: analyzer.analyze_sentiment_vader(x)[1]
        )
        
        # Topic modeling
        texts = cleaned_twitter['Clean_Content'].tolist()
        if len(texts) > 10:  # Only if we have enough data
            topics, dominant_topics, model, vectorizer = analyzer.perform_topic_modeling_lda(
                texts, num_topics=5
            )
            cleaned_twitter['Dominant_Topic'] = dominant_topics
            
            # Map topic numbers to names
            topic_names = {i: f"Topic_{i+1}" for i in range(len(topics))}
            cleaned_twitter['Topic_Name'] = cleaned_twitter['Dominant_Topic'].map(topic_names)
    
    # Analyze Reddit data  
    if not cleaned_reddit.empty:
        print("\n📊 Analyzing Reddit data...")
        
        # Sentiment analysis
        cleaned_reddit['Sentiment_Score'] = cleaned_reddit['Clean_Text'].apply(
            lambda x: analyzer.analyze_sentiment_vader(x)[0]
        )
        cleaned_reddit['Sentiment_Label'] = cleaned_reddit['Clean_Text'].apply(
            lambda x: analyzer.analyze_sentiment_vader(x)[1]
        )
        
        # Topic modeling
        texts = cleaned_reddit['Clean_Text'].tolist()
        if len(texts) > 10:
            topics, dominant_topics, model, vectorizer = analyzer.perform_topic_modeling_lda(
                texts, num_topics=5
            )
            cleaned_reddit['Dominant_Topic'] = dominant_topics
            topic_names = {i: f"Topic_{i+1}" for i in range(len(topics))}
            cleaned_reddit['Topic_Name'] = cleaned_reddit['Dominant_Topic'].map(topic_names)
    
    # Step 4: Visualization
    print("\n" + "="*50)
    print("📈 STEP 4: Visualization")
    print("="*50)
    
    visualizer = TrendVisualizer(style='darkgrid')
    
    # Create visualizations for Twitter data
    if not cleaned_twitter.empty:
        print("\n📊 Creating Twitter visualizations...")
        
        # Sentiment timeline
        fig_timeline = visualizer.plot_sentiment_timeline(cleaned_twitter)
        fig_timeline.write_html("results/twitter_sentiment_timeline.html")
        
        # Sentiment distribution
        fig_distribution = visualizer.plot_sentiment_distribution(cleaned_twitter)
        fig_distribution.write_html("results/twitter_sentiment_distribution.html")
        
        # Engagement metrics (if available)
        if 'like_count' in cleaned_twitter.columns or 'retweet_count' in cleaned_twitter.columns:
            # Create engagement analysis
            engagement_cols = [col for col in ['like_count', 'retweet_count', 'reply_count'] if col in cleaned_twitter.columns]
            if engagement_cols:
                fig_engagement = visualizer.plot_engagement_metrics(cleaned_twitter, platform='twitter')
                fig_engagement.write_html("results/twitter_engagement.html")
        
        # Word cloud
        if not cleaned_twitter['Clean_Content'].empty:
            visualizer.create_word_cloud(
                cleaned_twitter['Clean_Content'],
                save_path="results/twitter_wordcloud.png"
            )
    
    # Create visualizations for Reddit data
    if not cleaned_reddit.empty:
        print("\n📊 Creating Reddit visualizations...")
        
        # Sentiment timeline
        fig_timeline = visualizer.plot_sentiment_timeline(cleaned_reddit)
        fig_timeline.write_html("results/reddit_sentiment_timeline.html")
        
        # Sentiment distribution  
        fig_distribution = visualizer.plot_sentiment_distribution(cleaned_reddit)
        fig_distribution.write_html("results/reddit_sentiment_distribution.html")
        
        # Word cloud
        if not cleaned_reddit['Clean_Text'].empty:
            visualizer.create_word_cloud(
                cleaned_reddit['Clean_Text'], 
                save_path="results/reddit_wordcloud.png"
            )
    
    print("\n" + "="*50)
    print("✅ ANALYSIS COMPLETE!")
    print("="*50)
    
    # Summary
    if not cleaned_twitter.empty:
        print(f"\n📊 Twitter Analysis Summary:")
        print(f"   - Total posts: {len(cleaned_twitter)}")
        print(f"   - Average sentiment: {cleaned_twitter['Sentiment_Score'].mean():.2f}")
        print(f"   - Positive posts: {(cleaned_twitter['Sentiment_Label'] == 'positive').mean()*100:.1f}%")
        
        # Engagement metrics if available
        if 'like_count' in cleaned_twitter.columns:
            avg_likes = cleaned_twitter['like_count'].mean()
            print(f"   - Average likes: {avg_likes:.1f}")
        
        if 'retweet_count' in cleaned_twitter.columns:
            avg_retweets = cleaned_twitter['retweet_count'].mean()
            print(f"   - Average retweets: {avg_retweets:.1f}")
    
    if not cleaned_reddit.empty:
        print(f"\n📊 Reddit Analysis Summary:")
        print(f"   - Total posts: {len(cleaned_reddit)}")
        print(f"   - Average sentiment: {cleaned_reddit['Sentiment_Score'].mean():.2f}")
        print(f"   - Positive posts: {(cleaned_reddit['Sentiment_Label'] == 'positive').mean()*100:.1f}%")
    
    print(f"\n📁 Results saved in 'results/' directory")
    print(f"🎛️  Run 'streamlit run dashboard/app.py' to start the dashboard")

if __name__ == "__main__":
    main()