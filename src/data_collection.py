import tweepy
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import os
import configparser

class DataCollector:
    def __init__(self, config_file="config/twitter_config.ini"):
        self.data_dir = "data/raw"
        os.makedirs(self.data_dir, exist_ok=True)
        self.setup_twitter_api(config_file)
    
    def setup_twitter_api(self, config_file):
        """Set up Twitter API credentials"""
        print("🔧 Setting up Twitter API...")
        
        # Create config file if it doesn't exist
        if not os.path.exists(config_file):
            self.create_sample_config(config_file)
            print(f"⚠️  Please fill in your Twitter API credentials in {config_file}")
            self.api = None
            return
        
        try:
            config = configparser.ConfigParser()
            config.read(config_file)
            
            # Twitter API v2 credentials
            bearer_token = config.get('twitter', 'bearer_token', fallback=None)
            
            if bearer_token:
                # For Twitter API v2
                self.client = tweepy.Client(bearer_token=bearer_token)
                print("✅ Twitter API v2 client initialized")
            else:
                # For Twitter API v1.1 (fallback)
                consumer_key = config.get('twitter', 'consumer_key')
                consumer_secret = config.get('twitter', 'consumer_secret')
                access_token = config.get('twitter', 'access_token')
                access_token_secret = config.get('twitter', 'access_token_secret')
                
                auth = tweepy.OAuth1UserHandler(
                    consumer_key, consumer_secret,
                    access_token, access_token_secret
                )
                self.api = tweepy.API(auth, wait_on_rate_limit=True)
                print("✅ Twitter API v1.1 initialized")
                
        except Exception as e:
            print(f"❌ Error setting up Twitter API: {e}")
            self.api = None
            self.client = None
    
    def create_sample_config(self, config_file):
        """Create a sample configuration file"""
        config = configparser.ConfigParser()
        
        config['twitter'] = {
            'bearer_token': 'your_bearer_token_here (for API v2)',
            'consumer_key': 'your_consumer_key_here',
            'consumer_secret': 'your_consumer_secret_here',
            'access_token': 'your_access_token_here',
            'access_token_secret': 'your_access_token_secret_here',
            'note': 'Use either bearer_token (v2) OR the four keys (v1.1)'
        }
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            config.write(f)
    
    def search_tweets_v2(self, query, max_tweets=100, days_back=7):
        """Search tweets using Twitter API v2"""
        if not self.client:
            print("❌ Twitter API v2 client not initialized")
            return []
        
        print(f"🔍 Searching tweets with query: {query}")
        
        # Calculate date range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        tweets = []
        next_token = None
        tweet_count = 0
        
        try:
            while tweet_count < max_tweets:
                # API v2 search
                response = self.client.search_recent_tweets(
                    query=query,
                    max_results=min(100, max_tweets - tweet_count),
                    start_time=start_time,
                    end_time=end_time,
                    tweet_fields=['created_at', 'public_metrics', 'author_id', 'context_annotations', 'entities'],
                    user_fields=['username', 'name', 'public_metrics'],
                    expansions='author_id',
                    next_token=next_token
                )
                
                if not response.data:
                    break
                
                # Create user lookup dictionary
                users = {user.id: user for user in response.includes.get('users', [])}
                
                for tweet in response.data:
                    user = users.get(tweet.author_id)
                    tweets.append({
                        'id': tweet.id,
                        'created_at': tweet.created_at,
                        'text': tweet.text,
                        'author_id': tweet.author_id,
                        'username': user.username if user else 'Unknown',
                        'author_name': user.name if user else 'Unknown',
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'quote_count': tweet.public_metrics['quote_count'],
                        'impression_count': tweet.public_metrics['impression_count'],
                        'hashtags': [tag['tag'] for tag in tweet.entities.get('hashtags', [])] if hasattr(tweet, 'entities') and tweet.entities else [],
                        'mentions': [mention['username'] for mention in tweet.entities.get('mentions', [])] if hasattr(tweet, 'entities') and tweet.entities else [],
                        'urls': [url['expanded_url'] for url in tweet.entities.get('urls', [])] if hasattr(tweet, 'entities') and tweet.entities else []
                    })
                    
                    tweet_count += 1
                    if tweet_count >= max_tweets:
                        break
                
                # Check if there are more results
                next_token = response.meta.get('next_token')
                if not next_token:
                    break
                    
                # Rate limiting respect
                time.sleep(1)
                
        except tweepy.TooManyRequests:
            print("⏳ Rate limit exceeded. Waiting for 15 minutes...")
            time.sleep(15 * 60)
        except Exception as e:
            print(f"❌ Error searching tweets: {e}")
        
        return tweets
    
    def search_tweets_v1(self, query, max_tweets=100, days_back=7):
        """Search tweets using Twitter API v1.1"""
        if not self.api:
            print("❌ Twitter API v1.1 not initialized")
            return []
        
        print(f"🔍 Searching tweets with query: {query}")
        
        tweets = []
        
        try:
            for tweet in tweepy.Cursor(
                self.api.search_tweets,
                q=query,
                lang='en',
                tweet_mode='extended',
                count=min(100, max_tweets)
            ).items(max_tweets):
                
                tweets.append({
                    'id': tweet.id,
                    'created_at': tweet.created_at,
                    'text': tweet.full_text,
                    'username': tweet.user.screen_name,
                    'author_name': tweet.user.name,
                    'retweet_count': tweet.retweet_count,
                    'like_count': tweet.favorite_count,
                    'reply_count': 0,  # Not directly available in v1.1
                    'quote_count': 0,  # Not directly available in v1.1
                    'impression_count': 0,  # Not available in v1.1
                    'hashtags': [hashtag['text'] for hashtag in tweet.entities.get('hashtags', [])],
                    'mentions': [mention['screen_name'] for mention in tweet.entities.get('user_mentions', [])],
                    'urls': [url['expanded_url'] for url in tweet.entities.get('urls', [])],
                    'user_followers': tweet.user.followers_count,
                    'user_friends': tweet.user.friends_count,
                    'user_statuses': tweet.user.statuses_count
                })
                
        except tweepy.RateLimitError:
            print("⏳ Rate limit exceeded. Please wait...")
        except Exception as e:
            print(f"❌ Error searching tweets: {e}")
        
        return tweets
    
    def scrape_twitter_data(self, queries, days_back=7, max_tweets_per_query=200):
        """Scrape tweets based on queries using Tweepy"""
        print("🔍 Collecting Twitter data using Tweepy...")
        
        all_tweets = []
        
        for query in queries:
            print(f"📝 Searching for: {query}")
            
            # Try API v2 first, then fall back to v1.1
            if hasattr(self, 'client') and self.client:
                tweets = self.search_tweets_v2(query, max_tweets_per_query, days_back)
            elif hasattr(self, 'api') and self.api:
                tweets = self.search_tweets_v1(query, max_tweets_per_query, days_back)
            else:
                print("❌ No Twitter API configured")
                continue
            
            if tweets:
                query_tweets_df = pd.DataFrame(tweets)
                query_tweets_df['search_query'] = query
                all_tweets.append(query_tweets_df)
                print(f"✅ Found {len(tweets)} tweets for query: {query}")
            else:
                print(f"❌ No tweets found for query: {query}")
            
            # Be respectful to API limits
            time.sleep(2)
        
        if all_tweets:
            final_df = pd.concat(all_tweets, ignore_index=True)
            
            # Convert created_at to datetime if it's string
            if 'created_at' in final_df.columns:
                final_df['created_at'] = pd.to_datetime(final_df['created_at'])
            
            filename = f"{self.data_dir}/tweets_tweepy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            final_df.to_csv(filename, index=False)
            print(f"💾 Saved {len(final_df)} tweets to {filename}")
            return final_df
        else:
            print("❌ No tweets collected")
            return pd.DataFrame()
    
    def get_trending_topics(self, woied=1):
        """Get trending topics for a specific location (WOEID)"""
        if not hasattr(self, 'api') or not self.api:
            print("❌ Twitter API v1.1 required for trending topics")
            return []
        
        try:
            trends = self.api.get_place_trends(woied)
            trending_list = []
            
            for trend in trends[0]['trends'][:20]:  # Top 20 trends
                trending_list.append({
                    'name': trend['name'],
                    'url': trend['url'],
                    'tweet_volume': trend.get('tweet_volume', 0),
                    'query': trend['query']
                })
            
            return trending_list
            
        except Exception as e:
            print(f"❌ Error getting trending topics: {e}")
            return []
    
    def get_user_timeline(self, username, count=100):
        """Get tweets from a specific user's timeline"""
        if not hasattr(self, 'api') or not self.api:
            print("❌ Twitter API required for user timeline")
            return []
        
        try:
            tweets = []
            for tweet in tweepy.Cursor(
                self.api.user_timeline,
                screen_name=username,
                count=count,
                tweet_mode='extended'
            ).items(count):
                
                tweets.append({
                    'id': tweet.id,
                    'created_at': tweet.created_at,
                    'text': tweet.full_text,
                    'retweet_count': tweet.retweet_count,
                    'like_count': tweet.favorite_count,
                    'is_retweet': hasattr(tweet, 'retweeted_status')
                })
            
            return tweets
            
        except Exception as e:
            print(f"❌ Error getting user timeline: {e}")
            return []
    
    def scrape_reddit_data(self, subreddits, days_back=7, max_posts=500):
        """Scrape Reddit posts from specified subreddits"""
        print("🔍 Collecting Reddit data...")
        
        all_posts = []
        
        for subreddit in subreddits:
            print(f"Scraping r/{subreddit}...")
            
            try:
                # Using Pushshift API
                url = "https://api.pushshift.io/reddit/search/submission/"
                params = {
                    'subreddit': subreddit,
                    'size': min(100, max_posts),
                    'after': f'{days_back}d',
                    'sort_type': 'created_utc',
                    'sort': 'desc'
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()['data']
                    
                    for post in data:
                        all_posts.append([
                            datetime.fromtimestamp(post['created_utc']),
                            post.get('title', ''),
                            post.get('selftext', ''),
                            post.get('subreddit', ''),
                            post.get('score', 0),
                            post.get('num_comments', 0),
                            post.get('upvote_ratio', 0),
                            post.get('author', ''),
                            post.get('url', ''),
                            post.get('permalink', '')
                        ])
                
                time.sleep(1)  # Be respectful to the API
                
            except Exception as e:
                print(f"Error scraping r/{subreddit}: {e}")
                continue
        
        if all_posts:
            reddit_df = pd.DataFrame(all_posts, columns=[
                'Date', 'Title', 'Content', 'Subreddit', 'Score', 
                'Comments', 'Upvote_Ratio', 'Author', 'URL', 'Permalink'
            ])
            filename = f"{self.data_dir}/reddit_posts_{datetime.now().strftime('%Y%m%d')}.csv"
            reddit_df.to_csv(filename, index=False)
            print(f"✅ Saved {len(reddit_df)} Reddit posts to {filename}")
            return reddit_df
        else:
            print("❌ No Reddit posts collected")
            return pd.DataFrame()

# Example usage and testing
if __name__ == "__main__":
    collector = DataCollector()
    
    # Test Twitter data collection
    if hasattr(collector, 'client') or hasattr(collector, 'api'):
        twitter_df = collector.scrape_twitter_data(
            queries=["AI OR ChatGPT", "machine learning", "data science"],
            days_back=3,
            max_tweets_per_query=50
        )
        
        # Test trending topics (if API v1.1 is available)
        if hasattr(collector, 'api') and collector.api:
            trends = collector.get_trending_topics(woied=1)  # Worldwide trends
            if trends:
                print("\n🔥 Trending Topics:")
                for i, trend in enumerate(trends[:10], 1):
                    volume = f" ({trend['tweet_volume']} tweets)" if trend['tweet_volume'] else ""
                    print(f"{i}. {trend['name']}{volume}")
    
    # Test Reddit data collection
    reddit_df = collector.scrape_reddit_data(
        subreddits=["technology", "MachineLearning"],
        days_back=3,
        max_posts=100
    )