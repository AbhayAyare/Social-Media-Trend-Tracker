import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import TrendVisualizer
from src.nlp_analysis import NLPAnalyzer

# Page configuration
st.set_page_config(
    page_title="Social Media Trend Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
    }
    .positive { color: #2E8B57; }
    .negative { color: #DC143C; }
    .neutral { color: #4682B4; }
</style>
""", unsafe_allow_html=True)

class TrendTrackerDashboard:
    def __init__(self):
        self.visualizer = TrendVisualizer(style='darkgrid')
        self.analyzer = NLPAnalyzer()
        
    def load_sample_data(self):
        """Load sample data for demonstration"""
        # Generate sample data
        dates = pd.date_range('2024-01-01', periods=500, freq='H')
        topics = ['AI', 'Machine Learning', 'Data Science', 'Technology', 'Innovation']
        sentiments = ['positive', 'negative', 'neutral']
        
        sample_data = pd.DataFrame({
            'Date': dates,
            'Content': [f"Sample post about {topic}" for topic in np.random.choice(topics, 500)],
            'Platform': np.random.choice(['Twitter', 'Reddit'], 500),
            'Sentiment_Score': np.random.normal(0, 0.3, 500),
            'Sentiment_Label': np.random.choice(sentiments, 500, p=[0.4, 0.2, 0.4]),
            'Dominant_Topic': np.random.choice(topics, 500),
            'Likes': np.random.poisson(50, 500),
            'Retweets': np.random.poisson(10, 500),
            'Engagement': np.random.poisson(75, 500)
        })
        
        return sample_data
    
    def render_sidebar(self):
        """Render the sidebar with filters and controls"""
        st.sidebar.title("🔧 Dashboard Controls")
        
        st.sidebar.markdown("### Data Filters")
        
        # Date range filter
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-01-07'))
        )
        
        # Platform filter
        platforms = st.sidebar.multiselect(
            "Platforms",
            options=['Twitter', 'Reddit', 'Both'],
            default=['Both']
        )
        
        # Sentiment filter
        sentiments = st.sidebar.multiselect(
            "Sentiments",
            options=['positive', 'negative', 'neutral'],
            default=['positive', 'negative', 'neutral']
        )
        
        # Topic filter
        topics = st.sidebar.multiselect(
            "Topics",
            options=['AI', 'Machine Learning', 'Data Science', 'Technology', 'Innovation'],
            default=['AI', 'Machine Learning', 'Data Science', 'Technology', 'Innovation']
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Analysis Settings")
        
        # Number of topics for modeling
        num_topics = st.sidebar.slider(
            "Number of Topics",
            min_value=2,
            max_value=10,
            value=5
        )
        
        return {
            'date_range': date_range,
            'platforms': platforms,
            'sentiments': sentiments,
            'topics': topics,
            'num_topics': num_topics
        }
    
    def render_header_metrics(self, df):
        """Render header metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_posts = len(df)
            st.metric("Total Posts", f"{total_posts:,}")
        
        with col2:
            avg_sentiment = df['Sentiment_Score'].mean()
            sentiment_color = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"
            st.metric("Avg Sentiment", f"{avg_sentiment:.2f}", delta_color="off")
        
        with col3:
            engagement_rate = df['Engagement'].mean()
            st.metric("Avg Engagement", f"{engagement_rate:.1f}")
        
        with col4:
            positive_ratio = (df['Sentiment_Label'] == 'positive').mean() * 100
            st.metric("Positive Posts", f"{positive_ratio:.1f}%")
    
    def render_sentiment_analysis(self, df):
        """Render sentiment analysis section"""
        st.header("📈 Sentiment Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sentiment timeline
            fig_timeline = self.visualizer.plot_sentiment_timeline(df)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            # Sentiment distribution
            fig_distribution = self.visualizer.plot_sentiment_distribution(df)
            st.plotly_chart(fig_distribution, use_container_width=True)
            
            # Sentiment by platform
            if 'Platform' in df.columns:
                sentiment_by_platform = pd.crosstab(df['Platform'], df['Sentiment_Label'], normalize='index') * 100
                fig_platform = px.bar(
                    sentiment_by_platform, 
                    barmode='stack',
                    title='Sentiment by Platform',
                    color_discrete_map={
                        'positive': '#2E8B57',
                        'negative': '#DC143C', 
                        'neutral': '#4682B4'
                    }
                )
                st.plotly_chart(fig_platform, use_container_width=True)
    
    def render_topic_analysis(self, df):
        """Render topic analysis section"""
        st.header("🧠 Topic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Topic evolution
            fig_evolution = self.visualizer.plot_topic_evolution(df)
            st.plotly_chart(fig_evolution, use_container_width=True)
        
        with col2:
            # Topic distribution
            topic_counts = df['Dominant_Topic'].value_counts()
            fig_topics = px.pie(
                values=topic_counts.values,
                names=topic_counts.index,
                title='Topic Distribution'
            )
            st.plotly_chart(fig_topics, use_container_width=True)
        
        # Word cloud
        st.subheader("Word Cloud")
        if not df['Content'].empty:
            fig_wordcloud = self.visualizer.create_word_cloud_plotly(df['Content'])
            st.plotly_chart(fig_wordcloud, use_container_width=True)
    
    def render_engagement_analysis(self, df):
        """Render engagement analysis section"""
        st.header("💫 Engagement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Engagement metrics
            fig_engagement = self.visualizer.plot_engagement_metrics(df, platform='twitter')
            if fig_engagement:
                st.plotly_chart(fig_engagement, use_container_width=True)
        
        with col2:
            # Correlation heatmap
            fig_correlation = self.visualizer.plot_correlation_heatmap(df)
            if fig_correlation:
                st.plotly_chart(fig_correlation, use_container_width=True)
    
    def render_trend_insights(self, df):
        """Render trend insights section"""
        st.header("🔍 Trend Insights")
        
        # Calculate insights
        insights = []
        
        # Top trending topics
        top_topics = df['Dominant_Topic'].value_counts().head(3)
        insights.append(f"**Top Trending Topics**: {', '.join(top_topics.index)}")
        
        # Sentiment trends
        positive_trend = (df['Sentiment_Label'] == 'positive').mean() * 100
        insights.append(f"**Positive Sentiment**: {positive_trend:.1f}% of posts")
        
        # Engagement trends
        avg_engagement = df['Engagement'].mean()
        insights.append(f"**Average Engagement**: {avg_engagement:.1f} per post")
        
        # Display insights
        for insight in insights:
            st.info(insight)
        
        # Recent posts sample
        st.subheader("Recent Posts Sample")
        recent_posts = df[['Date', 'Content', 'Sentiment_Label', 'Dominant_Topic']].head(10)
        st.dataframe(recent_posts, use_container_width=True)
    
    def run(self):
        """Run the dashboard"""
        # Header
        st.markdown('<h1 class="main-header">📊 Social Media Trend Tracker</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        filters = self.render_sidebar()
        
        # Load data (in real scenario, this would be from your processed data)
        df = self.load_sample_data()
        
        # Apply filters (simplified for demo)
        filtered_df = df.copy()
        
        # Render dashboard sections
        self.render_header_metrics(filtered_df)
        
        st.markdown("---")
        
        self.render_sentiment_analysis(filtered_df)
        
        st.markdown("---")
        
        self.render_topic_analysis(filtered_df)
        
        st.markdown("---")
        
        self.render_engagement_analysis(filtered_df)
        
        st.markdown("---")
        
        self.render_trend_insights(filtered_df)

# Run the dashboard
if __name__ == "__main__":
    dashboard = TrendTrackerDashboard()
    dashboard.run()