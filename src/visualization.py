import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class TrendVisualizer:
    def __init__(self, style='darkgrid'):
        self.style = style
        self.set_style()
        
    def set_style(self):
        """Set visualization style"""
        if self.style == 'darkgrid':
            plt.style.use('dark_background')
            sns.set_style("darkgrid")
        else:
            plt.style.use('default')
            sns.set_style("whitegrid")
        
        # Set color palette
        self.colors = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#4682B4',   # Steel Blue
            'background': '#1E1E1E' if self.style == 'darkgrid' else '#FFFFFF'
        }
        
    def plot_sentiment_timeline(self, df, date_col='Date', sentiment_col='Sentiment_Score', 
                               title='Sentiment Trend Over Time'):
        """Plot sentiment scores over time"""
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Sentiment Score Timeline', 'Daily Average Sentiment'),
                           vertical_spacing=0.1)
        
        # Raw sentiment scores
        fig.add_trace(
            go.Scatter(x=df[date_col], y=df[sentiment_col], 
                      mode='markers', name='Individual Posts',
                      marker=dict(size=4, opacity=0.6, color=df[sentiment_col],
                                colorscale='RdYlGn', showscale=True)),
            row=1, col=1
        )
        
        # Daily average
        daily_avg = df.groupby(df[date_col].dt.date)[sentiment_col].mean().reset_index()
        daily_avg[date_col] = pd.to_datetime(daily_avg[date_col])
        
        fig.add_trace(
            go.Scatter(x=daily_avg[date_col], y=daily_avg[sentiment_col],
                      mode='lines+markers', name='Daily Average',
                      line=dict(color='white', width=3),
                      marker=dict(size=8)),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white' if self.style == 'darkgrid' else 'black')
        )
        
        return fig
    
    def plot_sentiment_distribution(self, df, sentiment_col='Sentiment_Label'):
        """Plot distribution of sentiment labels"""
        sentiment_counts = df[sentiment_col].value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='Sentiment Distribution',
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': self.colors['positive'],
                'negative': self.colors['negative'],
                'neutral': self.colors['neutral']
            }
        )
        
        fig.update_layout(
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white' if self.style == 'darkgrid' else 'black')
        )
        
        return fig
    
    def plot_topic_evolution(self, df, date_col='Date', topic_col='Dominant_Topic'):
        """Plot how topics evolve over time"""
        # Prepare data
        df['Date'] = pd.to_datetime(df[date_col])
        df['Week'] = df['Date'].dt.isocalendar().week
        
        weekly_topics = df.groupby(['Week', topic_col]).size().reset_index(name='Count')
        weekly_total = df.groupby('Week').size().reset_index(name='Total')
        weekly_topics = weekly_topics.merge(weekly_total, on='Week')
        weekly_topics['Percentage'] = (weekly_topics['Count'] / weekly_topics['Total']) * 100
        
        fig = px.area(
            weekly_topics, 
            x='Week', 
            y='Percentage', 
            color=topic_col,
            title='Topic Evolution Over Time',
            labels={'Percentage': 'Topic Percentage (%)', 'Week': 'Week'}
        )
        
        fig.update_layout(
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white' if self.style == 'darkgrid' else 'black')
        )
        
        return fig
    
    def plot_engagement_metrics(self, df, platform='twitter'):
        """Plot engagement metrics based on platform"""
        if platform == 'twitter':
            metrics = ['Likes', 'Retweets', 'Replies']
            title = 'Twitter Engagement Metrics'
        else:  # reddit
            metrics = ['Score', 'Comments', 'Upvote_Ratio']
            title = 'Reddit Engagement Metrics'
        
        fig = make_subplots(rows=1, cols=len(metrics), 
                           subplot_titles=metrics)
        
        for i, metric in enumerate(metrics, 1):
            if metric in df.columns:
                fig.add_trace(
                    go.Histogram(x=df[metric], name=metric, nbinsx=20),
                    row=1, col=i
                )
        
        fig.update_layout(
            title=title,
            height=400,
            showlegend=False,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white' if self.style == 'darkgrid' else 'black')
        )
        
        return fig
    
    def plot_hashtag_network(self, df, top_n=20):
        """Plot most frequent hashtags (for Twitter data)"""
        if 'Extracted_Hashtags' not in df.columns:
            print("No hashtag data available")
            return None
        
        # Extract and count hashtags
        all_hashtags = []
        for hashtags in df['Extracted_Hashtags'].dropna():
            if isinstance(hashtags, list):
                all_hashtags.extend(hashtags)
        
        hashtag_counts = Counter(all_hashtags)
        top_hashtags = hashtag_counts.most_common(top_n)
        
        if not top_hashtags:
            print("No hashtags found")
            return None
        
        hashtags_df = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Count'])
        
        fig = px.bar(
            hashtags_df,
            x='Count',
            y='Hashtag',
            orientation='h',
            title=f'Top {top_n} Most Frequent Hashtags',
            color='Count',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white' if self.style == 'darkgrid' else 'black'),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_word_cloud_plotly(self, texts, title='Word Cloud'):
        """Create an interactive word cloud using Plotly"""
        from wordcloud import WordCloud
        import base64
        from io import BytesIO
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white' if self.style != 'darkgrid' else 'black',
            colormap='viridis',
            max_words=100
        ).generate(' '.join([str(text) for text in texts if pd.notna(text)]))
        
        # Convert to image
        img_buffer = BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Create figure
        fig = go.Figure()
        
        fig.add_layout_image(
            dict(
                source=f'data:image/png;base64,{img_str}',
                xref="paper", yref="paper",
                x=0, y=1,
                sizex=1, sizey=1,
                xanchor="left", yanchor="top"
            )
        )
        
        fig.update_layout(
            title=title,
            height=450,
            showlegend=False,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white' if self.style == 'darkgrid' else 'black')
        )
        
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        return fig
    
    def plot_correlation_heatmap(self, df, numerical_cols=None):
        """Plot correlation heatmap for numerical columns"""
        if numerical_cols is None:
            numerical_cols = ['Sentiment_Score', 'Likes', 'Retweets', 'Replies', 'Word_Count']
        
        # Select only columns that exist in dataframe
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if len(numerical_cols) < 2:
            print("Not enough numerical columns for correlation analysis")
            return None
        
        corr_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Correlation Heatmap',
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        
        fig.update_layout(
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white' if self.style == 'darkgrid' else 'black')
        )
        
        return fig

# Example usage
if __name__ == "__main__":
    # Sample data for testing
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_df = pd.DataFrame({
        'Date': dates,
        'Sentiment_Score': np.random.normal(0, 0.5, 100),
        'Sentiment_Label': np.random.choice(['positive', 'negative', 'neutral'], 100),
        'Dominant_Topic': np.random.choice(['AI', 'Tech', 'Privacy', 'Innovation', 'Future'], 100),
        'Likes': np.random.poisson(50, 100),
        'Retweets': np.random.poisson(10, 100)
    })
    
    visualizer = TrendVisualizer(style='darkgrid')
    
    # Generate sample plots
    fig1 = visualizer.plot_sentiment_timeline(sample_df)
    fig2 = visualizer.plot_sentiment_distribution(sample_df)
    fig3 = visualizer.plot_topic_evolution(sample_df)
    
    # Show plots (in a real environment, these would be displayed)
    print("Visualization functions ready to use!")