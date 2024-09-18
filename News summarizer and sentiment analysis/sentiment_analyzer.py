import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Load the CSV file into a DataFrame
df = pd.read_csv('D:/DS/Suvidha/indian_news_with_summaries_final.csv')

# Function to calculate sentiment and return sentiment and score
def analyze_sentiment(text):
    # Check if the text is valid
    if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
        return ('neutral', 0.0)
    
    # Get the sentiment scores
    sentiment_scores = sia.polarity_scores(text)
    
    # Get the compound score (which is from -1 to 1)
    compound_score = sentiment_scores['compound']
    
    # Classify the sentiment based on the compound score
    if compound_score >= 0.05:
        sentiment = 'positive'
    elif compound_score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    # Return the sentiment category and normalized score (0 to 1)
    return (sentiment, (compound_score + 1) / 2)

# Apply the sentiment analysis function to each row in the 'Content' column
df['Sentiment'], df['Sentiment_Score'] = zip(*df['Content'].apply(analyze_sentiment))

# Save the updated DataFrame to a new CSV file
df.to_csv('indian_news_with_sentiment.csv', index=False)

print("Sentiment analysis completed and saved to 'indian_news_with_sentiment.csv'.")
