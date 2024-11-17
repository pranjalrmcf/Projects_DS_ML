import yfinance as yf
from transformers import pipeline
import pandas as pd
import talib 
import streamlit as st
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go

# Streamlit app for stock selection
st.title("Real-Time Financial Analysis")

# Dropdown to select a stock
stock_symbol = st.selectbox(
    "Select a Stock Symbol:",
    options = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NFLX", "ADBE", "ORCL", "INTC",
    "IBM", "CRM", "CSCO", "SAP", "NVDA",
    "PEP", "KO", "BAC", "WMT", "PG",
    "JPM", "V", "MA", "DIS",
    "XOM", "CVX", "PFE", "T", "UNH",
    "NKE", "MCD", "HD", "BA", "MRK",
    "ABBV", "LLY", "CMCSA", "COST", "TXN",
    "SBUX", "AMGN", "QCOM", "UPS", "GS"
],
    index=0
)

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"
    )

def fetch_stock_data(symbol):
    """
    Fetches historical stock data for a given symbol using yfinance.
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1mo", interval="1d")
        if data.empty:
            st.error(f"No stock data found for {symbol}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {e}")
        return None

def fetch_news_data(symbol):
    """
    Fetches news headlines for a given stock symbol using yfinance.
    """
    try:
        stock = yf.Ticker(symbol)
        news_data = stock.news[:5]  # Get top 5 latest news
        if not news_data:
            st.warning(f"No news data found for {symbol}")
            return []
        return news_data
    except Exception as e:
        st.error(f"Error fetching news data for {symbol}: {e}")
        return []

def analyze_sentiment(news_headlines):
    """
    Performs sentiment analysis on a list of news headlines.
    """
    if not news_headlines:
        return []
    sentiment_analyzer = load_sentiment_analyzer()
    sentiments = [{"headline": news['title'], 
                   "sentiment": sentiment_analyzer(news['title'])[0]} for news in news_headlines]
    return sentiments

def analyze_trends(stock_data):
    """
    Analyzes market trends using advanced technical indicators.
    """
    if len(stock_data) < 20:
        st.error("Not enough data to calculate technical indicators.")
        return None

    stock_data['SMA'] = talib.SMA(stock_data['Close'], timeperiod=14)
    stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)
    stock_data['upper_band'], stock_data['middle_band'], stock_data['lower_band'] = talib.BBANDS(stock_data['Close'], timeperiod=20)
    stock_data['MACD'], stock_data['MACD_signal'], _ = talib.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    stock_data['ATR'] = talib.ATR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
    stock_data['ADX'] = talib.ADX(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
    stock_data['OBV'] = talib.OBV(stock_data['Close'], stock_data['Volume'])
    stock_data['STOCH_K'], stock_data['STOCH_D'] = talib.STOCH(
        stock_data['High'], stock_data['Low'], stock_data['Close'],
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )
    stock_data['WilliamsR'] = talib.WILLR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
    stock_data['SAR'] = talib.SAR(stock_data['High'], stock_data['Low'], acceleration=0.02, maximum=0.2)
    
    return stock_data[
        ['Close', 'SMA', 'RSI', 'upper_band', 'lower_band', 'MACD', 
         'MACD_signal', 'ATR', 'ADX', 'OBV', 'STOCH_K', 'STOCH_D', 'WilliamsR', 'SAR']
    ]

def generate_trading_signals(trend_data, sentiment_data):
    """
    Generates trading signals based on general market trends, sentiment analysis, 
    and Bollinger Bands, including all indicator values in the signals.
    """
    signals = []
    
    # Extract latest values for technical indicators
    rsi_value = trend_data['RSI'].iloc[-1]
    macd_value = trend_data['MACD'].iloc[-1]
    macd_signal = trend_data['MACD_signal'].iloc[-1]
    atr_value = trend_data['ATR'].iloc[-1]
    close_price = trend_data['Close'].iloc[-1]
    upper_band = trend_data['upper_band'].iloc[-1]
    lower_band = trend_data['lower_band'].iloc[-1]

    # Handle missing sentiment data
    if not sentiment_data:
        signals.append({
            "action": "HOLD", 
            "reason": "No sentiment data available.",
            "RSI": rsi_value, 
            "MACD": macd_value, 
            "MACD_signal": macd_signal, 
            "ATR": atr_value, 
            "Close Price": close_price, 
            "Bollinger Upper Band": upper_band, 
            "Bollinger Lower Band": lower_band
        })
        return signals
    
    # Aggregate sentiment
    positive_count = sum(1 for sentiment in sentiment_data if sentiment['sentiment']['label'] == 'POSITIVE')
    negative_count = sum(1 for sentiment in sentiment_data if sentiment['sentiment']['label'] == 'NEGATIVE')

    # Generalized Signal Logic
    if positive_count > negative_count:
        if rsi_value < 40 or macd_value > macd_signal:
            if close_price < lower_band:
                signals.append({
                    "action": "STRONG BUY",
                    "reason": "Positive sentiment with oversold Bollinger Bands and bullish RSI or MACD.",
                    "RSI": rsi_value, 
                    "MACD": macd_value, 
                    "MACD_signal": macd_signal, 
                    "ATR": atr_value, 
                    "Close Price": close_price, 
                    "Bollinger Upper Band": upper_band, 
                    "Bollinger Lower Band": lower_band
                })
            else:
                signals.append({
                    "action": "BUY",
                    "reason": "Positive sentiment with bullish RSI or MACD signal.",
                    "RSI": rsi_value, 
                    "MACD": macd_value, 
                    "MACD_signal": macd_signal, 
                    "ATR": atr_value, 
                    "Close Price": close_price, 
                    "Bollinger Upper Band": upper_band, 
                    "Bollinger Lower Band": lower_band
                })
        else:
            signals.append({
                "action": "HOLD",
                "reason": "Positive sentiment but no strong confirmation from indicators.",
                "RSI": rsi_value, 
                "MACD": macd_value, 
                "MACD_signal": macd_signal, 
                "ATR": atr_value, 
                "Close Price": close_price, 
                "Bollinger Upper Band": upper_band, 
                "Bollinger Lower Band": lower_band
            })
    elif negative_count > positive_count:
        if rsi_value > 60 or macd_value < macd_signal:
            if close_price > upper_band:
                signals.append({
                    "action": "STRONG SELL",
                    "reason": "Negative sentiment with overbought Bollinger Bands and bearish RSI or MACD.",
                    "RSI": rsi_value, 
                    "MACD": macd_value, 
                    "MACD_signal": macd_signal, 
                    "ATR": atr_value, 
                    "Close Price": close_price, 
                    "Bollinger Upper Band": upper_band, 
                    "Bollinger Lower Band": lower_band
                })
            else:
                signals.append({
                    "action": "SELL",
                    "reason": "Negative sentiment with bearish RSI or MACD signal.",
                    "RSI": rsi_value, 
                    "MACD": macd_value, 
                    "MACD_signal": macd_signal, 
                    "ATR": atr_value, 
                    "Close Price": close_price, 
                    "Bollinger Upper Band": upper_band, 
                    "Bollinger Lower Band": lower_band
                })
        else:
            signals.append({
                "action": "HOLD",
                "reason": "Negative sentiment but no strong confirmation from indicators.",
                "RSI": rsi_value, 
                "MACD": macd_value, 
                "MACD_signal": macd_signal, 
                "ATR": atr_value, 
                "Close Price": close_price, 
                "Bollinger Upper Band": upper_band, 
                "Bollinger Lower Band": lower_band
            })
    else:
        signals.append({
            "action": "HOLD",
            "reason": "Neutral sentiment and no clear trend.",
            "RSI": rsi_value, 
            "MACD": macd_value, 
            "MACD_signal": macd_signal, 
            "ATR": atr_value, 
            "Close Price": close_price, 
            "Bollinger Upper Band": upper_band, 
            "Bollinger Lower Band": lower_band
        })

    return signals


def review_signals(signals):
    """
    Displays signals in a Streamlit dashboard for manual review.
    """
    st.subheader("Trading Signals Review")
    signals_df = pd.DataFrame(signals)
    st.table(signals_df)

def log_trades(signals, symbol):
    """
    Logs trading signals to a SQLite database with clean separate columns for each indicator.
    """
    with sqlite3.connect('trading_signals.db') as conn:
        cursor = conn.cursor()
        
        # Drop the table if it exists (optional, for schema update purposes)
        cursor.execute('DROP TABLE IF EXISTS signals')
        
        # Create the signals table with individual columns for indicators
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                reason TEXT,
                RSI REAL,
                MACD REAL,
                MACD_signal REAL,
                ATR REAL,
                close_price REAL,
                bollinger_upper REAL,
                bollinger_lower REAL
            )
        ''')

        # Insert each signal into the table
        for signal in signals:
            cursor.execute('''
                INSERT INTO signals (
                    timestamp, symbol, action, reason, 
                    RSI, MACD, MACD_signal, ATR, 
                    close_price, bollinger_upper, bollinger_lower
                ) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                symbol,
                signal['action'],
                signal['reason'],
                signal['RSI'],
                signal['MACD'],
                signal['MACD_signal'],
                signal['ATR'],
                signal['Close Price'],
                signal['Bollinger Upper Band'],
                signal['Bollinger Lower Band']
            ))

        conn.commit()


# Run Analysis
if st.button("Run Analysis"):
    stock_data = fetch_stock_data(stock_symbol)
    news_data = fetch_news_data(stock_symbol)
    
    if stock_data is not None:
        st.subheader(f"{stock_symbol} Stock Data (Last 30 Days)")
        st.dataframe(stock_data)
        
        trend_data = analyze_trends(stock_data)
        if trend_data is not None:
            st.subheader("Stock Trends")
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=trend_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Candlestick'
            ))
            fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['SMA'], mode='lines', name='SMA'))
            fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['upper_band'], mode='lines', name='Upper Band'))
            fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['lower_band'], mode='lines', name='Lower Band'))
            st.plotly_chart(fig)

            sentiment_data = analyze_sentiment(news_data)
            st.subheader(f"Latest News for {stock_symbol}")
            for item in sentiment_data:
                st.write(f"**{item['headline']}**")
                st.write(f"Sentiment: {item['sentiment']['label']} (Score: {item['sentiment']['score']:.2f})")
                st.markdown("---")
            
            signals = generate_trading_signals(trend_data, sentiment_data)
            review_signals(signals)
            log_trades(signals, stock_symbol)
        else:
            st.error("Trend analysis failed.")
    else:
        st.error("Unable to fetch data.")
