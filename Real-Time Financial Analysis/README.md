
# Real-Time Financial Analysis Dashboard  

## Overview  
This project is an AI-powered financial dashboard that provides real-time analysis of stock data and financial news. The dashboard integrates advanced technical analysis and sentiment evaluation to generate trading signals, all within an interactive **Streamlit** interface.

## Features  
- **Real-Time Stock Data**: Fetches historical and live stock data using **yfinance**.  
- **Technical Analysis**: Computes key indicators such as Simple Moving Averages (SMA) and Relative Strength Index (RSI) using **ta**.  
- **Sentiment Analysis**: Analyzes real-time financial news headlines using **Hugging Face Transformers** with **PyTorch** backend.  
- **Trading Signal Generation**: Provides actionable insights (Buy, Sell, Hold) based on combined technical and sentiment analysis.  
- **Data Logging**: Logs trading signals and analysis results into a **SQLite** database for tracking and evaluation.  
- **Interactive Dashboard**: A user-friendly **Streamlit** interface for exploring stock trends, news sentiment, and trading signals.

## Installation  

### Prerequisites  
- **Python 3.8+**  
- **Streamlit**  
- **yfinance**  
- **ta**  
- **transformers**  
- **torch**  
- **SQLite3**  

### Steps  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/financial-dashboard.git
   cd financial-dashboard
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Launch the Streamlit app:
   ```bash
   streamlit run app.py

## Technologies Used
1. Streamlit: For building the interactive dashboard.
2. yfinance: To fetch real-time stock market data.
3. ta: For technical analysis indicators.
4. Hugging Face Transformers: For sentiment analysis using pre-trained models.
5. PyTorch: Backend framework for running the sentiment analysis models.
6. SQLite: For logging and managing trading signals.
