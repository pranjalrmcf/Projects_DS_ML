# News Processing Pipeline

## Overview
This project consists of three scripts to collect, process, and analyze news articles from various Indian news websites. The pipeline includes gathering news articles, summarizing them, and performing sentiment analysis to understand the sentiment behind the content.

### Scripts Overview:
1. **`news_collection.py`**: Collects news articles from multiple Indian newspapers.
2. **`summarization.py`**: Generates summaries of the collected news articles.
3. **`sentiment_analyzer.py`**: Analyzes the sentiment of the summarized news articles.

---

## Features

### News Collection (`news_collection.py`)
- Scrapes news articles from a list of popular Indian news websites.
- Extracts article information including headline, publication date, URL, and content.
- Assigns a category to each article based on its keywords (e.g., Politics, Business, Sports).
- Saves the collected data in a CSV file (`indian_news_2.csv`).

### Summarization (`summarization.py`)
- Uses the DistilBART model to generate concise summaries of the collected articles.
- Saves the summarized content into a new CSV file (`indian_news_with_summaries_final.csv`).
- Uses a progress bar for monitoring the summarization process.

### Sentiment Analysis (`sentiment_analyzer.py`)
- Analyzes the sentiment of each news article using VADER (a pre-trained sentiment analysis model).
- Classifies sentiment as `positive`, `negative`, or `neutral` and calculates a sentiment score.
- Saves the sentiment results to a new CSV file (`indian_news_with_sentiment.csv`).

---

## Requirements

- Python 3.x
- Required Python packages:
  - `newspaper3k`
  - `pandas`
  - `nltk`
  - `torch`
  - `transformers`
  - `tqdm`

To install the required dependencies, run:
```bash
pip install newspaper3k pandas nltk torch transformers tqdm
```

Additionally, you need to download the VADER lexicon for sentiment analysis:
```python
import nltk
nltk.download('vader_lexicon')
```

---

## Installation and Usage

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```

### Step 2: News Collection
1. Run the `news_collection.py` script to collect news articles:
   ```bash
   python news_collection.py
   ```
2. The script will save the news data to `indian_news_2.csv`.

### Step 3: Summarization
1. Run the `summarization.py` script to generate summaries of the collected articles:
   ```bash
   python summarization.py
   ```
2. The script will save the summarized articles to `indian_news_with_summaries_final.csv`.

### Step 4: Sentiment Analysis
1. Run the `sentiment_analyzer.py` script to perform sentiment analysis:
   ```bash
   python sentiment_analyzer.py
   ```
2. The script will save the sentiment results to `indian_news_with_sentiment.csv`.

---

## Project Workflow

1. **Collect News Articles**: Using `news_collection.py`, articles from various Indian news websites are scraped and saved to a CSV file.
2. **Generate Summaries**: The `summarization.py` script processes the collected articles and generates summaries, making it easier to digest lengthy content.
3. **Analyze Sentiment**: The `sentiment_analyzer.py` script assesses the overall sentiment of each article, allowing for better understanding of the tone of the news.

---

## File Descriptions

- **`news_collection.py`**: 
  - Collects news articles from specified URLs.
  - Extracts details like headline, content, publication date, and assigns a category.
  - Saves the collected articles in `indian_news_2.csv`.

- **`summarization.py`**:
  - Loads articles from the CSV file.
  - Summarizes the content using the DistilBART model.
  - Saves the summarized articles in `indian_news_with_summaries_final.csv`.

- **`sentiment_analyzer.py`**:
  - Loads summarized articles.
  - Performs sentiment analysis on each article using the VADER sentiment analysis tool.
  - Saves the sentiment results in `indian_news_with_sentiment.csv`.

---

## Dependencies
- **newspaper3k**: For scraping news articles.
- **Pandas**: For data manipulation and reading/writing CSV files.
- **NLTK**: For sentiment analysis using the VADER lexicon.
- **PyTorch and Transformers**: For running the summarization model.
- **Tqdm**: For tracking the progress of summarization.

## Notes
- **Device Compatibility**: The summarization script (`summarization.py`) will use GPU (CUDA) if available; otherwise, it will default to CPU.
- **Data Source**: The URLs used for scraping are of popular Indian newspapers, including The Hindu, Hindustan Times, Indian Express, and more.

---

## Acknowledgments
- **Newspaper3k** for providing a convenient interface for news scraping.
- **Hugging Face Transformers** for the summarization model.
- **NLTK** for sentiment analysis tools.

