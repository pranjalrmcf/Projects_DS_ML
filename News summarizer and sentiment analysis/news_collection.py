import newspaper
import csv
from datetime import datetime
import time

# Updated list of Indian newspaper URLs
newspaper_urls = [
    'https://www.thehindu.com',
    'https://www.hindustantimes.com',
    'https://indianexpress.com',
    'https://www.telegraphindia.com',
    'https://www.deccanchronicle.com',
    'https://www.newindianexpress.com',
    'https://www.livemint.com',
    'https://www.business-standard.com',
    'https://www.financialexpress.com',
    'https://www.dnaindia.com',
    'https://www.tribuneindia.com',
    'https://www.thestatesman.com',
    'https://www.asianage.com',
    'https://www.dailypioneer.com',
    'https://www.freepressjournal.in',
    'https://economictimes.indiatimes.com',
    'https://www.thehansindia.com',
    'https://www.orissapost.com',
    'https://www.thehitavada.com',
    'https://www.sentinelassam.com',
    'https://www.navhindtimes.in',
    'https://assamtribune.com',
    'https://arunachaltimes.in',
    'https://theshillongtimes.com',
    'https://www.thesangaiexpress.com',
]

csv_columns = ['Newspaper Name', 'Published Date', 'URL', 'Headline', 'Content', 'Category']
processed_urls = set()

with open('indian_news.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(csv_columns)

for url in newspaper_urls:
    print(f"Processing newspaper: {url}")
    paper = newspaper.build(url, language='en', memoize_articles=False)
    
    newspaper_name = paper.brand or url.split('//')[1].split('/')[0]
    article_count = 0
    
    for article in paper.articles:
        if article.url in processed_urls:
            continue  # Skip duplicates
        
        if article_count >= 1000:
            break  # Limit per newspaper
        
        try:
            article.download()
            article.parse()
            article.nlp()
            
            # Check if article has content
            if not article.text or not article.text.strip():
                print(f"Skipping article with no content: {article.url}")
                continue  # Skip articles without content

            # Filter by publish date
            if article.publish_date and article.publish_date.year >= 2020:
                published_date = article.publish_date.strftime('%Y-%m-%d')
            else:
                continue  # Skip articles without publish date or before 2020
            
            # Assign category
            def assign_category(keywords):
                category_keywords = {
                    'Politics': ['politics', 'government', 'election', 'policy', 'minister', 'parliament', 'senate', 'congress', 'political'],
                    'International News': ['international', 'global', 'world', 'foreign', 'overseas'],
                    'National News': ['national', 'domestic', 'india', 'countrywide'],
                    'Local News': ['local', 'city', 'town', 'regional', 'community'],
                    'Business and Finance': ['business', 'finance', 'economy', 'market', 'stock', 'trade', 'industry', 'investment'],
                    'Science and Technology': ['science', 'technology', 'tech', 'research', 'innovation', 'discovery', 'scientific'],
                    'Health and Wellness': ['health', 'medicine', 'medical', 'wellness', 'fitness', 'disease', 'doctor', 'hospital', 'nutrition'],
                    'Entertainment': ['entertainment', 'movie', 'film', 'music', 'television', 'celebrity', 'show', 'drama', 'concert'],
                    'Sports': ['sports', 'cricket', 'football', 'match', 'tournament', 'athlete', 'olympics', 'game'],
                    'Lifestyle and Features': ['lifestyle', 'culture', 'fashion', 'travel', 'food', 'feature', 'art', 'design', 'style'],
                    'Opinion and Editorial': ['opinion', 'editorial', 'commentary', 'column', 'analysis', 'perspective'],
                    'Environment': ['environment', 'climate', 'pollution', 'wildlife', 'conservation', 'sustainability', 'ecology'],
                    'Education': ['education', 'school', 'university', 'college', 'student', 'teacher', 'exam', 'academic', 'learning'],
                    'Crime and Justice': ['crime', 'justice', 'law', 'court', 'police', 'legal', 'trial', 'verdict', 'criminal'],
                    'Human Interest': ['human interest', 'story', 'community', 'society', 'people', 'life'],
                    'Obituaries': ['obituary', 'death', 'passed away', 'memorial', 'tribute'],
                    'Weather': ['weather', 'forecast', 'temperature', 'rain', 'storm', 'climate'],
                    'Religion and Spirituality': ['religion', 'spirituality', 'faith', 'worship', 'belief', 'church', 'temple', 'mosque'],
                    'Technology and Gadgets': ['gadgets', 'devices', 'smartphone', 'computer', 'software', 'hardware', 'app', 'innovation'],
                    'Automotive': ['automotive', 'car', 'vehicle', 'motor', 'transport', 'automobile', 'bike'],
                }

                for category, words in category_keywords.items():
                    if any(word.lower() in (kw.lower() for kw in keywords) for word in words):
                        return category
                return 'General'

            category = assign_category(article.keywords)
            
            data = [
                newspaper_name,
                published_date,
                article.url,
                article.title,
                article.text,
                category
            ]
            
            with open('indian_news_2.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            
            processed_urls.add(article.url)
            article_count += 1
            time.sleep(1)  # Delay between articles
        except Exception as e:
            print(f"Error processing article at {article.url}: {e}")
            continue

print("\nNews articles have been saved to 'indian_news_2.csv'.")
