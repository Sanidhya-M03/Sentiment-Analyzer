# News Sentiment Analysis

This project fetches news articles using the NewsAPI and performs sentiment analysis on the content using NLTK's VADER sentiment analyzer. The sentiment scores are then visualized using a histogram.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required packages:
    ```bash
    pip install newsapi-python yfinance nltk matplotlib numpy pandas
    ```

3. Download the VADER lexicon for NLTK:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    ```

## Usage

1. Import necessary libraries and initialize the NewsAPI client:
    ```python
    import nltk
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from newsapi import NewsApiClient
    from datetime import date, timedelta, datetime
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    pd.set_option('display.max_colwidth', 1000)
    ```

2. Set your NewsAPI key and initialize the client:
    ```python
    NEWS_API_KEY = 'your_newsapi_key_here'
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    ```

3. Define a function to get articles and their sentiments:
    ```python
    def get_articles_sentiments(keywrd, startd, sources_list=None, show_all_articles=False):
        if type(startd) == str:
            my_date = datetime.strptime(startd, '%d-%b-%Y')
        else:
            my_date = startd
        if sources_list:
            articles = newsapi.get_everything(q=keywrd,
                                              from_param=my_date.isoformat(),
                                              to=(my_date + timedelta(days=1)).isoformat(),
                                              language="en",
                                              sources=",".join(sources_list),
                                              sort_by="relevancy",
                                              page_size=100)
        else:
            articles = newsapi.get_everything(q=keywrd,
                                              from_param=my_date.isoformat(),
                                              to=(my_date + timedelta(days=1)).isoformat(),
                                              language="en",
                                              sort_by="relevancy",
                                              page_size=100)
        article_content = ''
        date_sentiments = {}
        date_sentiments_list = []
        seen = set()
        for article in articles['articles']:
            if str(article['title']) in seen:
                continue
            else:
                seen.add(str(article['title']))
                article_content = str(article['title']) + '. ' + str(article['description'])
                sentiment = sia.polarity_scores(article_content)['compound']
                date_sentiments.setdefault(my_date, []).append(sentiment)
                date_sentiments_list.append((sentiment, article['url'], article['title'], article['description']))
        return pd.DataFrame(date_sentiments_list, columns=['Sentiment', 'URL', 'Title', 'Description'])
    ```

4. Fetch and analyze the news articles:
    ```python
    keywrd = 'Tata'
    my_date = date.today() - timedelta(days=1)
    return_articles = get_articles_sentiments(keywrd=keywrd, startd=my_date, sources_list=None, show_all_articles=True)
    return_articles.Sentiment.hist(bins=30, grid=False)
    print(return_articles)
    ```

5. Save the results to a CSV file:
    ```python
    return_articles["Date"] = my_date
    return_articles.to_csv("Sentiment_Details_" + my_date.isoformat() + ".csv")
    ```

## Features

- Fetch news articles from the NewsAPI
- Perform sentiment analysis on the articles using NLTK's VADER
- Visualize sentiment scores using histograms
- Save results to a CSV file
