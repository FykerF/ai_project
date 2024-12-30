from datetime import datetime, timedelta
import requests


def get_news(date_str: str, ticker: str = 'SPY') -> dict:
    """
    Fetches news from stocknewsapi.com for the given date and ticker.
    Returns a dictionary mapping news titles to their article text.

    If any errors occur (network, JSON parsing, missing fields), 
    an empty dictionary is returned.

    :param date_str: Date in 'YYYY-MM-DD' format.
    :param ticker: Stock ticker, default 'SPY'.
    :return: Dictionary {title: text} of news articles, or {} on error.
    """
    API_KEY = 'api_key'
    try:
        # Parse and format dates
        date_obj = datetime.strptime(date_str.strip(), '%Y-%m-%d')
        today = date_obj.strftime("%m%d%Y")
        yesterday = (date_obj - timedelta(days=1)).strftime("%m%d%Y")

        url = (
            f'https://stocknewsapi.com/api/v1?tickers={ticker}'
            f'&items=100&date={yesterday}-{today}&page=1&token={API_KEY}'
        )

        # Make the API request
        response = requests.get(url, timeout=10)
        # Raise an HTTPError if the response was unsuccessful (4xx/5xx)
        response.raise_for_status()

        # Parse JSON
        data = response.json()

        # Validate the JSON structure
        if 'data' not in data:
            print(f"Warning: 'data' key not found in response. Returning empty dict.")
            return {}

        # Expecting 'data' to be a list of news items
        if not isinstance(data['data'], list):
            print(f"Warning: 'data' field in response is not a list. Returning empty dict.")
            return {}

        # Build the result dictionary
        news_list = data['data']
        news = {}
        for item in news_list:
            title = item.get('title')
            text = item.get('text')
            if title and text:
                news[title] = text

        return news

    except ValueError as ve:
        # Catches date parsing errors or JSON decode errors
        print(f"ValueError encountered (date parsing or JSON issue): {ve}")
    except requests.exceptions.RequestException as re:
        # Catches any network-related issues (timeout, connection errors, etc.)
        print(f"Network error occurred while fetching news: {re}")
    except Exception as e:
        # Catches any other unexpected errors
        print(f"An unexpected error occurred while fetching news: {e}")

    # Return empty dict if any exception was raised
    return {}