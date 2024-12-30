from pydantic import BaseModel, ValidationError
from openai import OpenAI



# Initialize OpenAI client
client = OpenAI(
    api_key='api_key')  

# Define the Pydantic model for structured output
class PolarityResponse(BaseModel):
    #polarity_score: float  # Continuous value in the interval [-1, 1]
    news_analysis_score: float  # Первый результат анализа новостей
    combined_score: float  # Результат, основанный на новостях и анализе BKM



def get_score_from_news(text: dict, ticker: str):
    prompt = (
        f"You are a financial expert who reads public news and identifies whether the news is about the company '{ticker}'"
        f"or not to predict the movement in stock prices for the '{ticker}’. You read the following news: {text}. Firstly, classify the news as related to the company "
        f"'{ticker}' (1=yes, 0=no). Then classify the news as related to one  of the following categories:\n"
        "1. «Financial News» - news about earnings, stock performance, mergers\n"
        "2. «Product News» - news about launches, updates, innovations\n"
        "3. «Regulatory News» - news about legal issues, government regulations, antitrust cases\n"
        "4. «Market Competition» - news about competitor comparisons, industry trends\n"
        "5. «Executive and Leadership News» - news about CEO changes, executive decisions\n"
        "6. «Public Relations and Reputation» - news about scandals, controversies, social impact\n"
        "7. «Technology and Innovation» - news about R&D, patents, new technologies\n"
        "8. «Partnerships and Collaborations» - news about strategic alliances, joint ventures\n"
        "9. «Customer and Market Feedback» - news about customer reviews, market reactions\n"
        "10. «Other» - some news that do not belong to any other categories\n"
        "Then classify the sentiment as positive (1) if the news can affect the up movement in stock prices, negative (-1) if the news can affect "
        "the down movement in stock prices and neutral (0) otherwise.\n"
        "And the last one is the overall classification whether the following news can cause the price movement at the market (1=yes, 0=no).\n"
        "Here is the example of the output: (1, «Financial News», 1, 1)"
    )
    response = completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a financial sentiment analysis expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0
    )

    message = completion.choices[0].message.content
    return message



def analyze_news(bl_data, ticker, news_analysis, pcr_data, call_volume, put_volume):
    prompt = (
        f"Hello, can you please\n"
        f"Analyze the Breeden-Litzenberger estimators (variance, skewness, and kurtosis) by comparing today’s values with those from yesterday.\n"
        f"Data for Breeden-Litzenberger estimators: {bl_data}.\n"
        f"The data for each day is organized by expiration date, structured as follows for both `data_today` and `data_yesterday`:\n"
        f"data_today = {{expiration_date_1: {{'variance': variance_today_1, 'skewness': skewness_today_1, 'kurtosis': kurtosis_today_1}},\n"
        f"expiration_date_2: {{...}}, ...}}\n"
        f"data_yesterday = {{expiration_date_1: {{'variance': variance_yesterday_1, 'skewness': skewness_yesterday_1, 'kurtosis': kurtosis_yesterday_1}},\n"
        f"expiration_date_2: {{...}}, ...}}\n"
        f"Your task is to analyze the differences for each parameter (variance, skewness, kurtosis) between today and yesterday, providing a metric-specific score for each parameter on a scale: [-1, -0.5, 0, 0.5, 1]. Use the following parameter-specific criteria:\n\n"

        f"1. **Calculate the daily difference**:\n"
        f"   - For each expiration date and parameter, subtract yesterday’s value from today’s, i.e., delta = today_value - yesterday_value.\n\n"

        f"2. **Apply scoring criteria to each difference by parameter**:\n"
        f"   - For each parameter, map `delta` to a score based on parameter-specific trends:\n"
        f"     - **Variance**:\n"
        f"       - -1: Large decrease, indicating a notable reduction in volatility.\n"
        f"       - -0.5: Small decrease, implying a minor reduction in volatility.\n"
        f"       - 0: Little or no change in volatility.\n"
        f"       - 0.5: Small increase, suggesting a slight uptick in volatility.\n"
        f"       - 1: Large increase, indicating heightened volatility.\n"
        f"     - **Skewness**:\n"
        f"       - -1: Large decrease, reflecting a strong shift towards negative skew (left-tail risk).\n"
        f"       - -0.5: Small decrease, suggesting a minor shift towards negative skew.\n"
        f"       - 0: No significant change, skew remains stable.\n"
        f"       - 0.5: Small increase, suggesting a minor shift towards positive skew (right-tail risk).\n"
        f"       - 1: Large increase, indicating a strong shift towards positive skew.\n"
        f"     - **Kurtosis**:\n"
        f"       - -1: Large decrease, indicating a drop in tail risk or fewer extreme values.\n"
        f"       - -0.5: Small decrease, suggesting a minor decrease in tail risk.\n"
        f"       - 0: No significant change, distribution remains stable in terms of tails.\n"
        f"       - 0.5: Small increase, indicating slightly more tail risk or extreme events.\n"
        f"       - 1: Large increase, suggesting a sharp increase in tail risk or extreme values.\n"
        f"     - For each parameter, use `threshold_high` and `threshold_low` values specific to historical norms or expected ranges to determine whether changes are 'large' or 'small'.\n\n"

        f"3. **Weight the scores based on expiration dates**:\n"
        f"   - Closer expiration dates should be weighted higher in the final analysis, as they are more immediate. Factor this weighting into your final summary or aggregate score for each parameter.\n\n"

        f"4. **Interpret the scores**:\n"
        f"   - High absolute scores (like -1 or 1) indicate strong trends or notable shifts in that parameter.\n"
        f"   - Scores closer to zero imply stability or minimal change.\n\n"

        f"5. **Output the results**:\n"
        f"   - For each expiration date and parameter, provide the calculated score and a brief explanation describing the trend (stable, increasing, or decreasing) specific to that parameter.\n"
        f"   - Summarize the overall trend by aggregating or averaging the scores, prioritizing closer expiration dates in the final interpretation.\n\n"

        f"PLEASE\n\n"
    
        f"Also Incorporate the following additional option market data into analysis of bkm_estimators. "
        f"- Put-Call Ratio (PCR): {pcr_data}.\n"
        f"- Call Option Volume: {call_volume}.\n"
        f"- Put Option Volume: {put_volume}.\n"
        f"- Then give a score based on estimators and this metrics for a day"
    
        f"Then here is also the news for {ticker}:\n{news_analysis}\n. Please take into account only news which really can influence the price movements And correspond to the company. Please give a score from range (-1, 0,1), 1 if the news can positively influence the price movement, -1 if news can negatively affect and 0 if the news are neutral\n"
        f"Conclude with an overall sentiment based on BKM_estimators and the news analysis and volumes.\n"
        f"Return the result as scores in format (news_score, estimators_score).\n"
    )

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "Final part of the five-part analysis."},
                {"role": "user", "content": prompt}
            ],
            response_format=PolarityResponse,
            temperature=0.0
        )

        message = completion.choices[0].message
        news_analysis_score = message.parsed.news_analysis_score if message.parsed else 0.0
        combined_score = message.parsed.combined_score if message.parsed else 0.0

        return (float(news_analysis_score), float(combined_score))

    except ValidationError as e:
        print(f"Validation error with structured response: {e}")
        return (0.0, 0.0)
    except Exception as e:
        print(f"An error occurred while fetching polarity: {e}")
        return (0.0, 0.0)