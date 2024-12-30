import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from BL_dynamics import compute_risk_neutral_pdfs
from news_api_wrapper import get_news
from news_analysis import analyze_news, get_score_from_news

def run_pipeline(n_days=10, progress_callback=None):
    """
    Runs the pipeline for the last `n_days` of data.
    Returns:
      - df_plot: A DataFrame containing the date, cumulative returns, and cumulative sentiment
                 for the last n_days.
      - fig:     A Matplotlib Figure object with the 2-axis plot.
    """

    # Connect to SQLite database (adjust path if needed)
    db_path = '/app/spx_data.db'
    conn = sqlite3.connect(db_path)

    # Query to load options data
    query = """
    SELECT
        " [QUOTE_DATE]"   AS quote_date,
        " [EXPIRE_DATE]"  AS expire_date,
        " [STRIKE]"       AS strike,
        " [C_LAST]"       AS c_last,
        " [P_LAST]"       AS p_last,
        " [DTE]"          AS dte,
        " [UNDERLYING_LAST]" AS underlying_last,
        " [C_VOLUME]"     AS c_volume,
        " [P_VOLUME]"     AS p_volume
    FROM spx_data
    ORDER BY quote_date, strike
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Unique dates
    unique_dates = df['quote_date'].drop_duplicates().values

    if len(unique_dates) < 2:
        print("Not enough distinct dates to process.")
        return None, None

    # We'll store a daily 'combined sentiment' (from BKM + news) in this dict
    daily_sentiment_scores = {}

    # Define the start index for the last n days
    start_idx = len(unique_dates) - n_days
    if start_idx < 1:
        start_idx = 1  # ensure we have at least one prior day to compare

    # The loop will run from day index (start_idx) up to the last day
    total_iters = len(unique_dates) - start_idx
    current_iter = 0

    for i in range(start_idx, len(unique_dates)):
        # if we’re at i=0 (edge case), skip
        if i == 0:
            continue
        # Each iteration compares day (i-1) vs. day (i)
        if i == 0:
            continue

        t_minus_1 = unique_dates[i - 1]
        t = unique_dates[i]

        df_t_minus_1 = df[df['quote_date'] == t_minus_1]
        df_t         = df[df['quote_date'] == t]

        if df_t_minus_1.empty or df_t.empty:
            continue

        # Focus on near-term expirations (≤ 10 DTE), if desired
        expiration_dates = df_t.loc[df_t['dte'] <= 10, 'expire_date'].drop_duplicates().values

        resultik = {}
        try:
            for expiration_date in expiration_dates:
                df_t_given_exp = df_t[df_t['expire_date'] == expiration_date]
                df_t_minus_1_given_exp = df_t_minus_1[df_t_minus_1['expire_date'] == expiration_date]

                # Compute BL for calls
                call_data_t, call_data_t_minus_1 = compute_risk_neutral_pdfs(
                    df_t_given_exp['c_last'],
                    df_t_given_exp['strike'],
                    df_t_minus_1_given_exp['c_last'],
                    df_t_minus_1_given_exp['strike'],
                    df_t_given_exp['dte'].iloc[-1] / 252,
                    df_t_minus_1_given_exp['dte'].iloc[-1] / 252
                )
                # Compute BL for puts
                put_data_t, put_data_t_minus_1 = compute_risk_neutral_pdfs(
                    df_t_given_exp['p_last'],
                    df_t_given_exp['strike'],
                    df_t_minus_1_given_exp['p_last'],
                    df_t_minus_1_given_exp['strike'],
                    df_t_given_exp['dte'].iloc[-1] / 252,
                    df_t_minus_1_given_exp['dte'].iloc[-1] / 252
                )

                bl_estimators = {
                    "call_data_t": call_data_t,
                    "call_data_t_minus_1": call_data_t_minus_1,
                    "put_data_t": put_data_t,
                    "put_data_t_minus_1": put_data_t_minus_1
                }
                resultik[f'{expiration_date}_bl_estimators'] = bl_estimators


        except Exception as e:
            print(f"Error {e}")
        
        current_iter += 1
        if progress_callback:
            pct = int((current_iter / total_iters) * 100)
            progress_callback(pct)

        # Fetch relevant news for date t
        news_dict = get_news(t, ticker="SPY")

        # First layer of sentiment extraction
        new_analysis_step_1 = get_score_from_news(news_dict, 'SPY')

        # Parse volume-based metrics for day t
        total_call_vol = df_t['c_volume'].sum()
        total_put_vol  = df_t['p_volume'].sum()
        pcr_data = total_put_vol / total_call_vol if total_call_vol != 0 else np.nan

        # LLM-based function to combine BKM + news
        news_score, combined_score = analyze_news(
            resultik,
            ticker="SPY",
            news_analysis=new_analysis_step_1,
            pcr_data=pcr_data,
            call_volume=total_call_vol,
            put_volume=total_put_vol
        )

        daily_sentiment_scores[t] = combined_score
        print(f"[{t}] PCR={pcr_data:.3f}, News Score={news_score}, Combined Score={combined_score}")

    # Build a DataFrame of daily underlying prices
    df_price_only = (
        df[['quote_date', 'underlying_last']]
        .drop_duplicates(subset=['quote_date'])
        .sort_values(by='quote_date')
        .reset_index(drop=True)
    )

    # Compute daily & cumulative returns
    df_price_only['daily_return'] = df_price_only['underlying_last'].pct_change()
    df_price_only['cumulative_return'] = (1 + df_price_only['daily_return']).cumprod() - 1

    # Merge day-level combined sentiment
    df_price_only['daily_sentiment'] = df_price_only['quote_date'].map(daily_sentiment_scores).fillna(0.0)
    df_price_only['cumulative_sentiment'] = df_price_only['daily_sentiment'].cumsum()

    # Filter final results to the same last n days
    relevant_dates = unique_dates[start_idx:]
    df_plot = df_price_only[df_price_only['quote_date'].isin(relevant_dates)].copy()

    # Create 2-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Left axis: Cumulative Return
    ax1.plot(df_plot['quote_date'], df_plot['cumulative_return'],
             label='Cumulative Return', color='blue')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative Return", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Right axis: Cumulative Sentiment
    ax2.plot(df_plot['quote_date'], df_plot['cumulative_sentiment'],
             label='Cumulative Sentiment', color='red')
    ax2.set_ylabel("Cumulative Sentiment", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Build a combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    ax1.set_title(f"Last {n_days} Days: Cumulative Return vs. Combined Sentiment (with PCR)")
    ax1.grid(True)

    # Return df_plot and the figure
    return df_plot, fig


'''if __name__ == "__main__":
    # If you run main.py directly, we just run the pipeline for last 10 days
    df_plot, fig = run_pipeline(n_days=10)
    if fig is not None:
        plt.show()
'''