# Databricks notebook source
# DBTITLE 1,Data Loading and Saving Functions
import pandas as pd
import anthropic
import re

def load_input_dataset(input_path):
    """
    Load input dataset from CSV file.
    
    Args:
        input_path: Full path to the input CSV file
    
    Returns:
        DataFrame containing the input data
    """
    return pd.read_csv(input_path)

def save_results(results_df, output_name, output_folder_path):
    """
    Save results DataFrame to CSV file.
    
    Args:
        results_df: DataFrame containing the results
        output_name: Name for the output file (without .csv extension)
        output_folder_path: Full path to the output folder
    
    Returns:
        Full path to the saved file
    """
    output_path = f"{output_folder_path}/{output_name}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return output_path

def save_raw_and_normalized_results(results_df, strategy_name, output_folder_path):
    """
    Save both raw API results and normalized results.
    
    Args:
        results_df: DataFrame containing the results with 'raw_sentiment' and 'sentiment' columns
        strategy_name: Name of the strategy (e.g., 'baseline', 'batch_processing')
        output_folder_path: Full path to the output folder
    
    Returns:
        Tuple of (raw_results_path, normalized_results_path)
    """
    # Save raw results (with raw_sentiment column)
    raw_results_path = save_results(results_df, f"{strategy_name}_raw_results", output_folder_path)
    
    # Save normalized results (replace sentiment with just the normalized version)
    normalized_df = results_df.copy()
    if 'raw_sentiment' in normalized_df.columns:
        normalized_df = normalized_df.drop('raw_sentiment', axis=1)
    normalized_results_path = save_results(normalized_df, f"{strategy_name}_normalized_results", output_folder_path)
    
    return raw_results_path, normalized_results_path

# COMMAND ----------

# DBTITLE 1,API Client Setup
def setup_anthropic_client(api_key):
    """
    Set up and return Anthropic API client.
    
    Args:
        api_key: Anthropic API key
    
    Returns:
        Configured Anthropic client
    """
    return anthropic.Anthropic(api_key=api_key)

# COMMAND ----------

# DBTITLE 1,Sentiment Normalization Functions
def normalize_sentiment(sentiment_text):
    """
    Extract and normalize sentiment from API response text.
    Used for baseline strategy (single review per request).
    
    Args:
        sentiment_text: Raw text response from API
    
    Returns:
        Normalized sentiment label (Positive, Negative, Mixed, Neutral, or Unknown)
    """
    if pd.isna(sentiment_text) or not sentiment_text:
        return 'Unknown'
    
    # Convert to string and clean
    sentiment_text = str(sentiment_text).strip()
    
    # Try multiple extraction patterns in order of specificity
    patterns = [
        r'Overall Sentiment:\s*\*\*([^\*]+)\*\*',  # **Sentiment**
        r'Overall Sentiment:\s*([^\n\*#]+)',        # Plain text
        r'Sentiment:\s*\*\*([^\*]+)\*\*',          # **Sentiment**
        r'Sentiment:\s*([^\n\*#]+)',                # Plain text
        r'^\s*\*\*([^\*]+)\*\*',                   # Starts with **word**
        r'^\s*([A-Z][a-z]+)',                       # Starts with capitalized word
    ]
    
    sentiment = None
    for pattern in patterns:
        match = re.search(pattern, sentiment_text, re.MULTILINE)
        if match:
            sentiment = match.group(1).strip()
            break
    
    # If no pattern matched, try to find sentiment keywords anywhere in text
    if not sentiment:
        text_lower = sentiment_text.lower()
        # Check for explicit sentiment words at start of lines or sentences
        for keyword in ['positive', 'negative', 'mixed', 'neutral']:
            if keyword in text_lower[:100]:  # Check first 100 chars
                sentiment = keyword
                break
    
    if not sentiment:
        return 'Unknown'
    
    # Remove any markdown formatting
    sentiment = sentiment.replace('**', '').replace('*', '').replace('#', '').strip()
    
    # Skip if empty or just whitespace
    if not sentiment or sentiment.isspace():
        return 'Unknown'
    
    # Normalize to core sentiment words: Positive, Negative, Mixed, Neutral
    sentiment_lower = sentiment.lower()
    if 'positive' in sentiment_lower and 'negative' not in sentiment_lower:
        if 'mixed' in sentiment_lower or 'caveat' in sentiment_lower:
            return 'Mixed'
        else:
            return 'Positive'
    elif 'negative' in sentiment_lower and 'positive' not in sentiment_lower:
        return 'Negative'
    elif 'mixed' in sentiment_lower or ('positive' in sentiment_lower and 'negative' in sentiment_lower):
        return 'Mixed'
    elif 'neutral' in sentiment_lower:
        return 'Neutral'
    elif sentiment_lower in ['positive', 'negative', 'mixed', 'neutral']:
        return sentiment.capitalize()
    # If still not normalized, keep as Unknown
    elif len(sentiment) < 3 or not sentiment[0].isalpha():
        return 'Unknown'
    
    return sentiment

def clean_sentiment(text):
    """
    Clean and normalize sentiment text.
    Used for post-processing and comparison.
    
    Args:
        text: Sentiment text (can be None or NaN)
    
    Returns:
        Normalized sentiment label (Positive, Negative, Mixed, Neutral, or Unknown)
    """
    if pd.isna(text):
        return 'Unknown'
    
    # Remove markdown formatting
    text = str(text).replace('**', '').replace('*', '').strip()
    
    # Normalize to core sentiment words
    text_lower = text.lower()
    if 'positive' in text_lower and 'negative' not in text_lower:
        if 'mixed' in text_lower or 'caveat' in text_lower:
            return 'Mixed'
        else:
            return 'Positive'
    elif 'negative' in text_lower and 'positive' not in text_lower:
        return 'Negative'
    elif 'mixed' in text_lower or ('positive' in text_lower and 'negative' in text_lower):
        return 'Mixed'
    elif 'neutral' in text_lower:
        return 'Neutral'
    elif text_lower in ['positive', 'negative', 'mixed', 'neutral']:
        return text.capitalize()
    else:
        return text if text else 'Unknown'

# COMMAND ----------

# DBTITLE 1,Results Display Functions
def print_experiment_summary(results_df, total_cost, num_records, strategy_name, **kwargs):
    """
    Print a standardized summary of experiment results.
    
    Args:
        results_df: DataFrame containing the experiment results
        total_cost: Total cost of the experiment
        num_records: Number of records processed
        strategy_name: Name of the strategy (e.g., "Baseline", "Batch Processing")
        **kwargs: Additional strategy-specific parameters to display
    """
    print(f"\n{'='*60}")
    print(f"{strategy_name.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"Number of records processed: {num_records}")
    
    # Print any strategy-specific parameters
    for key, value in kwargs.items():
        formatted_key = key.replace('_', ' ').title()
        print(f"{formatted_key}: {value}")
    
    print(f"\nToken Usage:")
    print(f"  Total input tokens: {results_df['input_tokens'].sum():,}")
    print(f"  Total output tokens: {results_df['output_tokens'].sum():,}")
    print(f"  Average input tokens per review: {results_df['input_tokens'].mean():.1f}")
    print(f"  Average output tokens per review: {results_df['output_tokens'].mean():.1f}")
    
    print(f"\nCost Analysis:")
    print(f"  Total cost: ${total_cost:.6f}")
    print(f"  Average cost per review: ${results_df['total_cost'].mean():.6f}")
    print(f"  Total input cost: ${results_df['input_cost'].sum():.6f}")
    print(f"  Total output cost: ${results_df['output_cost'].sum():.6f}")
    
    print(f"\nPerformance:")
    print(f"  Average processing time per review: {results_df['processing_time'].mean():.3f}s")
    print(f"{'='*60}")

def join_and_display_results(input_df, results_df):
    """
    Join original reviews with sentiment analysis results and display.
    
    Args:
        input_df: Original input DataFrame with review data
        results_df: Results DataFrame with sentiment analysis
    
    Returns:
        Merged DataFrame with reviews and sentiment results
    """
    reviews_with_sentiment = input_df.merge(
        results_df[['review_id', 'sentiment', 'input_tokens', 'output_tokens', 'total_cost']],
        left_on='Id',
        right_on='review_id',
        how='inner'
    )
    
    result = reviews_with_sentiment[['Id', 'Text', 'sentiment', 'input_tokens', 'output_tokens', 'total_cost']]
    return result

# COMMAND ----------

