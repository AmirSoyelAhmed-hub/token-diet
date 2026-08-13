# Databricks notebook source
# DBTITLE 1,Install dependencies
# MAGIC %pip install "anthropic<0.42.0"

# COMMAND ----------

# DBTITLE 1,Get parameters from job
# Define parameter widgets
dbutils.widgets.text("input_dataset_path", "/Workspace/Users/<your-databricks-username>/token_strategy_comparison/input_dataset/reviews_subset_50.csv")
dbutils.widgets.text("output_folder_path", "/Workspace/Users/<your-databricks-username>/token_strategy_comparison/output_result")
dbutils.widgets.text("output_name", "preprocessing_filtering")

# Get parameter values
input_dataset_path = dbutils.widgets.get("input_dataset_path")
output_folder_path = dbutils.widgets.get("output_folder_path")
output_name = dbutils.widgets.get("output_name")

print(f"Input dataset: {input_dataset_path}")
print(f"Output folder: {output_folder_path}")
print(f"Output name: {output_name}")

# COMMAND ----------

# DBTITLE 1,Import utilities notebook
# MAGIC %run "/Users/<your-databricks-username>/token_strategy_comparison/utilities"

# COMMAND ----------

# DBTITLE 1,Setup configuration
import time
import pandas as pd
import re

# Configuration
import os
INPUT_DATASET_NAME = os.path.splitext(os.path.basename(input_dataset_path))[0]
API_KEY = dbutils.secrets.get(scope="token_strategy_comparison", key="anthropic_api_key")  # set via Databricks secret scope, never hardcode
MODEL = "claude-haiku-4-5-20251001"
INPUT_COST_PER_TOKEN = 0.00042
OUTPUT_COST_PER_TOKEN = 0.00128
MAX_TOKENS = 100
MAX_REVIEW_LENGTH = 500  # Truncate reviews to this many characters

# Setup client
client = setup_anthropic_client(API_KEY)

# COMMAND ----------

# DBTITLE 1,Preprocessing and Filtering Experiment
def preprocess_review(text, max_length=500):
    """
    Preprocess review text to reduce input tokens:
    - Remove extra whitespace
    - Remove special characters
    - Truncate to max length
    - Remove duplicate sentences
    """
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Remove excessive punctuation (keep single instances)
    text = re.sub(r'([!?.])\1+', r'\1', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Truncate if too long (keep first N characters, they're usually most important)
    if len(text) > max_length:
        # Try to cut at a sentence boundary
        text = text[:max_length]
        last_period = text.rfind('.')
        last_exclaim = text.rfind('!')
        last_question = text.rfind('?')
        cut_point = max(last_period, last_exclaim, last_question)
        if cut_point > max_length * 0.7:  # Only cut if we keep at least 70%
            text = text[:cut_point + 1]
    
    return text.strip()

def experiment_preprocessing_filtering(input_df, client, input_cost_per_token, output_cost_per_token, model, max_tokens, max_review_length):
    """
    Run preprocessing and filtering experiment.
    Strategy: Clean and truncate reviews before sending to reduce input tokens.
    """
    results = []
    total_cost = 0
    token_savings = []
    
    for _, row in input_df.iterrows():
        original_text = row["Text"]
        
        # Preprocess the review
        preprocessed_text = preprocess_review(original_text, max_review_length)
        
        # Calculate token savings (estimate)
        original_tokens_est = len(original_text) // 4
        preprocessed_tokens_est = len(preprocessed_text) // 4
        tokens_saved = original_tokens_est - preprocessed_tokens_est
        token_savings.append(tokens_saved)
        
        # Standard prompt with preprocessed text
        prompt = f"""Analyze the sentiment of this review and start your response with "Overall Sentiment: [Positive/Negative/Mixed/Neutral]".

Review: {preprocessed_text}

Overall Sentiment:"""
        
        start_time = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_cost = input_tokens * input_cost_per_token
        output_cost = output_tokens * output_cost_per_token
        total_cost_per_review = input_cost + output_cost
        
        # Extract and normalize sentiment
        raw_sentiment = response.content[0].text.strip()
        normalized_sentiment = normalize_sentiment(raw_sentiment)
        
        result = {
            "review_id": row["Id"],
            "model": model,
            "processing_time": processing_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost_per_review,
            "raw_sentiment": raw_sentiment,
            "sentiment": normalized_sentiment,
            "original_length": len(original_text),
            "preprocessed_length": len(preprocessed_text),
            "tokens_saved_est": tokens_saved
        }
        results.append(result)
        total_cost += total_cost_per_review
    
    results_df = pd.DataFrame(results)
    
    # Print preprocessing stats
    print(f"\nPreprocessing Statistics:")
    print(f"Average original length: {results_df['original_length'].mean():.0f} chars")
    print(f"Average preprocessed length: {results_df['preprocessed_length'].mean():.0f} chars")
    print(f"Average reduction: {(1 - results_df['preprocessed_length'].mean() / results_df['original_length'].mean()) * 100:.1f}%")
    print(f"Estimated tokens saved per review: {sum(token_savings) / len(token_savings):.1f}")
    
    return results_df, total_cost

print("Preprocessing and Filtering Strategy:")
print(f"- Truncate reviews to max {MAX_REVIEW_LENGTH} characters")
print("- Remove extra whitespace, URLs, emails")
print("- Clean excessive punctuation")
print("- Goal: Reduce input tokens while preserving sentiment")

# COMMAND ----------

# DBTITLE 1,Run experiment and save results
# Load input data
input_df = load_input_dataset(input_dataset_path)

print(f"Total reviews: {len(input_df)}")

# Run preprocessing and filtering experiment
results, total_cost = experiment_preprocessing_filtering(
    input_df,
    client,
    INPUT_COST_PER_TOKEN,
    OUTPUT_COST_PER_TOKEN,
    MODEL,
    MAX_TOKENS,
    MAX_REVIEW_LENGTH
)

# Save both raw and normalized results
print("\nSaving results...")
save_raw_and_normalized_results(results, output_name, output_folder_path)

# Print summary
print_experiment_summary(
    results,
    total_cost,
    len(input_df),
    "Preprocessing and Filtering",
    model=MODEL,
    max_tokens=MAX_TOKENS,
    strategy="Clean and truncate reviews to reduce input tokens"
)

# Display sample results
print("\nSample Results:")
display(results[['review_id', 'original_length', 'preprocessed_length', 'tokens_saved_est', 'input_tokens', 'sentiment']].head(10))

# COMMAND ----------

# DBTITLE 1,Join and display reviews with sentiment
result = join_and_display_results(input_df, results)
display(result)

# COMMAND ----------

