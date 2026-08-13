# Databricks notebook source
# DBTITLE 1,Install dependencies
# MAGIC %pip install "anthropic<0.42.0"

# COMMAND ----------

# DBTITLE 1,Get parameters from job
# Define parameter widgets
dbutils.widgets.text("input_dataset_path", "/Workspace/Users/<your-databricks-username>/token_strategy_comparison/input_dataset/reviews_subset_50.csv")
dbutils.widgets.text("output_folder_path", "/Workspace/Users/<your-databricks-username>/token_strategy_comparison/output_result")
dbutils.widgets.text("output_name", "dynamic_prompt")

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

# Dynamic prompt thresholds
SHORT_REVIEW_THRESHOLD = 100  # characters
LONG_REVIEW_THRESHOLD = 300   # characters

# Setup client
client = setup_anthropic_client(API_KEY)

# COMMAND ----------

# DBTITLE 1,Dynamic Prompt Construction Experiment
def construct_dynamic_prompt(review_text):
    """
    Construct a prompt dynamically based on review characteristics.
    - Short reviews (<100 chars): Ultra-minimal prompt
    - Medium reviews (100-300 chars): Standard prompt
    - Long reviews (>300 chars): Detailed prompt with focus instruction
    """
    review_length = len(review_text)
    
    if review_length < SHORT_REVIEW_THRESHOLD:
        # Ultra-minimal for short reviews
        prompt = f"Sentiment: {review_text}\n\nAnswer (Positive/Negative/Mixed/Neutral):"
        prompt_type = "minimal"
    
    elif review_length < LONG_REVIEW_THRESHOLD:
        # Standard prompt for medium reviews
        prompt = f"""Analyze sentiment of this review. Start with "Sentiment: [Positive/Negative/Mixed/Neutral]".

Review: {review_text}

Sentiment:"""
        prompt_type = "standard"
    
    else:
        # Detailed prompt for long reviews - ask for overall sentiment to avoid confusion
        prompt = f"""This is a longer review. Focus on the OVERALL sentiment (not individual aspects).

Review: {review_text}

Overall Sentiment (Positive/Negative/Mixed/Neutral):"""
        prompt_type = "detailed"
    
    return prompt, prompt_type

def experiment_dynamic_prompt(input_df, client, input_cost_per_token, output_cost_per_token, model, max_tokens):
    """
    Run dynamic prompt construction experiment.
    Strategy: Adapt prompt complexity based on review length to optimize tokens.
    """
    results = []
    total_cost = 0
    prompt_distribution = {"minimal": 0, "standard": 0, "detailed": 0}
    
    for _, row in input_df.iterrows():
        review_text = row["Text"]
        
        # Construct prompt dynamically
        prompt, prompt_type = construct_dynamic_prompt(review_text)
        prompt_distribution[prompt_type] += 1
        
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
        normalized_sentiment = clean_sentiment(raw_sentiment)
        
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
            "review_length": len(review_text),
            "prompt_type": prompt_type
        }
        results.append(result)
        total_cost += total_cost_per_review
    
    results_df = pd.DataFrame(results)
    
    # Print prompt distribution
    print(f"\nPrompt Distribution:")
    print(f"Minimal prompts (short reviews): {prompt_distribution['minimal']}")
    print(f"Standard prompts (medium reviews): {prompt_distribution['standard']}")
    print(f"Detailed prompts (long reviews): {prompt_distribution['detailed']}")
    
    # Show average tokens by prompt type
    if len(results_df) > 0:
        print(f"\nAverage Input Tokens by Prompt Type:")
        for ptype in ['minimal', 'standard', 'detailed']:
            subset = results_df[results_df['prompt_type'] == ptype]
            if len(subset) > 0:
                print(f"{ptype.capitalize()}: {subset['input_tokens'].mean():.1f} tokens")
    
    return results_df, total_cost

print("Dynamic Prompt Construction Strategy:")
print(f"- Short reviews (<{SHORT_REVIEW_THRESHOLD} chars): Minimal prompt")
print(f"- Medium reviews ({SHORT_REVIEW_THRESHOLD}-{LONG_REVIEW_THRESHOLD} chars): Standard prompt")
print(f"- Long reviews (>{LONG_REVIEW_THRESHOLD} chars): Detailed prompt with focus")
print("- Goal: Optimize prompt length for each review type")

# COMMAND ----------

# DBTITLE 1,Run experiment and save results
# Load input data
input_df = load_input_dataset(input_dataset_path)

print(f"Total reviews: {len(input_df)}")
print(f"\nReview Length Distribution:")
print(f"Short (<{SHORT_REVIEW_THRESHOLD} chars): {len(input_df[input_df['Text'].str.len() < SHORT_REVIEW_THRESHOLD])}")
print(f"Medium ({SHORT_REVIEW_THRESHOLD}-{LONG_REVIEW_THRESHOLD} chars): {len(input_df[(input_df['Text'].str.len() >= SHORT_REVIEW_THRESHOLD) & (input_df['Text'].str.len() < LONG_REVIEW_THRESHOLD)])}")
print(f"Long (>{LONG_REVIEW_THRESHOLD} chars): {len(input_df[input_df['Text'].str.len() >= LONG_REVIEW_THRESHOLD])}")

# Run dynamic prompt experiment
results, total_cost = experiment_dynamic_prompt(
    input_df,
    client,
    INPUT_COST_PER_TOKEN,
    OUTPUT_COST_PER_TOKEN,
    MODEL,
    MAX_TOKENS
)

# Save both raw and normalized results
print("\nSaving results...")
save_raw_and_normalized_results(results, output_name, output_folder_path)

# Print summary
print_experiment_summary(
    results,
    total_cost,
    len(input_df),
    "Dynamic Prompt Construction",
    model=MODEL,
    max_tokens=MAX_TOKENS,
    strategy="Adapt prompt complexity based on review length"
)

# Display sample results
print("\nSample Results:")
display(results[['review_id', 'review_length', 'prompt_type', 'input_tokens', 'output_tokens', 'sentiment']].head(10))

# COMMAND ----------

# DBTITLE 1,Join and display reviews with sentiment
result = join_and_display_results(input_df, results)
display(result)

# COMMAND ----------

