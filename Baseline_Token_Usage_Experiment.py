# Databricks notebook source
# DBTITLE 1,Install dependencies
# MAGIC %pip install "anthropic<0.42.0"

# COMMAND ----------

# DBTITLE 1,Get parameters from job
# Define parameter widgets
dbutils.widgets.text("input_dataset_path", "/Workspace/Users/<your-databricks-username>/token_strategy_comparison/input_dataset/reviews_subset_50.csv")
dbutils.widgets.text("output_folder_path", "/Workspace/Users/<your-databricks-username>/token_strategy_comparison/output_result")
dbutils.widgets.text("output_name", "baseline")

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

# DBTITLE 1,Import utilities and setup
import time
import pandas as pd

# Configuration
# Extract dataset name from path
import os
INPUT_DATASET_NAME = os.path.splitext(os.path.basename(input_dataset_path))[0]
API_KEY = dbutils.secrets.get(scope="token_strategy_comparison", key="anthropic_api_key")  # set via Databricks secret scope, never hardcode
MODEL = "claude-haiku-4-5-20251001"
INPUT_COST_PER_TOKEN = 0.001
OUTPUT_COST_PER_TOKEN = 0.005
MAX_TOKENS = 100

# Setup client
client = setup_anthropic_client(API_KEY)

# COMMAND ----------

# DBTITLE 1,Baseline Experiment - Single Review Per Request
def experiment_baseline(input_df, client, input_cost_per_token, output_cost_per_token, model, max_tokens):
    """
    Run baseline experiment: one API call per review.
    """
    results = []
    total_cost = 0
    
    for _, row in input_df.iterrows():
        review_text = row["Text"]
        
        start_time = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": f"""Analyze the sentiment of this review and start your response with "Overall Sentiment: [Positive/Negative/Mixed/Neutral]".

Review: {review_text}

Overall Sentiment:"""
            }]
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_cost = input_tokens * input_cost_per_token
        output_cost = output_tokens * output_cost_per_token
        total_cost_per_review = input_cost + output_cost
        
        # Extract raw sentiment and normalize it
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
            "sentiment": normalized_sentiment
        }
        results.append(result)
        total_cost += total_cost_per_review
    
    results_df = pd.DataFrame(results)
    return results_df, total_cost

# COMMAND ----------

# DBTITLE 1,Run experiment and save results
# Load input data
input_df = load_input_dataset(input_dataset_path)

# Run baseline experiment
baseline_results, baseline_total_cost = experiment_baseline(
    input_df, 
    client, 
    INPUT_COST_PER_TOKEN, 
    OUTPUT_COST_PER_TOKEN, 
    MODEL, 
    MAX_TOKENS
)

# Save both raw and normalized results
print("\nSaving results...")
save_raw_and_normalized_results(baseline_results, output_name, output_folder_path)

# Print summary
print_experiment_summary(
    baseline_results, 
    baseline_total_cost, 
    len(input_df), 
    "Baseline Experiment",
    model=MODEL,
    strategy="One API call per review"
)

# COMMAND ----------

# DBTITLE 1,Display detailed results
display(baseline_results)

# COMMAND ----------

# DBTITLE 1,Join and display reviews with sentiment
result = join_and_display_results(input_df, baseline_results)
display(result)

# COMMAND ----------

