# Databricks notebook source
# DBTITLE 1,Install dependencies
# MAGIC %pip install "anthropic<0.42.0"

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Get parameters from job
# Define parameter widgets
dbutils.widgets.text("input_dataset_path", "/Workspace/Users/<your-databricks-username>/token_strategy_comparison/input_dataset/reviews_subset_50.csv")
dbutils.widgets.text("output_folder_path", "/Workspace/Users/<your-databricks-username>/token_strategy_comparison/output_result")
dbutils.widgets.text("output_name", "batch_processing")

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
# Extract dataset name from path
import os
INPUT_DATASET_NAME = os.path.splitext(os.path.basename(input_dataset_path))[0]
API_KEY = dbutils.secrets.get(scope="token_strategy_comparison", key="anthropic_api_key")  # set via Databricks secret scope, never hardcode
MODEL = "claude-haiku-4-5-20251001"
INPUT_COST_PER_TOKEN = 0.00042
OUTPUT_COST_PER_TOKEN = 0.00128
MAX_TOKENS = 100
BATCH_SIZE = 5

# Setup client
client = setup_anthropic_client(API_KEY)

# COMMAND ----------

# DBTITLE 1,Batch Processing Experiment - Multiple Reviews Per Request
def experiment_batch_processing(input_df, client, input_cost_per_token, output_cost_per_token, model, max_tokens, batch_size):
    """
    Run batch processing experiment: multiple reviews per API call.
    """
    results = []
    total_cost = 0
    
    # Process reviews in batches
    for i in range(0, len(input_df), batch_size):
        batch = input_df.iloc[i:i+batch_size]
        
        # Build batch prompt with multiple reviews - IMPROVED PROMPT
        batch_prompt = """Analyze the sentiment of each review below and return ONLY the ID and sentiment.

IMPORTANT INSTRUCTIONS:
- Return EXACTLY one line per review
- Use this EXACT format: ID 123: Positive
- Do NOT use markdown formatting (no ** or *)
- Do NOT add explanations, bullet points, or analysis
- Only use these sentiments: Positive, Negative, Mixed, Neutral

Reviews:
"""
        for idx, (_, row) in enumerate(batch.iterrows(), 1):
            batch_prompt += f"{idx}. ID {row['Id']}: {row['Text']}\n\n"
        
        batch_prompt += "\nReturn format (one line per review):\nID 123: Positive\nID 456: Negative"
        
        start_time = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": batch_prompt
            }]
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        # Parse the batch response - IMPROVED PARSER
        sentiment_text = response.content[0].text.strip()
        
        # Create a mapping of review_id to sentiment using robust regex
        sentiment_map = {}
        # Pattern handles: "ID 123: Positive", "**ID 123:** Positive", "ID 123: **Positive**", etc.
        pattern = r'ID\s*(\d+)\s*:?\s*\*?\*?\s*(Positive|Negative|Mixed|Neutral)'
        
        for match in re.finditer(pattern, sentiment_text, re.IGNORECASE):
            try:
                review_id = int(match.group(1))
                sentiment = match.group(2).strip().capitalize()
                sentiment_map[review_id] = sentiment
            except (ValueError, IndexError):
                continue
        
        # Distribute tokens across reviews in batch
        tokens_per_review_input = input_tokens / len(batch)
        tokens_per_review_output = output_tokens / len(batch)
        
        for _, row in batch.iterrows():
            review_id = row['Id']
            
            # Extract sentiment for this specific review ID
            extracted_sentiment = sentiment_map.get(review_id, None)
            
            # Store only this review's sentiment, not the entire batch response
            if extracted_sentiment:
                raw_sentiment = f"ID {review_id}: {extracted_sentiment}"
                normalized_sentiment = extracted_sentiment
            else:
                raw_sentiment = f"ID {review_id}: [extraction failed]"
                normalized_sentiment = 'Unknown'
            
            input_cost = tokens_per_review_input * input_cost_per_token
            output_cost = tokens_per_review_output * output_cost_per_token
            total_cost_per_review = input_cost + output_cost
            
            result = {
                "review_id": review_id,
                "model": model,
                "processing_time": processing_time / len(batch),
                "input_tokens": int(tokens_per_review_input),
                "output_tokens": int(tokens_per_review_output),
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost_per_review,
                "raw_sentiment": raw_sentiment,  # Store only this review's sentiment
                "sentiment": normalized_sentiment,
                "batch_size": len(batch)
            }
            results.append(result)
            total_cost += total_cost_per_review
        
        # Validation: Log if any reviews in batch didn't get extracted
        missing_ids = set(batch['Id']) - set(sentiment_map.keys())
        if missing_ids:
            print(f"⚠️  Warning: Failed to extract sentiment for IDs: {missing_ids}")
            print(f"   Full batch response was: {sentiment_text[:200]}...")
    
    results_df = pd.DataFrame(results)
    return results_df, total_cost

print("Running Batch Processing Experiment...")
print(f"Batch size: {BATCH_SIZE} reviews per API call")

# COMMAND ----------

# DBTITLE 1,Run experiment and save results
# Load input data
input_df = load_input_dataset(input_dataset_path)

print(f"Total reviews: {len(input_df)}")
print(f"Expected API calls: {(len(input_df) + BATCH_SIZE - 1) // BATCH_SIZE}")

# Run batch processing experiment
batch_results, batch_total_cost = experiment_batch_processing(
    input_df,
    client,
    INPUT_COST_PER_TOKEN,
    OUTPUT_COST_PER_TOKEN,
    MODEL,
    MAX_TOKENS,
    BATCH_SIZE
)

# Save both raw and normalized results
print("\nSaving results...")
save_raw_and_normalized_results(batch_results, output_name, output_folder_path)

# Print summary
print_experiment_summary(
    batch_results,
    batch_total_cost,
    len(input_df),
    "Batch Processing Optimization",
    model=MODEL,
    batch_size=BATCH_SIZE,
    api_calls=(len(input_df) + BATCH_SIZE - 1) // BATCH_SIZE,
    strategy="Multiple reviews per API call"
)

# Display sample results
print("\nSample Results:")
display(batch_results[['review_id', 'input_tokens', 'output_tokens', 'total_cost', 'batch_size']].head(10))

# COMMAND ----------

# DBTITLE 1,Join and display reviews with sentiment
result = join_and_display_results(input_df, batch_results)
display(result)