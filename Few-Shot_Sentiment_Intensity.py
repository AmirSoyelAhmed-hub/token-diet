# Databricks notebook source
# DBTITLE 1,Install dependencies
# MAGIC %pip install "anthropic<0.42.0"

# COMMAND ----------

# DBTITLE 1,Get parameters from job
# Define parameter widgets
dbutils.widgets.text("input_dataset_path", "/Workspace/Users/<your-databricks-username>/token_strategy_comparison/input_dataset/reviews_subset_50.csv")
dbutils.widgets.text("output_folder_path", "/Workspace/Users/<your-databricks-username>/token_strategy_comparison/output_result")
dbutils.widgets.text("output_name", "few_shot_intensity")
dbutils.widgets.text("model", "claude-haiku-4-5-20251001")

# Get parameter values
input_dataset_path = dbutils.widgets.get("input_dataset_path")
output_folder_path = dbutils.widgets.get("output_folder_path")
output_name = dbutils.widgets.get("output_name")
selected_model = dbutils.widgets.get("model")

print(f"Input dataset: {input_dataset_path}")
print(f"Output folder: {output_folder_path}")
print(f"Output name: {output_name}")
print(f"Model: {selected_model}")

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
MODEL = selected_model
MAX_TOKENS = 10  # Single digit output needs very few tokens

# Model-specific pricing (per 1M tokens)
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.00042, "output": 0.00128},
    "claude-sonnet-5": {"input": 0.002, "output": 0.010},
    "claude-opus-5": {"input": 0.005, "output": 0.025},
}

# Get pricing for selected model
if MODEL in MODEL_PRICING:
    INPUT_COST_PER_TOKEN = MODEL_PRICING[MODEL]["input"]
    OUTPUT_COST_PER_TOKEN = MODEL_PRICING[MODEL]["output"]
else:
    # Default to Haiku pricing
    INPUT_COST_PER_TOKEN = 0.00042
    OUTPUT_COST_PER_TOKEN = 0.00128

print(f"Using model: {MODEL}")
print(f"Input cost per token: ${INPUT_COST_PER_TOKEN}")
print(f"Output cost per token: ${OUTPUT_COST_PER_TOKEN}")
print(f"Max tokens: {MAX_TOKENS} (optimized for single digit output)")

# Setup client
client = setup_anthropic_client(API_KEY)

# COMMAND ----------

# DBTITLE 1,Few-Shot Sentiment Intensity Experiment - Reduce Input Tokens
def experiment_few_shot_intensity(input_df, client, input_cost_per_token, output_cost_per_token, model, max_tokens):
    """
    Few-Shot Learning Strategy: Reduces input tokens by 30-50%.
    
    Approach:
    - Include 2-3 examples in the prompt (few-shot learning)
    - Model learns from examples instead of lengthy instructions
    - Output: Single digit (1-5) for sentiment intensity
    - 1 = Most Negative, 5 = Most Positive
    
    Expected Savings:
    - Input tokens: 30-50% reduction vs detailed instructions
    - Output tokens: Minimal (single digit)
    - Captures nuanced sentiment strength
    """
    results = []
    total_cost = 0
    
    for _, row in input_df.iterrows():
        review_text = row["Text"]
        
        # Few-shot prompt with examples - more efficient than lengthy explanations
        prompt = f"""Analyze sentiment intensity on a scale of 1 to 5:
1 = Most Negative
5 = Most Positive

Examples:
Review: This product is amazing! I highly recommend it.
Sentiment Intensity: 5

Review: The quality is disappointing. Not worth the money.
Sentiment Intensity: 2

Review: It's okay, nothing special but does the job.
Sentiment Intensity: 3

Review: {review_text}
Sentiment Intensity:"""
        
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
        
        # Extract intensity score and convert to sentiment
        raw_sentiment = response.content[0].text.strip()
        
        # Map intensity to sentiment category
        try:
            intensity = int(re.search(r'\d+', raw_sentiment).group())
            if intensity <= 2:
                normalized_sentiment = "negative"
            elif intensity == 3:
                normalized_sentiment = "neutral"
            else:
                normalized_sentiment = "positive"
        except:
            normalized_sentiment = "unknown"
            intensity = None
        
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
            "intensity_score": intensity,
            "sentiment": normalized_sentiment
        }
        results.append(result)
        total_cost += total_cost_per_review
    
    results_df = pd.DataFrame(results)
    return results_df, total_cost

print("Few-Shot Sentiment Intensity Strategy:")
print(f"- Testing model: {MODEL}")
print("- Approach: Learn from examples (few-shot)")
print("- Output: Single digit (1-5 intensity scale)")
print("- Expected input token reduction: 30-50%")
print("- Goal: Reduce input tokens via examples instead of instructions")

# COMMAND ----------

# DBTITLE 1,Run experiment and save results
# Load input data
input_df = load_input_dataset(input_dataset_path)

print(f"Total reviews: {len(input_df)}")

# Run few-shot intensity experiment
results, total_cost = experiment_few_shot_intensity(
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
    "Few-Shot Sentiment Intensity",
    model=MODEL,
    max_tokens=MAX_TOKENS,
    strategy=f"Few-shot learning with intensity scale (1-5) - Optimized for reduced input tokens"
)

# Display sample results with intensity scores
print("\nSample Results:")
display(results[['review_id', 'model', 'processing_time', 'input_tokens', 'output_tokens', 'total_cost', 'intensity_score', 'sentiment']].head(10))

# COMMAND ----------

# DBTITLE 1,Join and display reviews with sentiment and intensity
result = join_and_display_results(input_df, results)
# Add intensity score to the display
if 'intensity_score' in results.columns:
    result_with_intensity = result.merge(
        results[['review_id', 'intensity_score']], 
        left_on='Id', 
        right_on='review_id', 
        how='left'
    )
    display(result_with_intensity)
else:
    display(result)

# COMMAND ----------

