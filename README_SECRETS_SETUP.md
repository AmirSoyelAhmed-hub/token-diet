# Before running these notebooks

These notebooks read the Anthropic API key from a Databricks secret scope —
no key is ever hardcoded in the code.

## One-time setup (Databricks CLI)

```bash
databricks secrets create-scope token_strategy_comparison
databricks secrets put-secret token_strategy_comparison anthropic_api_key
# paste your key when prompted
```

## Widget paths

Each notebook takes `input_dataset_path` / `output_folder_path` as job widgets.
Replace `<your-databricks-username>` in the default widget values with your
own Databricks workspace username, or override them at run time.
