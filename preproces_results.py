import ast
import numpy as np

def read_metrics(file_path):
    train_loss = []
    val_metric = []
    perplexity = []

    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip the first line

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = ast.literal_eval(line)
                if 'loss' in data:
                    train_loss.append(data['loss'])
                elif 'eval_masked_lm_accuracy' in data:
                    val_metric.append(data['eval_masked_lm_accuracy'])
                    perplexity.append(data['eval_masked_lm_perplexity'])
            except Exception as e:
                print(f"Failed to parse line: {line}\nError: {e}")

    return {
        'train_loss': train_loss[:200],
        'val_metric': val_metric[:200],
        'val_perplexity': perplexity[:200]
    }

# Example usage
if __name__ == "__main__":
    metrics = read_metrics("results_save/experiments_2/outputwikitextbert-lsuv_with_dataloader-adamw_torch.log")
    print(metrics)
