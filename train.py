import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import json
import os
import argparse
import inspect
from preprocess import preprocess_text
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    # pred.predictions can be a tuple (logits, hidden_states, attentions)
    if isinstance(pred.predictions, tuple):
        logits = pred.predictions[0]
    else:
        logits = pred.predictions
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train(data_path, epochs=3, batch_size=16):
    # Load data
    texts = []
    labels = []
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ';' in line:
                parts = line.strip().rsplit(';', 1)
                if len(parts) == 2:
                    text, label = parts
                    # Normalize text and labels
                    processed_text = preprocess_text(text)
                    processed_label = label.strip().lower()
                    if processed_text and processed_label:
                        texts.append(processed_text)
                        labels.append(processed_label)

    if not texts:
        print("Error: No valid data found in the file.")
        return

    unique_labels = sorted(list(set(labels)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {str(i): label for label, i in label2id.items()}

    num_labels = len(unique_labels)
    label_ids = [label2id[l] for l in labels]

    # Create dataset
    df = pd.DataFrame({'text': texts, 'label': label_ids})
    dataset = Dataset.from_pandas(df)

    # Ensure we have enough samples to split
    if len(dataset) >= 2:
        # If dataset is very small, ensure at least 1 sample in test
        test_size = max(0.1, 1.0/len(dataset)) if len(dataset) < 10 else 0.2
        dataset = dataset.train_test_split(test_size=test_size)
        train_ds = dataset["train"]
        test_ds = dataset["test"]
    else:
        train_ds = dataset
        test_ds = dataset

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    train_tokenized = train_ds.map(tokenize_function, batched=True)
    test_tokenized = test_ds.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    output_dir = "./models/sentiment_model"
    os.makedirs(output_dir, exist_ok=True)

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    training_args = TrainingArguments(**build_training_args_kwargs(epochs, batch_size, len(dataset) >= 2))

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
)

    

    trainer.train()

    # Save model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping explicitly
    with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding='utf-8') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    # Save final metrics
    try:
        eval_results = trainer.evaluate()
        with open("./outputs/metrics.json", "w", encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2)
    except Exception as e:
        print(f"Could not save metrics: {e}")

    print(f"Training complete. Model saved to {output_dir}")


def build_training_args_kwargs(epochs, batch_size, use_eval):
    """Create TrainingArguments kwargs compatible across transformers versions."""
    kwargs = {
        "output_dir": "./outputs",
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "warmup_steps": 0,
        "weight_decay": 0.01,
        "logging_dir": "./outputs/logs",
        "logging_steps": 1,
    }

    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_value = "epoch" if use_eval else "no"

    # evaluation_strategy was renamed to eval_strategy in newer versions.
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_value

    # These options are only valid when evaluation/checkpointing is enabled.
    if use_eval:
        if "save_strategy" in params:
            kwargs["save_strategy"] = "epoch"
        if "load_best_model_at_end" in params:
            kwargs["load_best_model_at_end"] = True
    else:
        if "save_strategy" in params:
            kwargs["save_strategy"] = "no"
        if "load_best_model_at_end" in params:
            kwargs["load_best_model_at_end"] = False

    return kwargs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train.txt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size)
