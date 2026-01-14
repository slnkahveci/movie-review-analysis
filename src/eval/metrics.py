"""Evaluation metrics for transformer model."""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import Trainer
from typing import Dict, List
import pandas as pd


def compute_metrics(trainer: Trainer, dataset, device: str = "cpu") -> Dict:
    """
    Compute accuracy, F1, confusion matrix, and perplexity.

    Args:
        trainer: HuggingFace Trainer instance
        dataset: Dataset to evaluate
        device: Device to use for inference

    Returns:
        Dictionary with metrics and predictions
    """
    model = trainer.model
    model.eval()

    all_preds = []
    all_labels = []
    all_loss = []

    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            input_ids = torch.tensor([batch["input_ids"]]).to(device)
            attention_mask = torch.tensor([batch["attention_mask"]]).to(device)
            labels = torch.tensor([batch["labels"]]).to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            all_loss.append(outputs.loss.item())
            all_preds.append(torch.argmax(outputs.logits, dim=-1).item())
            all_labels.append(labels.item())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    conf_matrix = confusion_matrix(all_labels, all_preds)
    perplexity = np.exp(np.mean(all_loss))

    return {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": conf_matrix.tolist(),
        "perplexity": perplexity,
        "predictions": all_preds,
        "labels": all_labels,
    }


def print_metrics(metrics: Dict, split_name: str):
    """Pretty print metrics."""
    print(f"\n{split_name} Metrics:")
    print("-" * 30)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")


def create_evaluation_dataframe(
    reviews: List[str],
    predictions: List[int],
    gold_labels: List[int],
) -> pd.DataFrame:
    """
    Create a DataFrame for evaluation.

    Args:
        reviews: List of review texts
        predictions: Model predictions (0/1)
        gold_labels: Ground truth labels (0/1)

    Returns:
        DataFrame with review, prediction, gold label columns
    """
    label_map = {0: "negative", 1: "positive"}

    return pd.DataFrame({
        "review": reviews,
        "model_prediction": [label_map[p] for p in predictions],
        "gold_label": [label_map[l] for l in gold_labels],
        "human_rating": [""] * len(reviews),  # To be filled manually
        "llm_judge": [""] * len(reviews),  # To be filled by LLM
    })
