"""Evaluation pipeline for transformer model with three evaluation approaches."""

import os
import random
import torch
import pandas as pd
from src.models.transformer import get_device, TransformerDataset
from src.data.dataloader import TextPreprocessor, SENTIMENT_TO_ID
from src.eval.metrics import compute_metrics, print_metrics, create_evaluation_dataframe
from src.eval.llm_judge import evaluate_with_llm_judge, analyze_agreement
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


def load_model(model_dir: str, device: str):
    """Load trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_transformer(
    model_dir: str = "out/minilm_imdb_model",
    data_path: str = "dataset/imdb-dataset.csv",
    sample_size: int = 1000,
    test_subsample: int = 100,
    output_csv: str = "out/evaluation_results.csv",
    skip_llm_judge: bool = False,
    skip_human_eval: bool = False,
    openrouter_api_key: str = None,
):
    """
    Run full evaluation pipeline with three approaches:
    1. Metric-based evaluation (accuracy, F1, perplexity) using gold labels
    2. Human evaluation (manual ratings by you and teammates)
    3. LLM-as-a-judge evaluation (using Llama 3.2 3B via OpenRouter free API)

    Args:
        model_dir: Path to saved model
        data_path: Path to dataset
        sample_size: Total dataset size
        test_subsample: Number of test instances to randomly subsample for evaluation
        output_csv: Output CSV file for human annotation
        skip_llm_judge: Skip LLM judge evaluation (useful if API key not available)
        skip_human_eval: Skip human evaluation CSV generation
        openrouter_api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(model_dir, device)

    # Load data
    print("\nLoading data...")
    preprocessor = TextPreprocessor(
        data_path,
        sample_size=sample_size,
        tokenizer_type="unigram",
        tokenizer=tokenizer,
    )
    df = preprocessor.load_data(remove_stopwords=False, max_length=320)
    train_df, val_df, test_df = preprocessor.get_splits()

    # Create datasets
    train_dataset = TransformerDataset(
        train_df["_input_ids"].tolist(),
        train_df["_attention_mask"].tolist(),
        train_df["sentiment"].map(SENTIMENT_TO_ID).tolist(),
    )
    val_dataset = TransformerDataset(
        val_df["_input_ids"].tolist(),
        val_df["_attention_mask"].tolist(),
        val_df["sentiment"].map(SENTIMENT_TO_ID).tolist(),
    )
    test_dataset = TransformerDataset(
        test_df["_input_ids"].tolist(),
        test_df["_attention_mask"].tolist(),
        test_df["sentiment"].map(SENTIMENT_TO_ID).tolist(),
    )

    # Create trainer (for evaluation interface)
    training_args = TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=64)
    trainer = Trainer(model=model, args=training_args)

    # ==================== PART 1: Metric-based Evaluation ====================
    print(f"\n{'='*60}")
    print("PART 1: METRIC-BASED EVALUATION (Full Splits)")
    print(f"{'='*60}")

    # Evaluate on all splits
    train_metrics = compute_metrics(trainer, train_dataset, device)
    val_metrics = compute_metrics(trainer, val_dataset, device)
    test_metrics = compute_metrics(trainer, test_dataset, device)

    print_metrics(train_metrics, "Train")
    print_metrics(val_metrics, "Validation")
    print_metrics(test_metrics, "Test")

    # ==================== PART 2: Subsample 100 Test Instances ====================
    print(f"\n{'='*60}")
    print(f"PART 2: RANDOM SUBSAMPLE ({test_subsample} Test Instances)")
    print(f"{'='*60}")

    # Random subsample from test set
    random.seed(42)
    test_indices = random.sample(range(len(test_df)), min(test_subsample, len(test_df)))
    test_subsample_df = test_df.iloc[test_indices].reset_index(drop=True)

    # Get model predictions for subsample
    subsample_reviews = []
    subsample_predictions = []
    subsample_gold_labels = []

    print(f"\nGenerating predictions for {len(test_subsample_df)} samples...")

    for idx in range(len(test_subsample_df)):
        row = test_subsample_df.iloc[idx]
        review = row["review"]
        gold_label = SENTIMENT_TO_ID[row["sentiment"]]

        # Encode and predict
        encoding = tokenizer(
            review,
            truncation=True,
            padding=False,
            max_length=320,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        subsample_reviews.append(review)
        subsample_predictions.append(prediction)
        subsample_gold_labels.append(gold_label)

    # Create evaluation DataFrame
    eval_df = create_evaluation_dataframe(
        subsample_reviews,
        subsample_predictions,
        subsample_gold_labels
    )

    # Compute metrics on subsample
    from sklearn.metrics import accuracy_score, f1_score
    subsample_accuracy = accuracy_score(subsample_gold_labels, subsample_predictions)
    subsample_f1 = f1_score(subsample_gold_labels, subsample_predictions, average="binary")

    print(f"\nSubsample Metrics (using gold labels):")
    print(f"  Accuracy: {subsample_accuracy:.4f}")
    print(f"  F1 Score: {subsample_f1:.4f}")

    # ==================== PART 3: Human Evaluation ====================
    if not skip_human_eval:
        print(f"\n{'='*60}")
        print("PART 3: HUMAN EVALUATION (Manual Rating)")
        print(f"{'='*60}")

        # Save CSV for manual annotation
        eval_df.to_csv(output_csv, index=False)
        print(f"\n✓ Saved evaluation data to: {output_csv}")
        print("\nInstructions for Human Evaluation:")
        print("  1. Open the CSV file in a spreadsheet editor")
        print("  2. Fill in the 'human_rating' column with 'positive' or 'negative'")
        print("  3. You and your teammates should rate each review independently")
        print("  4. Save the file and re-run this script to see agreement analysis")
        print("\nNote: Gold labels are provided for reference, but rate based on your judgment!")
    else:
        print(f"\n{'='*60}")
        print("PART 3: HUMAN EVALUATION (Skipped)")
        print(f"{'='*60}")
        print("\n⚠️  Human evaluation skipped (--skip_human_eval flag)")

    # ==================== PART 4: LLM-as-a-Judge Evaluation ====================
    if not skip_llm_judge:
        print(f"\n{'='*60}")
        print("PART 4: LLM-AS-A-JUDGE EVALUATION (Llama 3.2 3B)")
        print(f"{'='*60}")

        try:
            eval_df = evaluate_with_llm_judge(eval_df, api_key=openrouter_api_key)

            # Save updated CSV with LLM judgments
            eval_df.to_csv(output_csv, index=False)
            print(f"\n✓ Updated {output_csv} with LLM judgments")

        except ValueError as e:
            print(f"\n⚠️  {e}")
            print("\nTo enable LLM judge:")
            print("  1. Get free API key from: https://openrouter.ai/keys")
            print("  2. Set environment variable: export OPENROUTER_API_KEY='your-key'")
            print("  3. Re-run this script")
            skip_llm_judge = True
        except Exception as e:
            print(f"\n⚠️  Error during LLM evaluation: {e}")
            print("Skipping LLM judge evaluation.")
            skip_llm_judge = True

    # ==================== PART 5: Agreement Analysis ====================
    # Only show agreement analysis if we have LLM judge results OR human ratings
    has_llm = not skip_llm_judge and "llm_judge" in eval_df.columns
    has_human = not skip_human_eval and (eval_df["human_rating"] != "").any()

    if has_llm or has_human:
        print(f"\n{'='*60}")
        print("PART 5: CROSS-EVALUATION AGREEMENT ANALYSIS")
        print(f"{'='*60}")

        # Reload CSV if user has filled human ratings
        if not skip_human_eval and os.path.exists(output_csv):
            eval_df_reloaded = pd.read_csv(output_csv)

            # Check if human ratings were added
            if (eval_df_reloaded["human_rating"].notna().any() and
                (eval_df_reloaded["human_rating"] != "").any()):
                eval_df = eval_df_reloaded
                print("\n✓ Human ratings detected in CSV!")

        analyze_agreement(eval_df)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")

    if not skip_human_eval:
        print(f"\nResults saved to: {output_csv}")
        print("\nNext steps:")
        print("  - Fill in human ratings in the CSV file")
        print("  - Re-run this script to see complete agreement analysis")
    else:
        print("\nEvaluation completed with metric-based approach" +
              (" and LLM judge" if not skip_llm_judge else ""))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate transformer model")
    parser.add_argument("--model_dir", default="out/minilm_imdb_model", help="Model directory")
    parser.add_argument("--data_path", default="dataset/imdb-dataset.csv", help="Dataset path")
    parser.add_argument("--sample_size", type=int, default=1000, help="Total dataset size")
    parser.add_argument("--test_subsample", type=int, default=100, help="Test subsample size")
    parser.add_argument("--output_csv", default="out/evaluation_results.csv", help="Output CSV")
    parser.add_argument("--skip_llm_judge", action="store_true", help="Skip LLM judge evaluation")
    parser.add_argument("--skip_human_eval", action="store_true", help="Skip human evaluation CSV generation")
    parser.add_argument("--openrouter_api_key", default=None, help="OpenRouter API key (or use OPENROUTER_API_KEY env var)")

    args = parser.parse_args()

    evaluate_transformer(
        model_dir=args.model_dir,
        data_path=args.data_path,
        sample_size=args.sample_size,
        test_subsample=args.test_subsample,
        output_csv=args.output_csv,
        skip_llm_judge=args.skip_llm_judge,
        skip_human_eval=args.skip_human_eval,
        openrouter_api_key=args.openrouter_api_key,
    )
