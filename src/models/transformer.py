# finetune minilm on imdb set

from src.data.dataloader import TextPreprocessor, SENTIMENT_TO_ID
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# ==================== Transformer Dataset ====================


class TransformerDataset:
    def __init__(self, input_ids, attention_mask, labels):
        # Store as lists for dynamic padding (much faster than padding all to max_length)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

        lengths = [len(x) for x in input_ids]
        print(
            f"Dataset: {len(input_ids)} samples, avg_len={sum(lengths)/len(lengths):.1f}, max_len={max(lengths)}"
        )

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

    def __len__(self):
        return len(self.labels)


def finetune_minilm(data_path="dataset/imdb-dataset.csv", sample_size=1000, output_dir="out/minilm_imdb_model", tokenization="unigram", epochs=3, batch_size=32):
    """Finetune miniLM on IMDB dataset for sentiment classification."""

    # Detect device
    device = get_device()
    print(f"Using device: {device}")

    # Model configurations
    models = {
        "wordpiece": "microsoft/MiniLM-L12-H384-uncased",
        "bpe": "distilroberta-base", # 82 M not ideal for laptop
        "unigram": "albert-base-v2"
    }

    tokenizations = ["wordpiece", "bpe", "unigram"] if tokenization == "all" else [tokenization]
    outputs = {}

    for tok_type in tokenizations:
        model_name = models[tok_type]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # Enable gradient checkpointing for memory efficiency (allows larger batches)
        # Only enable if model supports it
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        except (ValueError, AttributeError):
            print("Gradient checkpointing not supported for this model")

        # Move model to device
        model = model.to(device)
        if hasattr(torch, 'compile') and device == "cuda":
            # torch.compile works best on CUDA
            model = torch.compile(model)


        preprocessor = TextPreprocessor(data_path, sample_size=sample_size, tokenizer_type=tok_type, tokenizer=tokenizer)
        df = preprocessor.load_data(remove_stopwords=False, max_length=320)
        train_df, val_df, test_df = preprocessor.get_splits()

        # Create datasets with already encoded data
        train_dataset = TransformerDataset(
            train_df["_input_ids"].tolist(),
            train_df["_attention_mask"].tolist(),
            train_df["sentiment"].map(SENTIMENT_TO_ID).tolist()
        )
        val_dataset = TransformerDataset(
            val_df["_input_ids"].tolist(),
            val_df["_attention_mask"].tolist(),
            val_df["sentiment"].map(SENTIMENT_TO_ID).tolist()
        )

        # Training arguments - optimize for device
        # bf16 is better than fp16 on modern hardware (Apple Silicon, A100, H100, etc.)
        # Apple Silicon (M1/M2/M3) has native bf16 support
        use_fp16 = False  # Disable fp16 in favor of bf16
        if device == "cuda":
            use_bf16 = torch.cuda.is_bf16_supported()
        elif device == "mps":
            # MPS backend supports bf16 on PyTorch 2.0+
            use_bf16 = True
        else:
            use_bf16 = False

        training_args = TrainingArguments(
            output_dir=(
                f"{output_dir}_{tok_type}" if tokenization == "all" else output_dir
            ),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,  # Increased from 16 (dynamic padding allows larger batches)
            per_device_eval_batch_size=64,  # Even larger for eval (no gradients)
            fp16=use_fp16,  # Enable for CUDA with bf16 not available
            bf16=use_bf16,  # Enable bf16 on CUDA if supported (better than fp16)
            dataloader_pin_memory=device == "cuda",  # Pin memory only for CUDA
            dataloader_num_workers=0 if device == "mps" else 2,  # Disable for MPS to avoid fork issues
            gradient_accumulation_steps=2,  
            eval_strategy="no",  # Skip mid-training eval (evaluate manually at end)
            save_strategy="no",  # Skip checkpointing
        )

        # Data collator for dynamic padding (pads to max length in each batch)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,  # Enable dynamic padding
        )

        # Train
        trainer.train()

        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(f"{output_dir}_{tok_type}" if tokenization == "all" else output_dir)

        # Evaluate on test set
        test_dataset = TransformerDataset(
            test_df["_input_ids"].tolist(),
            test_df["_attention_mask"].tolist(),
            test_df["sentiment"].map(SENTIMENT_TO_ID).tolist()
        )
        test_metrics = trainer.evaluate(test_dataset)
        print(f"{tok_type.upper()} Test results: {test_metrics}")

        # Collect run artifacts keyed by tokenizer type
        outputs[tok_type] = {
            "trainer": trainer,
            "model": model,
            "tokenizer": tokenizer,
            "metrics": test_metrics,
        }

    # Return a dict for all runs (single entry when tokenization != "all")
    return outputs


if __name__ == "__main__":
    finetune_minilm()
