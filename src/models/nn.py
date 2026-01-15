# nn consists of:
# embedding layer
# LSTM or GRU layer
# softmax output layer
# input: sequences of token ids
# output: probabilities over vocabulary for next token

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import tqdm
import functools
import pandas as pd
from collections import Counter
from typing import Optional

from src.data.preprocessing import TextPreprocessor, SENTIMENT_TO_ID

# Special token IDs
padding_idx = 0
unk_token_id = 1
start_token_id = 2
end_token_id = 3


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# ==================== RNN Dataset and DataModule ====================


class RNNDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for RNN language modeling.
    Stores raw tokens and labels; token-to-ID conversion happens in collate_fn.
    """

    def __init__(self, tokens_list: list[list[str]], labels: list[int]):
        self.tokens_list = tokens_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.tokens_list)

    def __getitem__(self, idx: int) -> dict:
        return {
            "tokens": self.tokens_list[idx],
            "label": self.labels[idx],
        }


def rnn_collate_fn(
    batch,
    vocab: dict[str, int],
    padding_idx: int = 0,
    max_seq_length: Optional[int] = None,
    start_token: Optional[str] = None,
    end_token: Optional[str] = None,
):
    """
    Collate function for RNN that converts tokens to IDs and creates input/target pairs.
    """
    unk_id = vocab.get("<UNK>", 1)

    # Convert tokens to IDs with optional start/end tokens
    token_ids_list = []
    for item in batch:
        tokens = item["tokens"]
        if start_token:
            tokens = [start_token] + tokens
        if end_token:
            tokens = tokens + [end_token]
        token_ids = [vocab.get(t, unk_id) for t in tokens]
        token_ids_list.append(token_ids)

    labels = [item["label"] for item in batch]

    # Truncate BUT preserve end token
    if max_seq_length is not None:
        truncated = []
        for seq in token_ids_list:
            if len(seq) > max_seq_length:
                truncated.append(seq[: max_seq_length - 1] + [seq[-1]])
            else:
                truncated.append(seq)
        token_ids_list = truncated

    # Build input/target pairs
    inputs = []
    targets = []
    for seq in token_ids_list:
        if len(seq) > 1:
            inputs.append(seq[:-1])
            targets.append(seq[1:])
        else:
            inputs.append([padding_idx])
            targets.append([padding_idx])

    # Pad all sequences
    max_len = max(len(s) for s in inputs)
    input_ids = torch.full((len(inputs), max_len), padding_idx, dtype=torch.long)
    target_ids = torch.full((len(targets), max_len), padding_idx, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        input_ids[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        target_ids[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "label": torch.tensor(labels, dtype=torch.long),
    }


class RNNDataModule:
    """
    Creates PyTorch DataLoader instances for RNN models.
    Handles vocabulary building and token-to-ID conversion.

    Usage:
        preprocessor = TextPreprocessor("dataset/imdb-dataset.csv")
        df = preprocessor.load_data(remove_stopwords=True)
        train_df, val_df, test_df = preprocessor.get_splits()

        data_module = RNNDataModule()
        data_module.build_vocab(train_df["_words"].tolist())
        train_loader = data_module.get_dataloader(train_df, batch_size=32, shuffle=True)
    """

    def __init__(self):
        self.vocab: Optional[dict[str, int]] = None
        self.padding_idx = 0

    def build_vocab(self, tokens_list: list[list[str]], min_freq: int = 10):
        """
        Build vocabulary from token sequences.

        Args:
            tokens_list: List of token sequences
            min_freq: Minimum frequency for a token to be included in vocabulary
        """
        token_counter = Counter()
        for tokens in tokens_list:
            token_counter.update(tokens)

        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<s>": 2, "</s>": 3}

        for token, freq in token_counter.items():
            if token not in self.vocab and freq >= min_freq:
                self.vocab[token] = len(self.vocab)

        print(f"Built vocabulary with {len(self.vocab)} tokens")

    def decode_sequence(self, token_ids: list[int]) -> str:
        """Convert token IDs back to string."""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        id_to_token = {idx: token for token, idx in self.vocab.items()}  
        tokens = [id_to_token.get(tid, "<UNK>") for tid in token_ids]
        return " ".join(tokens)

    def get_dataloader(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        shuffle: bool = False,
        start_token: Optional[str] = "<s>",
        end_token: Optional[str] = "</s>",
        max_seq_length: Optional[int] = None,
        num_workers: int = 0,
        device: Optional[str] = None,
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader from a DataFrame.

        Args:
            df: DataFrame with "_words" column from TextPreprocessor
            batch_size: batch size
            shuffle: whether to shuffle
            start_token: token to prepend (default: "<s>")
            end_token: token to append (default: "</s>")
            max_seq_length: max sequence length (truncates longer sequences)
            num_workers: number of worker processes
            device: device string for pin_memory optimization

        Returns:
            PyTorch DataLoader
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        if "_words" not in df.columns:
            raise ValueError(
                "DataFrame must contain '_words' column. "
                "Use TextPreprocessor with tokenizer_type='word' (default)."
            )

        dataset = RNNDataset(
            tokens_list=df["_words"].tolist(),
            labels=df["sentiment"].map(SENTIMENT_TO_ID).tolist(),
        )

        collate_fn = functools.partial(
            rnn_collate_fn,
            vocab=self.vocab,
            padding_idx=self.padding_idx,
            max_seq_length=max_seq_length,
            start_token=start_token,
            end_token=end_token,
        )

        pin_memory = device == "cuda" if device else False
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


# ==================== RNN Model ====================


class RNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, rnn_type: str = 'LSTM', device=None):
        super(RNN, self).__init__()
        self.device = device if device is not None else get_device()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("rnn_type must be either 'LSTM' or 'GRU'")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.to(self.device)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        rnn_out = self.dropout(rnn_out)
        logits = self.fc(rnn_out)
        return logits

    def train_next_token(self, input_sequence, target_sequence, criterion, optimizer, max_grad_norm=1.0):
        self.train()
        # The forward method returns logits.
        # Using nn.NLLLoss here would produce incorrect losses.
        if isinstance(criterion, nn.NLLLoss):
            raise ValueError(
                "RNN.forward returns raw logits, not log-probabilities. "
                "Use nn.CrossEntropyLoss (or another loss that expects raw logits), "
                "or modify RNN.forward to return log-probabilities before using nn.NLLLoss."
            )
        optimizer.zero_grad()

        output = self.forward(input_sequence)
        loss = criterion(output.view(-1, output.size(-1)), target_sequence.view(-1))
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        optimizer.step()

        loss_value = loss.item()

        return loss_value


    def train_loop(
        self,
        data_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs: int = 10,
        max_grad_norm: float = 1.0,
        accumulation_steps: int = 8,
    ):  # Effective batch = 8 * 8 = 64
        self.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            optimizer.zero_grad()

            for i, batch in enumerate(
                tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            ):
                inputs = batch["input_ids"].to(self.device)
                targets = batch["target_ids"].to(self.device)

                output = self.forward(inputs)
                loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
                loss = loss / accumulation_steps  # Scale loss
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)

    def generate(self, start_token: int, max_length: int = 20, temperature: float = 0.8):
        self.eval()
        generated_sequence = [start_token]
        input_token = torch.tensor([[start_token]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            for _ in range(max_length - 1):
                logits = self.forward(input_token)[:, -1, :]

                # Temperature sampling instead of argmax
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                generated_sequence.append(next_token)
                if next_token == end_token_id:  # assuming end_token_id is defined globally
                    break
                input_token = torch.tensor(
                    [[next_token]], dtype=torch.long, device=self.device
                )

        return generated_sequence

    def compute_perplexity(self, data_loader):
        self.eval()
        total_loss = 0.0
        total_tokens = 0

        criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)  # forward returns raw logits

        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader, desc="Computing Perplexity"):
                inputs = batch["input_ids"].to(self.device)
                targets = batch["target_ids"].to(self.device)
                outputs = self.forward(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss))
        return perplexity.item()
    
    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device, weight_only=True))
        self.to(self.device)


if __name__ == "__main__":    # Example usage
    device = get_device()
    print(f"Using device: {device}")
    tokenization = "word"  # or "bpe", "wordpiece"
    preprocessor = TextPreprocessor("dataset/imdb-dataset.csv", sample_size=1000, tokenizer_type=tokenization)
    df = preprocessor.load_data(remove_stopwords=False)

    data_module = RNNDataModule()
    token_column = df["_words"] if tokenization == "word" else df[f"_tokens"]
    data_module.build_vocab(token_column, min_freq=5)

    # Use smaller batch size and limit sequence length to prevent memory issues
    train_loader = data_module.get_dataloader(
        df=df,
        batch_size=8,
        shuffle=True,
        max_seq_length=128,
        start_token="<s>",
        end_token="</s>",
        device=device,
    )

    vocab_size = len(data_module.vocab)
    model = RNN(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, rnn_type='LSTM', device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    model.train_loop(train_loader, criterion, optimizer, scheduler=scheduler, num_epochs=10, max_grad_norm=1.0)
    model.save_model("out/rnn_imdb_model.pth")
    perplexity = model.compute_perplexity(train_loader)
    print(f"Perplexity: {perplexity:.4f}")

    # Get start token ID from vocabulary
    start_token_id = data_module.vocab.get("<s>", 1)
    generated_sequence = model.generate(start_token=start_token_id, max_length=50, temperature=0.8)
    generated_string = data_module.decode_sequence(generated_sequence)
    print("Generated sequence:", generated_string)
