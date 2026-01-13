# nn consists of:
# embedding layer
# LSTM or GRU layer
# softmax output layer
# input: sequences of token ids
# output: probabilities over vocabulary for next token
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import gc

from src.data.dataloader import TextPreprocessor, IMDBDataModule, IMDBDataStats
padding_idx = 0  # assuming 0 is the padding index

# TODO: check IMDBDataset and token id mappings


class RNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, rnn_type: str = 'LSTM', device=None):
        super(RNN, self).__init__()
        self.device = "mps"
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

    def train_loop(self, data_loader, criterion, optimizer, num_epochs: int = 10, max_grad_norm: float = 1.0):
        self.train()
        print(f"Training on device: {self.device}")

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs = batch["input_ids"]
                targets = batch["target_ids"]

                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                loss = self.train_next_token(inputs, targets, criterion, optimizer, max_grad_norm)
                total_loss += loss
                num_batches += 1

                # Periodic memory cleanup
                if num_batches % 100 == 0:
                    gc.collect()

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Memory cleanup after each epoch
            gc.collect()

    def generate(self, start_token: int, max_length: int = 20, temperature: float = 0.8):
        self.eval()
        generated_sequence = [start_token]
        input_token = torch.tensor([[start_token]], dtype=torch.long, device=self.device)
        hidden = None

        with torch.no_grad():
            for _ in range(max_length - 1):
                embedded = self.embedding(input_token)
                rnn_out, hidden = self.rnn(embedded, hidden)
                logits = self.fc(rnn_out[:, -1, :])

                # Temperature sampling instead of argmax
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                generated_sequence.append(next_token)
                if next_token == padding_idx:
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


if __name__ == "__main__":    # Example usage
    device = "mps" 
    print(f"Using device: {device}")

    preprocessor = TextPreprocessor("dataset/imdb-dataset.csv", sample_size=1000)
    df = preprocessor.load_data(remove_stopwords=False)

    data_module = IMDBDataModule()
    data_module.build_vocab(df["_tokens"].tolist())

    # Use smaller batch size and limit sequence length to prevent memory issues
    train_loader = data_module.get_torch_dataloader(
        df=df,
        batch_size=16,
        shuffle=True,
        max_seq_length=256,
        start_token="<s>",
        end_token="</s>",  
    )

    vocab_size = len(data_module.vocab)
    model = RNN(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, rnn_type='LSTM', device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train_loop(train_loader, criterion, optimizer, num_epochs=5, max_grad_norm=1.0)
    perplexity = model.compute_perplexity(train_loader)
    print(f"Perplexity: {perplexity:.4f}")

    # Get start token ID from vocabulary
    start_token_id = data_module.vocab.get("<s>", 1)
    generated_sequence = model.generate(start_token=start_token_id, max_length=15)
    generated_string = data_module.decode_sequence(generated_sequence)
    print("Generated sequence:", generated_string)
