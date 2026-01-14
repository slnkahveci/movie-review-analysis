import pandas as pd
from transformers import AutoTokenizer

DATA='dataset/imdb-dataset.csv'
sample_size=1000
thresholds=[256,320,384,448,512]

models = {
    'wordpiece': 'microsoft/MiniLM-L12-H384-uncased',
    'bpe': 'distilroberta-base',
    'unigram': 'albert-base-v2'
}

if __name__ == "__main__":
    df = pd.read_csv(DATA)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    text = df["review"].tolist()

    for name, model in models.items():
        tok = AutoTokenizer.from_pretrained(model)
        enc = tok(text, truncation=False, padding=False)
        lens = [len(ids) for ids in enc["input_ids"]]
        print("\n", name)
        for t in thresholds:
            pct = sum(l <= t for l in lens) / len(lens) * 100
            print(f"  <= {t}: {pct:5.1f}%")


# Load and preprocess data with tokenizer (set by measured length distribution)
# Token length stats on 1k samples (no truncation):
#   wordpiece median 232, p95 837, p99 1192
#   bpe       median 225, p95 804, p99 1196
#   unigram   median 240, p95 861, p99 1227
# Coverage at various caps (percent kept):
#   256: ~55-60%, 320: ~67-70%, 384: ~74-77%, 448: ~80-82%, 512: ~84-87%
# Chosen cap = 320 as a balance between truncation and throughput (model max is 512)
