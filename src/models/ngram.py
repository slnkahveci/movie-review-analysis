"""
Task 3.1 – N-gram Language Models
For this task you should implement a bigram and a trigram language model
Requirements:

Compute smoothed probabilities (e.g., Laplace smoothing)
Generate short text sequences from each model
Compute and compare perplexity across n-gram sizes

"""

from collections import defaultdict
import random
import pandas as pd
import math
from tqdm import tqdm
from time import time

from src.data.preprocessing import TextPreprocessor


class NGramModel:
    def __init__(self, n: int, laplace_smoothing: bool = True):
        self.n = n
        self.laplace_smoothing = laplace_smoothing
        self.ngrams = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        self.vocabulary = set()

    def train(self, corpus: pd.DataFrame): # train without removing stopwords
        for sentence in tqdm(corpus.values, desc="Training", unit="sentence"): # sentence is a list of tokens
            sentence_list = [token for token in sentence if token is not None]
            tokens = ['<s>'] * (self.n - 1) + sentence_list + ['</s>'] # to ensure first n-1 words have a valid context
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1]) # list to tuple for dict key
                word = tokens[i + self.n - 1]
                self.ngrams[context][word] += 1
                self.context_counts[context] += 1
                self.vocabulary.add(word) # includes <s> and </s>

    def get_probability(self, context: tuple, word: str) -> float:
        if self.laplace_smoothing:
            vocab_size = len(self.vocabulary) 
            word_count = self.ngrams[context][word] + 1
            context_count = self.context_counts[context] + vocab_size # effectively adding one to every word count
            return word_count / context_count
        else:
            if self.context_counts[context] == 0:
                return 0.0
            return self.ngrams[context][word] / self.context_counts[context]

    def generate_sentence(self, max_length: int = 15) -> str:
        context = ('<s>',) * (self.n - 1) # padding for valid starting context
        sentence = []
        for _ in range(max_length):
            words = list(self.ngrams[context].keys())
            probabilities = [self.get_probability(context, w) for w in words]
            total_prob = sum(probabilities)
            probabilities = [p / total_prob for p in probabilities]
            next_word = random.choices(words, weights=probabilities)[0]
            if next_word == '</s>':
                break
            sentence.append(next_word)
            context = context[1:] + (next_word,)
        return ' '.join(sentence)

    def compute_perplexity(self, corpus: pd.DataFrame) -> float:
        N = 0
        log_prob_sum = 0.0
        for sentence in tqdm(corpus.values, desc="Computing Perplexity", unit="sentence"):
            sentence_list = [token for token in sentence if token is not None]
            tokens = ['<s>'] * (self.n - 1) + sentence_list + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]
                prob = self.get_probability(context, word)
                if prob > 0:
                    log_prob_sum += math.log(prob)
                else:
                    log_prob_sum += float('-inf')
                N += 1
        perplexity = math.exp(-log_prob_sum / N) if N > 0 else float('inf')
        return perplexity

# Example usage:
if __name__ == "__main__":
    corpus = TextPreprocessor("dataset/imdb-dataset.csv", sample_size=50000).load_data(remove_stopwords=False)["_tokens"]   
    bigram_model = NGramModel(n=2)
    time_start = time()
    bigram_model.train(corpus)
    time_end = time()
    print(f"Training time for bigram model: {time_end - time_start:.2f} seconds")
    print(bigram_model.generate_sentence())
    print(bigram_model.compute_perplexity(corpus))


    trigram_model = NGramModel(n=3)
    time_start = time()
    trigram_model.train(corpus)
    time_end = time()
    print(f"Training time for trigram model: {time_end - time_start:.2f} seconds")
    print(trigram_model.generate_sentence())
    print(trigram_model.compute_perplexity(corpus))