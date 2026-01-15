"""LLM-as-a-judge evaluation using OpenRouter API."""

import pandas as pd
from typing import List
import os
import requests
import json
import time

import google.generativeai as genai


def llm_judge_gemini(review: str, api_key: str = None, retry_count: int = 0) -> str:
    """
    Use Google Gemini API to judge sentiment.

    Args:
        review: The movie review text
        api_key: Google Gemini API key (can be set via GOOGLE_API_KEY env var)
        retry_count: Internal counter to prevent infinite retries

    Returns:
        Predicted sentiment ("positive" or "negative")
    """
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY", "")

    if not api_key:
        raise ValueError(
            "Google API key not found. Please set GOOGLE_API_KEY environment variable "
            "or pass api_key parameter. Get your API key at https://ai.google.dev/"
        )

    prompt = f"""You are a sentiment classifier. Read the movie review below and determine if it expresses a positive or negative sentiment.

Review: "{review}"

Respond with ONLY ONE WORD: either "positive" or "negative" (lowercase, no explanation).

Answer:"""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=100,
            )
        )
        
        answer = response.text.strip().lower()

        # Extract positive/negative from response
        if "positive" in answer:
            return "positive"
        elif "negative" in answer:
            return "negative"
        else:
            # Fallback: check first word
            first_word = answer.split()[0] if answer.split() else ""
            return "positive" if "pos" in first_word else "negative"

    except Exception as e:
        print(f"  Warning: Error calling Gemini API ({e}), defaulting to negative")
        return "negative"

def llm_judge_sentiment_openrouter(review: str, api_key: str = None, retry_count: int = 0) -> str:
    """
    Use OpenRouter API with a free model to judge sentiment.

    Args:
        review: The movie review text
        api_key: OpenRouter API key (can be set via OPENROUTER_API_KEY env var)
        retry_count: Internal counter to prevent infinite retries

    Returns:
        Predicted sentiment ("positive" or "negative")
    """
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        raise ValueError(
            "OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable "
            "or pass api_key parameter. Get your free API key at https://openrouter.ai/keys"
        )

    prompt = f"""You are a sentiment classifier. Read the movie review below and determine if it expresses a positive or negative sentiment.

Review: "{review}"

Respond with ONLY ONE WORD: either "positive" or "negative" (lowercase, no explanation).

Answer:"""

    try:
        # Use free Llama model: meta-llama/llama-3.2-3b-instruct:free
        # Alternative free models:
        #   - meta-llama/llama-3.3-70b-instruct:free (larger, slower)
        #   - meta-llama/llama-3.1-405b-instruct:free (huge, very slow)
        #   - nvidia/nemotron-nano-9b-v2:free (fast alternative)
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/anthropics/claude-code",  # Required by OpenRouter
                "X-Title": "IMDB Sentiment Analysis",  # Optional but helpful
            },
            json={
                "model": "meta-llama/llama-3.2-3b-instruct:free",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 10,
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip().lower()

            # Extract positive/negative from response
            if "positive" in answer:
                return "positive"
            elif "negative" in answer:
                return "negative"
            else:
                # Fallback: check first word
                first_word = answer.split()[0] if answer.split() else ""
                return "positive" if "pos" in first_word else "negative"
        elif response.status_code == 429:
            if retry_count < 2:  # Max 2 retries
                wait_time = 5 * (retry_count + 1)  # Exponential backoff: 5s, 10s
                print(f"  Warning: Rate limit hit (429), waiting {wait_time}s...")
                time.sleep(wait_time)
                return llm_judge_sentiment_openrouter(review, api_key, retry_count + 1)
            else:
                print(f"  Error: Rate limit exceeded after retries, defaulting to negative")
                return "negative"
        elif response.status_code == 402:
            print(f"  Warning: Payment required (402) - free tier may be exhausted")
            print(f"  Response: {response.text[:200]}")
            raise ValueError("OpenRouter free tier limit reached. Try a different model or wait.")
        else:
            print(f"  Warning: API error (status {response.status_code}), defaulting to negative")
            print(f"  Response: {response.text[:200]}")
            return "negative"

    except requests.exceptions.Timeout:
        print(f"  Warning: Timeout for review, defaulting to negative")
        return "negative"
    except Exception as e:
        print(f"  Warning: Error calling OpenRouter API ({e}), defaulting to negative")
        return "negative"


def evaluate_with_llm_judge(df: pd.DataFrame, api_key: str = None, model_type: str = "gemini") -> pd.DataFrame:
    """
    Evaluate all samples using LLM judge.

    Args:
        df: DataFrame with 'review' column
        api_key: API key (OpenRouter API key for OpenRouter, Google API key for Gemini)
        model_type: Which model to use - "openrouter" (Llama 3.2 3B) or "gemini" (Gemini 2.5 Flash)

    Returns:
        DataFrame with added 'llm_judge' column
    """
    if model_type not in ["openrouter", "gemini"]:
        raise ValueError(f"model_type must be 'openrouter' or 'gemini', got '{model_type}'")
    
    model_name = "Llama 3.2 3B via OpenRouter" if model_type == "openrouter" else "Gemini 2.5 Flash"
    
    print(f"\n{'='*50}")
    print(f"LLM JUDGE EVALUATION ({model_name})")
    print(f"{'='*50}")
    print(f"Evaluating {len(df)} samples...")
    print("This may take a few minutes...\n")

    llm_judgments = []
    
    # Select the appropriate judgment function
    judge_fn = llm_judge_sentiment_openrouter if model_type == "openrouter" else llm_judge_gemini

    for i, row in df.iterrows():
        print(f"Processing {i+1}/{len(df)}...", end=" ")
        judgment = judge_fn(row["review"], api_key=api_key)
        llm_judgments.append(judgment)
        print(f"✓ {judgment}")

        # Delay to avoid rate limiting
        time.sleep(2.0)

    df["llm_judge"] = llm_judgments
    return df


def analyze_agreement(df: pd.DataFrame):
    """
    Analyze agreement between gold labels, model predictions, human ratings, and LLM judge.

    Args:
        df: DataFrame with all evaluation columns
    """
    print(f"\n{'='*50}")
    print("AGREEMENT ANALYSIS")
    print(f"{'='*50}")

    # Check which columns are available
    has_human = df["human_rating"].notna().any() and (df["human_rating"] != "").any()

    # Model vs Gold
    model_gold_agree = (df["model_prediction"] == df["gold_label"]).sum()
    print(f"\nModel vs Gold Labels:")
    print(f"  Agreement: {model_gold_agree}/{len(df)} ({model_gold_agree/len(df)*100:.1f}%)")

    # LLM vs Gold
    if "llm_judge" in df.columns:
        llm_gold_agree = (df["llm_judge"] == df["gold_label"]).sum()
        print(f"\nLLM Judge vs Gold Labels:")
        print(f"  Agreement: {llm_gold_agree}/{len(df)} ({llm_gold_agree/len(df)*100:.1f}%)")

        # Model vs LLM
        model_llm_agree = (df["model_prediction"] == df["llm_judge"]).sum()
        print(f"\nModel vs LLM Judge:")
        print(f"  Agreement: {model_llm_agree}/{len(df)} ({model_llm_agree/len(df)*100:.1f}%)")

    # Human vs others (if available)
    if has_human:
        human_gold_agree = (df["human_rating"] == df["gold_label"]).sum()
        print(f"\nHuman Rating vs Gold Labels:")
        print(f"  Agreement: {human_gold_agree}/{len(df)} ({human_gold_agree/len(df)*100:.1f}%)")

        human_model_agree = (df["human_rating"] == df["model_prediction"]).sum()
        print(f"\nHuman Rating vs Model:")
        print(f"  Agreement: {human_model_agree}/{len(df)} ({human_model_agree/len(df)*100:.1f}%)")

        if "llm_judge" in df.columns:
            human_llm_agree = (df["human_rating"] == df["llm_judge"]).sum()
            print(f"\nHuman Rating vs LLM Judge:")
            print(f"  Agreement: {human_llm_agree}/{len(df)} ({human_llm_agree/len(df)*100:.1f}%)")

    # Show disagreement cases
    print(f"\n{'='*50}")
    print("DISAGREEMENT EXAMPLES")
    print(f"{'='*50}")

    if "llm_judge" in df.columns:
        # Cases where model and LLM disagree with gold
        disagree_both = df[
            (df["model_prediction"] != df["gold_label"]) &
            (df["llm_judge"] != df["gold_label"])
        ]

        if len(disagree_both) > 0:
            print(f"\nBoth Model and LLM disagree with gold ({len(disagree_both)} cases):")
            for i, row in disagree_both.head(3).iterrows():
                print(f"\n  Review: {row['review'][:100]}...")
                print(f"  Gold: {row['gold_label']}, Model: {row['model_prediction']}, LLM: {row['llm_judge']}")

        # Cases where model and LLM agree but gold differs
        agree_wrong = df[
            (df["model_prediction"] == df["llm_judge"]) &
            (df["model_prediction"] != df["gold_label"])
        ]

        if len(agree_wrong) > 0:
            print(f"\nModel and LLM agree but differ from gold ({len(agree_wrong)} cases):")
            for i, row in agree_wrong.head(3).iterrows():
                print(f"\n  Review: {row['review'][:100]}...")
                print(f"  Gold: {row['gold_label']}, Model+LLM: {row['model_prediction']}")
