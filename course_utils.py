# course_utils.py

from __future__ import annotations

import getpass
import gradio as gr
import numpy as np
from openai import OpenAI
import os
from pathlib import Path
import random
import sys
import subprocess
import time


# If you already have a seed function, reuse it.
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

def _install(deps):
    for pkg in deps:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                check=True,
            )

# Install deps quietly if missing
def install_core_deps():
    _install(["openai", "gradio", "mermaid-python"])


def init_openai():
    # Ask for API key if not present
    if not os.environ.get("OPENAI_API_KEY"):
        print("Enter your OpenAI API key. It will only live in this Colab runtime.")
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
        print("✅ API key set.")
    else:
        print("✅ OPENAI_API_KEY already set.")


def show_mermaid(graph_str):
    _install(['mermaid-python'])
    from mermaid import Mermaid
    display(Mermaid(graph_str))

def lab1_setup() -> None:
    """
    Colab-specific bootstrap for Lab 1.

    - Installs required packages if needed.
    - Seeds randomness.
    - Prompts for OPENAI_API_KEY if not set.
    """
    install_core_deps()
    seed_everything(42)
    init_openai()
    print("✅ lab1_setup: environment ready.")


EXAMPLE_LAB1_PROMPTS = [
    "Explain this course to a 10-year-old.",
    "Write a short poem about Tulane.",
    "What is one cool application of AI?",
]

# ---------- Lab 1 LLM helpers ----------

def _lab1_client() -> OpenAI:
    """Return an OpenAI client. Assumes OPENAI_API_KEY is set."""
    return OpenAI()


def lab1_generate_reply(
    user_prompt: str,
    system_prompt: str,
    temperature: float = 0.7,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Simple one-shot chat completion for Lab 1.
    """
    client = _lab1_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def lab1_build_demo(system_prompt: str, default_temperature: float = 0.7):
    """
    Build a minimal Gradio Interface for Lab 1.
    """
    import gradio as gr

    def _fn(user_prompt: str, temperature: float):
        if not user_prompt.strip():
            return "Please enter a non-empty prompt."
        return lab1_generate_reply(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    demo = gr.Interface(
        fn=_fn,
        inputs=[
            gr.Textbox(label="Your prompt"),
            gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=default_temperature,
                step=0.1,
                label="Temperature",
            ),
        ],
        outputs=gr.Textbox(label="Model response"),
        title="Lab 1: Hello, API",
        description="Your first LLM-powered web app.",
    )
    return demo



# ---------- Setup ----------
def lab2_setup():
    """
    Setup for Lab 2: Temperature & Diversity.
    - Verifies OpenAI client connectivity
    - Loads scientific utilities (numpy, matplotlib)
    - Provides a safe helper for model calls
    - Prints confirmation banner
    """
    import sys, os, math, numpy as np
    import matplotlib.pyplot as plt
    install_core_deps()
    seed_everything(42)
    init_openai()

    if '/content/main' not in sys.path:
        sys.path.append('/content/main')

    # Define a small helper for consistent API experiments
    def run_prompt(prompt, temperature=1.0, model="gpt-4o-mini"):
        """
        Simple helper to send a prompt and return text + token stats.
        """
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.choices[0].message.content
            tokens = resp.usage.total_tokens
            print(f"[T={temperature}] Tokens: {tokens}")
            return text
        except Exception as e:
            print("API call failed:", e)
            return None

    globals()["run_prompt"] = run_prompt  # expose helper globally
    print("✅ lab2_setup complete — scientific libraries ready, helper function loaded.")


# ---------- Core Experiment ----------
def lab2_generate_samples(prompt: str, temperatures=[0.3, 1.0, 2.0], n_per_temp=5, model="gpt-4o-mini"):
    """
    Generate multiple completions for a given prompt across temperature settings.
    Returns a list of dicts with {temperature, output}.
    """
    client = OpenAI()
    results = []
    for T in temperatures:
        for i in range(n_per_temp):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=T,
                    messages=[{"role": "user", "content": prompt}],
                )
                output = resp.choices[0].message.content.strip()
                results.append({
                    "temperature": T,
                    "output": output
                })
                time.sleep(0.2)
            except Exception as e:
                print(f"⚠️ API call failed at T={T}, sample {i+1}: {e}")
    return results


# ---------- Diversity Measurement ----------
def lab2_measure_diversity(results):
    """
    Compute a simple diversity index per temperature group.
    Metric: unique outputs / total outputs (per temperature).
    """
    diversity_scores = {}
    grouped = {}
    for r in results:
        grouped.setdefault(r["temperature"], []).append(r["output"])

    for T, outputs in grouped.items():
        unique_count = len(set(outputs))
        total_count = len(outputs)
        diversity = round(unique_count / total_count, 3) if total_count else 0
        diversity_scores[T] = diversity
    return diversity_scores


# ---------- Gradio Demo ----------
def lab2_build_demo(default_prompt="Describe a sunrise.", default_temperature=1.0):
    """
    Build a simple Gradio app for interactive temperature exploration.
    """
    def generate(prompt, temperature):
        try:
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"⚠️ Error: {e}"

    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(value=default_prompt, label="Prompt"),
            gr.Slider(0.0, 2.0, value=default_temperature, label="Temperature")
        ],
        outputs="text",
        title="Lab 2 — Temperature Explorer",
        description="Experiment with LLM temperature: low = consistent, high = creative.",
    )
    return demo


# ---------- LAB 3 Setup ----------
def lab3_setup():
    """
    Setup for Lab 3
    """
    import sys, os, math, numpy as np
    import matplotlib.pyplot as plt
    install_core_deps()
    _install(["dspy"])
    seed_everything(42)
    init_openai()
    import dspy
    if '/content/main' not in sys.path:
        sys.path.append('/content/main')

    print("✅ lab3_setup complete — scientific libraries ready, helper function loaded.")


# ----------------------------------------------------------
# Lab 3: Safe Run Helper
# ----------------------------------------------------------
def safe_run_demo(text: str):
    """
    Demonstrate error handling and logging for pipeline steps.
    Simulates three possible outcomes:
    - Valid input: returns success dictionary.
    - Empty input: raises ValueError.
    - 'simulate_error': raises RuntimeError.
    """
    try:
        if not text:
            raise ValueError("Empty input text")
        if "simulate_error" in text.lower():
            raise RuntimeError("Simulated model crash")

        return {"status": "ok", "message": "All pipeline steps succeeded"}

    except Exception as e:
        print(f"⚠️ Error in pipeline step: {e}")
        return None


# ----------------------------------------------------------
# Lab 3: Sentiment Pipeline Builder
# ----------------------------------------------------------
def build_sentiment_pipeline_demo(verbose: bool = False):
    """
    Build a simple three-step DSPY pipeline:
      1. Extract sentences about Tulane University
      2. Annotate each sentence with sentiment (pos/neg/neutral)
      3. Summarize overall sentiment

    If verbose=True, prints intermediate steps for transparency.
    """
    import dspy
    extract = dspy.Predict("text -> sentences_about_tulane: list[str]")
    annotate = dspy.Predict("sentence -> sentiment: str")
    summarize = dspy.Predict("sentiments: list[str] -> summary: str")

    def pipeline(text: str):
        try:
            # Step 1: Extract sentences mentioning Tulane
            extraction_result = extract(text=text)
            sentences = getattr(extraction_result, "sentences_about_tulane", [])
            if verbose:
                print("🟢 Extracted Sentences:", sentences)

            # Step 2: Annotate sentiment
            labels = []
            for s in sentences:
                sentiment_result = annotate(sentence=s)
                sentiment = getattr(sentiment_result, "sentiment", "unknown")
                labels.append(sentiment)
                if verbose:
                    print(f"🟣 Sentiment for '{s[:50]}...': {sentiment}")

            # Step 3: Summarize
            summary_result = summarize(sentiments=labels)
            summary = getattr(summary_result, "summary", "")
            if verbose:
                print("🧩 Final Summary:", summary)

            return summary

        except Exception as e:
            print(f"⚠️ Pipeline failed: {e}")
            return "⚠️ Pipeline encountered an error."

    return pipeline


# ----------------------------------------------------------
# Lab 3: Single-Prompt Comparison Function
# ----------------------------------------------------------
def single_prompt_sentiment_summary(article: str) -> str:
    """
    Single-prompt approach for comparison:
    Asks the LLM to read the entire article and summarize the sentiment
    about Tulane University in one go.
    """
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful analyst of university sentiment.",
                },
                {
                    "role": "user",
                    "content": f"Analyze this article and summarize the overall sentiment about Tulane University:\n{article}",
                },
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️ Single-prompt call failed: {e}")
        return "⚠️ Error running single-prompt model."


# ----------------------------------------------------------
# Lab 3: Optional: Simple Gradio Demo Builder
# ----------------------------------------------------------
def lab3_build_demo():
    """
    Optional Gradio interface for interactive exploration.
    Lets students input an article and see the pipeline outputs.
    """

    pipeline = build_sentiment_pipeline_demo(verbose=True)

    def run_pipeline(article):
        return pipeline(article)

    with gr.Blocks() as demo:
        gr.Markdown("### 🧠 DSPY Sentiment Pipeline Explorer")
        inp = gr.Textbox(label="Article Text", placeholder="Paste text mentioning Tulane University")
        out = gr.Textbox(label="Pipeline Output (Summary)")
        btn = gr.Button("Run Pipeline")
        btn.click(run_pipeline, inputs=inp, outputs=out)
    return demo