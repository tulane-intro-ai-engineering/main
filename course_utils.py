# course_utils.py

from __future__ import annotations
import random
from pathlib import Path

import numpy as np

# If you already have a seed function, reuse it.
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def colab_bootstrap_lab1() -> None:
    """
    Colab-specific bootstrap for Lab 1.

    - Installs required packages if needed.
    - Seeds randomness.
    - Prompts for OPENAI_API_KEY if not set.
    """
    import os
    import sys
    import getpass
    import subprocess

    # Install deps quietly if missing
    for pkg in ["openai", "gradio"]:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                check=True,
            )

    seed_everything(42)

    # Ask for API key if not present
    if not os.environ.get("OPENAI_API_KEY"):
        print("Enter your OpenAI API key. It will only live in this Colab runtime.")
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
        print("✅ API key set.")
    else:
        print("✅ OPENAI_API_KEY already set.")

    print("✅ colab_bootstrap_lab1: environment ready.")


EXAMPLE_LAB1_PROMPTS = [
    "Explain this course to a 10-year-old.",
    "Write a short poem about Tulane.",
    "What is one cool application of AI?",
]


# ---------- Lab 1 LLM helpers ----------

from openai import OpenAI

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


def LAB2_colab_bootstrap():
    """
    Setup for Lab 2: Temperature & Diversity.
    - Verifies OpenAI client connectivity
    - Loads scientific utilities (numpy, matplotlib)
    - Provides a safe helper for model calls
    - Prints confirmation banner
    """
    import sys, os, math, numpy as np
    import matplotlib.pyplot as plt

    if '/content/main' not in sys.path:
        sys.path.append('/content/main')

    try:
        from openai import OpenAI
        client = OpenAI()
        print("✅ OpenAI client initialized successfully!")
    except Exception as e:
        print("⚠️ Warning: Could not initialize OpenAI client. Check your environment.")
        print(e)
        return

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
    print("✅ LAB2_colab_bootstrap complete — scientific libraries ready, helper function loaded.")
