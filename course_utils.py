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



import matplotlib.pyplot as plt
import networkx as nx

def draw_llm_workflow_networkx(highlight=None, figsize=(12, 8)):
    """
    Draw the LLM workflow using NetworkX with:
    - dynamic node sizing based on label length
    - optional highlighting of selected nodes

    Parameters
    ----------
    highlight : list[str] or None
        List of node labels to highlight. All others will be faded.
    figsize : tuple
        Figure size in inches
    """
    if highlight is None:
        highlight = []

    # Graph definition
    G = nx.DiGraph()

    nodes = [
        "Users",
        "Input Handling",
        "Prompt / Control",
        "Tools / Functions?",
        "Retrieval (RAG)?",
        "LLM (Model)",
        "Output Processing",
        "Final Output",
        "Logging & Monitoring",
    ]

    edges = [
        ("Users", "Input Handling"),
        ("Input Handling", "Prompt / Control"),
        ("Prompt / Control", "Tools / Functions?"),
        ("Prompt / Control", "Retrieval (RAG)?"),
        ("Tools / Functions?", "LLM (Model)"),
        ("Retrieval (RAG)?", "LLM (Model)"),
        ("LLM (Model)", "Output Processing"),
        ("Output Processing", "Final Output"),
        ("Final Output", "Logging & Monitoring"),
    ]

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    SPACING_X = 3.2
    SPACING_Y = .1
    # Manual layout (top-down)
    pos = {
        "Users": (0 * SPACING_X, 0),
        "Input Handling": (1 * SPACING_X, 0),
        "Prompt / Control": (2 * SPACING_X, 0),
        "Tools / Functions?": (3 * SPACING_X, 1 * SPACING_Y),
        "Retrieval (RAG)?": (3 * SPACING_X, -1 * SPACING_Y),
        "LLM (Model)": (4.5 * SPACING_X, 0),
        "Output Processing": (5.5 * SPACING_X, 0),
        "Final Output": (6.5 * SPACING_X, 0),
        "Logging & Monitoring": (7.5 * SPACING_X, 0),
    }


    node_sizes = []
    node_colors = []
    node_alphas = []

    for n in G.nodes():
        # Dynamic size based on label length
        size = 1200 + len(n) * 260
        node_sizes.append(size)

        # Color by type
        if "?" in n:
            color = "#f4a261"
        elif n == "Users":
            color = "#bdbdbd"
        else:
            color = "#4db6e2"
        node_colors.append(color)

        # Highlight logic
        if highlight and n not in highlight:
            node_alphas.append(0.25)
        else:
            node_alphas.append(1.0)

    plt.figure(figsize=figsize)

    # Draw nodes individually to support per-node alpha
    for i, node in enumerate(G.nodes()):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_size=node_sizes[i],
            node_color=node_colors[i],
            alpha=node_alphas[i],
        )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=22,
        edge_color="black",
        alpha=0.8,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_weight="bold",
    )

    plt.title("LLM System", fontsize=15, weight="bold")
    plt.axis("off")
    plt.show()

