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

def lab3_build_demo(verbose: bool = False):
    """Gradio app builder for the DSPY sentiment pipeline."""
    import gradio as gr
    pipeline = build_sentiment_pipeline_demo(verbose=verbose)

    def run_pipeline(article):
        return pipeline(article)

    demo = gr.Interface(
        fn=run_pipeline,
        inputs=gr.Textbox(label="Paste text mentioning Tulane University", lines=4),
        outputs="text",
        title="🧠 DSPY Sentiment Pipeline Explorer",
        description="Interactive demo for Lab 3: Sentiment analysis about Tulane University.",
    )
    return demo


# ---------- LAB 4 Setup ----------
def lab4_setup():
    """
    Setup for Lab 4
    """
    import sys, os, math, numpy as np
    import matplotlib.pyplot as plt
    install_core_deps()
    seed_everything(42)
    init_openai()
    if '/content/main' not in sys.path:
        sys.path.append('/content/main')


def simple_keyword_search(query, docs):
    """
    Return documents containing the query words (case-insensitive).
    """
    results = [d for d in docs if query.lower().split()[0] in d.lower()]
    return results or ["(no matches found)"]


def lab4_generate_visualization(texts, model="text-embedding-3-small"):
    """
    Generate and display a 2D PCA projection of embeddings for a list of text strings.

    Args:
        texts (list of str): Sentences or words to embed and plot.
        model (str): OpenAI embedding model to use.

    Returns:
        numpy.ndarray: 2D PCA-transformed coordinates of embeddings.
    """
    from openai import OpenAI
    from sklearn.decomposition import PCA

    client = OpenAI()
    print(f"🔢 Generating embeddings for {len(texts)} texts...")
    embeddings = [client.embeddings.create(input=t, model=model).data[0].embedding for t in texts]

    # Reduce to 2D
    pca = PCA(n_components=2)
    proj = pca.fit_transform(np.array(embeddings))

    # --- Plot ---
    plt.figure(figsize=(6, 5))
    plt.scatter(proj[:, 0], proj[:, 1], color="teal", s=60)
    for i, t in enumerate(texts):
        plt.text(proj[i, 0] + 0.01, proj[i, 1], t)
    plt.title("PCA Projection of Embeddings")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.show()

    return proj



def lab4_build_search_demo(search_fn, docs=None):
    """
    Build the interactive Gradio demo for Lab 4.

    Args:
        search_fn: function(query, docs) -> list of results (required)
        docs: list of text documents to search (optional)
    """

    if docs is None:
        docs = [
            "Tulane University is located in New Orleans.",
            "Jazz music was born in New Orleans.",
            "Crawfish season peaks in early spring.",
            "AI models learn from data patterns, not logic rules.",
        ]

    def run_search(query):
        if not query.strip():
            return "Please enter a query."
        try:
            results = search_fn(query, docs)
            if isinstance(results, (list, tuple)):
                return "\n\n".join(results[:3])
            else:
                return str(results)
        except Exception as e:
            return f"⚠️ Error in search function: {e}"

    demo = gr.Interface(
        fn=run_search,
        inputs=gr.Textbox(label="Enter a query:"),
        outputs=gr.Textbox(label="Top results"),
        title="Lab 4 – Semantic Search Explorer",
        description="Run your own semantic search implementation interactively.",
    )

    return demo


# ---------- LAB 5 Setup ----------
def lab5_setup():
    """
    Setup for Lab 5
    """
    import sys, os, math, numpy as np
    import matplotlib.pyplot as plt
    install_core_deps()
    seed_everything(42)
    init_openai()
    if '/content/main' not in sys.path:
        sys.path.append('/content/main')


def get_sample_corpus(name: str = "mini_wiki"):
    """
    Returns a small corpus for RAG experiments.

    Args:
        name (str): corpus variant name (currently only 'mini_wiki')

    Returns:
        list[str]: list of paragraph-length texts
    """
    import textwrap
    if name != "mini_wiki":
        raise ValueError("Only 'mini_wiki' corpus is supported for now.")

    corpus = [
        textwrap.dedent("""
            The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
            It was named after the engineer Gustave Eiffel, whose company designed and built the tower.
            Constructed from 1887 to 1889, it was initially criticized by some of France's leading artists
            but has become a global cultural icon of France and one of the most recognizable structures in the world.
        """).strip(),

        textwrap.dedent("""
            The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest
            in the Amazon biome that covers most of the Amazon basin of South America.
            This region includes territory belonging to nine nations and is known for its biodiversity.
        """).strip(),

        textwrap.dedent("""
            The Great Pyramid of Giza is the oldest and largest of the three pyramids in the Giza pyramid complex
            bordering present-day Giza in Greater Cairo, Egypt. It is the oldest of the Seven Wonders of the Ancient World.
        """).strip(),

        textwrap.dedent("""
            The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.
            It extends from the Arctic Ocean in the north to the Southern Ocean in the south.
        """).strip(),

        textwrap.dedent("""
            Machine learning is a field of computer science that uses statistical techniques
            to give computer systems the ability to 'learn' from data, without being explicitly programmed.
        """).strip(),
    ]

    return corpus


# -----------------------------
# 🧠 RAG Answer Generator
# -----------------------------
def lab5_generate_answer(query: str, context: str, model: str = "gpt-4o-mini", max_tokens: int = 150):
    """
    Generates a grounded answer using a small LLM call.

    Args:
        query (str): user question
        context (str): retrieved text snippets joined together
        model (str): OpenAI model (default: gpt-4o-mini)
        max_tokens (int): response length limit

    Returns:
        str: generated model answer
    """
    system_prompt = (
        "You are a helpful teaching assistant answering student questions. "
        "Only use information from the provided context. If unsure, say 'I don’t know.'"
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("⚠️ OpenAI call failed. Returning simulated answer.")
        # Offline fallback: a simple heuristic
        if "Paris" in context:
            return "The capital of France is Paris."
        if "Amazon" in context:
            return "The Amazon rainforest is in South America."
        if "Pyramid" in context:
            return "The Great Pyramid of Giza is in Egypt."
        return "I don’t know based on the given context."


def get_text_embedding(text: str, model: str = "text-embedding-3-small"):
    """
    Compute an embedding vector for the given text using the OpenAI API.

    Args:
        text (str): input text to embed
        model (str): embedding model (default: text-embedding-3-small)

    Returns:
        np.ndarray: normalized embedding vector
    """
    from openai import OpenAI
    try:
        client = OpenAI()
        response = client.embeddings.create(input=text, model=model)
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        # Normalize to unit length for cosine similarity
        vec = vec / np.linalg.norm(vec)
        return vec
    except Exception as e:
        print(e)
        print("⚠️ Embedding call failed, returning random vector for fallback.")
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.rand(1536)
        vec = vec / np.linalg.norm(vec)
        return vec

# -----------------------------
# 🧩 Gradio Demo Builder
# -----------------------------
def lab5_build_demo(retrieve_fn, chunk_fn, embed_fn):
    """
    Creates a Gradio interface for experimenting with RAG retrieval.

    Students can type a query and see which chunks are retrieved and how the model answers.

    Args:
        retrieve_fn: function(query, chunks, chunk_embeddings, k)
        chunk_fn: function(text, chunk_size, overlap)
        embed_fn: function(chunks)

    Returns:
        gr.Interface: Gradio app
    """
    corpus = get_sample_corpus()
    all_text = "\n".join(corpus)
    chunks = chunk_fn(all_text, chunk_size=150, overlap=30)
    chunk_embeddings = embed_fn(chunks)

    def run_rag(query, k, chunk_size):
        chunks_local = chunk_fn(all_text, chunk_size=int(chunk_size))
        embeddings_local = embed_fn(chunks_local)
        retrieved = retrieve_fn(query, chunks_local, embeddings_local, k=int(k))
        context = "\n\n".join(retrieved)
        answer = lab5_generate_answer(query, context)
        return context, answer

    with gr.Blocks() as demo:
        gr.Markdown("### 🧩 Mini RAG Explorer — Lab 5")
        query = gr.Textbox(label="Your question:")
        k = gr.Slider(1, 5, value=3, step=1, label="Top-k")
        chunk_size = gr.Slider(50, 400, value=150, step=50, label="Chunk size")
        btn = gr.Button("Run RAG")
        ctx_out = gr.Textbox(label="Retrieved Context")
        ans_out = gr.Textbox(label="Generated Answer")

        btn.click(fn=run_rag, inputs=[query, k, chunk_size], outputs=[ctx_out, ans_out])

    return demo