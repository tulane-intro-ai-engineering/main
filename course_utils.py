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


# ---------- LAB 6 Setup ----------
def lab6_setup():
    """
    Setup for Lab 6
    """
    import sys, os, math, numpy as np
    import matplotlib.pyplot as plt
    install_core_deps()
    seed_everything(42)
    init_openai()
    if '/content/main' not in sys.path:
        sys.path.append('/content/main')
        
# -------------------------
# Corpus + chunking
# -------------------------
def lab6_get_corpus() -> List[Dict[str, Any]]:
    """
    Return a tiny document set used for RAG retrieval in Lab 6.
    Keep this small and readable: students should be able to inspect it.
    """
    return [
        {
            "doc_id": "policy_oncall",
            "title": "On-Call Rotation Policy",
            "text": (
                "The on-call rotation is required for full-time engineers and optional for interns.\n"
                "Interns may join on-call only after completing onboarding and receiving manager approval.\n"
                "Interns should start with shadow shifts."
            ),
            "meta": {"type": "policy", "updated": "2025-11-15"},
        },
        {
            "doc_id": "policy_access",
            "title": "Access Control Policy",
            "text": (
                "Interns are granted access to internal tools in the first week.\n"
                "Interns may NOT access customer production data.\n"
                "Elevated access requires manager approval."
            ),
            "meta": {"type": "policy", "updated": "2025-09-01"},
        },
        {
            "doc_id": "runbook_incidents",
            "title": "Incident Response Runbook",
            "text": (
                "Responders should: (1) acknowledge the page, (2) assess severity, (3) mitigate,\n"
                "(4) communicate updates, and (5) write a postmortem."
            ),
            "meta": {"type": "runbook", "updated": "2025-05-02"},
        },
    ]

def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())

def _chunk_text_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    w = _words(text)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks: List[str] = []
    start = 0
    while start < len(w):
        end = min(start + chunk_size, len(w))
        chunks.append(" ".join(w[start:end]))
        if end == len(w):
            break
        start = end - overlap
    return chunks

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

@dataclass
class Lab6Retriever:
    """A minimal retriever object: chunks + embedding matrix + top_k."""
    chunks: List[Dict[str, Any]]
    X: np.ndarray
    top_k: int

def lab6_build_retriever(
    corpus: List[Dict[str, Any]],
    chunk_size: int = 60,
    overlap: int = 15,
    top_k: int = 4,
) -> Lab6Retriever:
    """
    Build a simple embedding-based retriever over chunked documents.
    Uses get_text_embedding(text) from course_utils.
    """
    # Late import to avoid circular issues if course_utils defines get_text_embedding below this code.
    from course_utils import get_text_embedding  # type: ignore

    chunks: List[Dict[str, Any]] = []
    for d in corpus:
        for i, ch in enumerate(_chunk_text_words(d["text"], chunk_size=chunk_size, overlap=overlap)):
            chunks.append(
                {
                    "chunk_id": f"{d['doc_id']}::c{i}",
                    "doc_id": d["doc_id"],
                    "title": d.get("title", ""),
                    "text": ch,
                    "meta": d.get("meta", {}),
                }
            )

    # Embed + normalize for cosine similarity
    X = np.vstack([_normalize(np.array(get_text_embedding(c["text"]))) for c in chunks]).astype(np.float32)
    return Lab6Retriever(chunks=chunks, X=X, top_k=int(top_k))

def lab6_rag_retrieve(query: str, retriever: Lab6Retriever) -> Dict[str, Any]:
    """
    Retrieve top-k passages for a query.
    Returns dict: {"passages": [str, ...], "scores": [float, ...]}
    """
    from course_utils import get_text_embedding  # type: ignore

    q = _normalize(np.array(get_text_embedding(query))).astype(np.float32)
    sims = retriever.X @ q
    k = max(1, int(retriever.top_k))
    idx = np.argsort(-sims)[:k]

    passages: List[str] = []
    scores: List[float] = []
    for i in idx:
        c = retriever.chunks[int(i)]
        passages.append(f"[{c['chunk_id']}] {c['text']}")
        scores.append(float(sims[int(i)]))

    return {"passages": passages, "scores": scores}


# -------------------------
# Calculator tool
# -------------------------
_ALLOWED_MATH = set("0123456789+-*/(). %")

def lab6_calculator(expression: str) -> Dict[str, Any]:
    """
    A tiny calculator tool.
    Supports basic arithmetic; rejects unexpected characters.
    """
    expr = (expression or "").strip()
    if not expr:
        return {"error": "Empty expression."}
    if any(ch not in _ALLOWED_MATH for ch in expr):
        return {"error": "Expression contains disallowed characters."}
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # no builtins
        # Keep result JSON-friendly
        if isinstance(result, (int, float)) and (math.isfinite(result) if isinstance(result, float) else True):
            return {"result": result}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


# -------------------------
# Optional LLM answer helper
# -------------------------
def _openai_client():
    """
    Best-effort OpenAI client creation.
    - Uses OPENAI_API_KEY from environment.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None

    try:
        # New OpenAI SDK (v1+)
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def lab6_generate_answer(
    question: str,
    passages: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    """
    Generate an answer (optionally grounded in retrieved passages).
    If OPENAI_API_KEY is missing, returns a safe placeholder.

    This keeps notebook code simple: the notebook passes question + passages.
    """
    client = _openai_client()
    if client is None:
        # Safe fallback for environments without keys
        if passages:
            return (
                "⚠️ (LLM not configured) Here are the retrieved passages you should use:\n\n"
                + "\n\n".join(passages[:3])
            )
        return "⚠️ (LLM not configured) Please set OPENAI_API_KEY to generate answers."

    context_block = ""
    if passages:
        context_block = "SOURCES:\n" + "\n".join(passages) + "\n\n"

    system = (
        "You are a helpful course assistant. "
        "If SOURCES are provided, answer using ONLY those sources. "
        "If the sources do not contain the answer, say 'I don't know based on the provided sources.' "
        "Be concise."
    )
    user = f"{context_block}QUESTION: {question}\nANSWER:"

    # New SDK call style
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI call failed: {e}"


# -------------------------
# Default eval set (tool-choice)
# -------------------------
def lab6_default_eval_set() -> List[Dict[str, Any]]:
    """
    Returns a small labeled dataset:
    each item: {"q": ..., "gold_tool": "calculator"|"rag"|"none"}

    Keep small so students can inspect it.
    """
    return [
        {"q": "What is 17% of 84? Use arithmetic.", "gold_tool": "calculator"},
        {"q": "Compute (12+5)*3.", "gold_tool": "calculator"},
        {"q": "According to the policy, can interns join the on-call rotation?", "gold_tool": "rag"},
        {"q": "What does the runbook say to do during an incident?", "gold_tool": "rag"},
        {"q": "Explain in one sentence what an agent is.", "gold_tool": "none"},
        {"q": "Write a short analogy for embeddings.", "gold_tool": "none"},
        # A couple “tricky” ones:
        {"q": "According to the docs, what is 12+5?", "gold_tool": "calculator"},  # docs phrase but math dominates
        {"q": "Policy question: do interns have production access?", "gold_tool": "rag"},
    ]


# -------------------------
# Gradio demo builder
# -------------------------
def lab6_build_demo(
    policy_fn: Callable[[str], str],
    agent_step_fn: Callable[..., Dict[str, Any]],
    retriever: Lab6Retriever,
):
    """
    Build a simple Gradio UI around the student's agent.
    We keep UI wiring here (not in notebooks).

    agent_step_fn is expected to accept:
      agent_step_fn(question, policy=policy_fn, retriever=retriever, use_llm_for_final_answer=True/False)
    """
    import gradio as gr

    def run(question: str, use_llm: bool):
        out = agent_step_fn(
            question,
            policy=policy_fn,
            retriever=retriever,
            use_llm_for_final_answer=use_llm,
        )
        tool = out.get("tool", "")
        trace = " → ".join(out.get("trace", []))
        answer = out.get("answer", "")
        tool_output = out.get("tool_output", None)
        tool_output_str = json.dumps(tool_output, indent=2) if tool_output is not None else ""
        return tool, trace, tool_output_str, answer

    demo = gr.Interface(
        fn=run,
        inputs=[
            gr.Textbox(label="Question", value="According to the policy, can interns join on-call?"),
            gr.Checkbox(label="Use LLM to write final answer (optional)", value=True),
        ],
        outputs=[
            gr.Textbox(label="Chosen tool"),
            gr.Textbox(label="Tool trace"),
            gr.Textbox(label="Tool output (JSON)"),
            gr.Textbox(label="Answer"),
        ],
        title="Lab 6: Tool-Choosing Agent (Calculator vs RAG)",
        description="Enter a question. The agent chooses a tool, calls it, and returns an answer + trace.",
    )
    return demo