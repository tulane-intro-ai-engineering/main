# Campus Info Assistant - Starter Template

## Setup
```python
# Standard setup cell with Google Drive mounting
from google.colab import drive
drive.mount('/content/drive')

!git clone --depth 1 -q https://github.com/tulane-intro-ai-engineering/main.git
import sys; sys.path.append('/content/main')
from course_utils import lab5_setup, get_text_embedding
lab5_setup()
```

## Step 1: Load Documents from Google Drive
```python
from pathlib import Path

drive_path = Path("/content/drive/My Drive/AI_Project/campus_docs/")

def load_text_files_from_drive(folder_path):
    """Load all .txt files from a Google Drive folder."""
    documents = []
    for file_path in folder_path.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if len(text) > 50:
                documents.append(text)
    return documents

# Load your campus documents
corpus = load_text_files_from_drive(drive_path)
print(f"Loaded {len(corpus)} documents")
```

## Step 2: Build RAG Pipeline
```python
# TODO: Implement chunking
def chunk_text(text, chunk_size=150, overlap=30):
    # Your chunking code here
    pass

# TODO: Implement embedding
def embed_corpus(chunks):
    # Your embedding code here
    pass

# TODO: Implement retrieval
def retrieve_top_k(query, chunks, embeddings, k=3):
    # Your retrieval code here
    pass

# TODO: Implement answer generation
def generate_answer(query, context):
    # Your answer generation code here
    pass

# Build your RAG system
chunks = chunk_text(corpus)
embeddings = embed_corpus(chunks)
```

## Step 3: Add Safety Guardrails
```python
def is_medical_question(query):
    """Check if query is about medical advice."""
    medical_keywords = ["medical", "health", "diagnosis", "treatment"]
    return any(kw in query.lower() for kw in medical_keywords)

def is_legal_question(query):
    """Check if query is about legal advice."""
    legal_keywords = ["legal", "lawyer", "lawsuit", "legal advice"]
    return any(kw in query.lower() for kw in legal_keywords)

def safe_rag_answer(query):
    # Check for unsafe topics
    if is_medical_question(query) or is_legal_question(query):
        return "I can't provide medical or legal advice. Please consult a qualified professional."
    
    # Use RAG
    retrieved = retrieve_top_k(query, chunks, embeddings, k=3)
    context = "\n\n".join(retrieved)
    return generate_answer(query, context)
```

## Step 4: Create Test Set
```python
# Example queries about campus
test_set = [
    {
        "query": "When is the dining hall open?",
        "expected_keywords": ["hours", "open", "closed"],
        "category": "hours"
    },
    {
        "query": "What is the refund policy?",
        "expected_keywords": ["refund", "policy", "days"],
        "category": "policy"
    },
    # TODO: Add more test cases
]

# TODO: Implement evaluation function
def evaluate_answer(query, answer, expected_keywords):
    # Your evaluation code here
    pass
```

## Step 5: Deploy with Gradio
```python
import gradio as gr

def gradio_handler(query):
    answer = safe_rag_answer(query)
    return answer

demo = gr.Interface(
    fn=gradio_handler,
    inputs="text",
    outputs="text",
    title="Campus Info Assistant"
)

demo.launch(share=True)
```

## Step 6: Run Experiments
```python
# PM3 experiments: chunk size, top-k, etc.
# TODO: Implement experiments
# - Vary chunk size
# - Vary top-k
# - Measure recall@k and hallucination rate
```

## Connection to Milestones

- **PM1**: Complete Steps 1-2 (data loading + RAG)
- **PM2**: Add agent logic (optional, using DSPy ReAct)
- **PM3**: Complete Step 4 (test set) and Step 6 (experiments)
- **PM4**: Complete Step 5 (Gradio app)

