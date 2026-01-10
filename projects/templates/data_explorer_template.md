# Local Data Explorer - Starter Template

## Setup
```python
# Standard setup
from google.colab import drive
drive.mount('/content/drive')

!git clone --depth 1 -q https://github.com/tulane-intro-ai-engineering/main.git
import sys; sys.path.append('/content/main')
from course_utils import lab5_setup, get_text_embedding
lab5_setup()

import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

## Step 1: Load Data
```python
# Option A: Load from Google Drive
from pathlib import Path
drive_path = Path("/content/drive/My Drive/AI_Project/data/")

def load_from_drive(folder_path):
    documents = []
    for file_path in folder_path.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if len(text) > 50:
                documents.append(text)
    return documents

corpus = load_from_drive(drive_path)

# Option B: Load from URLs
import requests
from bs4 import BeautifulSoup

def load_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    return text

# Option C: Load from CSV
import pandas as pd
# df = pd.read_csv("data.csv")
# corpus = df["text_column"].tolist()
```

## Step 2: Build RAG System
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

# Build RAG
chunks = chunk_text(corpus)
embeddings = embed_corpus(chunks)
```

## Step 3: Add Agent Logic
```python
# Define RAG as a tool
def rag_tool(query: str) -> str:
    """Answer questions using RAG over your data."""
    retrieved = retrieve_top_k(query, chunks, embeddings, k=3)
    context = "\n\n".join(retrieved)
    answer = generate_answer(query, context)
    return answer

# Create ReAct agent - it will automatically decide when to use RAG
agent = dspy.ReAct(
    signature="question -> answer",
    tools=[rag_tool],
    max_iters=5
)

# Use it
result = agent(question="What does the data say about...?")
print(result.answer)
```

## Step 4: Evaluation
```python
# Test on domain-specific questions
test_set = [
    {
        "query": "What are the main trends in the data?",
        "expected_keywords": ["trends", "data", "main"],
        "category": "analysis"
    },
    # TODO: Add more test cases
]

# TODO: Implement evaluation
def evaluate_answer(query, answer, expected_keywords):
    # Your evaluation code here
    pass
```

## Step 5: Deploy
```python
import gradio as gr

def gradio_handler(query):
    result = agent(question=query)
    return result.answer

# Build app with data visualization
with gr.Blocks() as demo:
    gr.Markdown("# Local Data Explorer")
    
    query_input = gr.Textbox(label="Question")
    answer_output = gr.Textbox(label="Answer")
    
    # TODO: Add data visualization components
    # - Charts showing data trends
    # - Tables with retrieved data
    
    submit_btn = gr.Button("Submit")
    submit_btn.click(
        fn=gradio_handler,
        inputs=query_input,
        outputs=answer_output
    )

demo.launch(share=True)
```

## Connection to Milestones

- **PM1**: Complete Steps 1-2 (data loading + RAG)
- **PM2**: Complete Step 3 (agent logic)
- **PM3**: Complete Step 4 (evaluation)
- **PM4**: Complete Step 5 (Gradio app with visualization)

