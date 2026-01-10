# Study Helper - Starter Template

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

## Step 1: Load Course Materials
```python
from pathlib import Path

drive_path = Path("/content/drive/My Drive/AI_Project/course_materials/")

def load_text_files_from_drive(folder_path):
    """Load lecture notes, problem sets from Google Drive."""
    documents = []
    for file_path in folder_path.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if len(text) > 50:
                documents.append(text)
    return documents

corpus = load_text_files_from_drive(drive_path)
print(f"Loaded {len(corpus)} documents")
```

## Step 2: Build RAG + Agent System
```python
# TODO: Implement RAG pipeline
def chunk_text(text, chunk_size=150, overlap=30):
    # Your chunking code here
    pass

def embed_corpus(chunks):
    # Your embedding code here
    pass

def retrieve_top_k(query, chunks, embeddings, k=3):
    # Your retrieval code here
    pass

def generate_answer(query, context):
    # Your answer generation code here
    pass

# Build RAG
chunks = chunk_text(corpus)
embeddings = embed_corpus(chunks)

# Define RAG as a tool for DSPy ReAct
def rag_tool(query: str) -> str:
    """Answer questions using RAG over course materials."""
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
result = agent(question="What is RAG?")
print(result.answer)
```

## Step 3: Add Evaluation
```python
# Test on practice questions
test_set = [
    {
        "query": "What is the difference between RAG and fine-tuning?",
        "expected_keywords": ["RAG", "fine-tuning", "difference"],
        "category": "concepts"
    },
    # TODO: Add more test cases
]

# TODO: Implement evaluation
def evaluate_answer(query, answer, expected_keywords):
    # Your evaluation code here
    pass
```

## Step 4: Deploy
```python
import gradio as gr

def gradio_handler(query):
    result = agent(question=query)
    return result.answer

# Build multi-tab app
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Query"):
            query_input = gr.Textbox(label="Question")
            answer_output = gr.Textbox(label="Answer")
            submit_btn = gr.Button("Submit")
            submit_btn.click(
                fn=gradio_handler,
                inputs=query_input,
                outputs=answer_output
            )
        
        with gr.Tab("Practice Problems"):
            # TODO: Add practice problem interface
            pass
        
        with gr.Tab("Evaluation"):
            # TODO: Add evaluation interface
            pass

demo.launch(share=True)
```

## Connection to Milestones

- **PM1**: Complete Steps 1-2 (data loading + RAG + Agent)
- **PM2**: Enhance agent with more tools (optional)
- **PM3**: Complete Step 3 (evaluation)
- **PM4**: Complete Step 4 (Gradio app with tabs)

