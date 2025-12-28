# Course Project – Intro to AI Engineering  

**Semester-Long Team Project**

## 1. Overview

Over the semester, you will work in a small team (up to 2 students; grads solo) to design, build, and study a **practical AI-powered system** using tools from this course:

- LLMs via APIs 
- Colab notebooks
- Gradio for a simple web UI
- Concepts like RAG, basic agents, safety/guardrails, evaluation, and monitoring

By the end, you should feel:  
> “I know how to build something *real* with AI, and I know how to reason about whether it’s safe, reliable, and actually helpful.”

You’ll reach this through **four milestones**:

1. PM1 – Project Proposal  
2. PM2 – MVP Prototype  
3. PM3 – GenAI Ops Experiment  
4. PM4 – Final System & Report (plus presentation)

---

## 2. Learning Goals

By completing this project, you will:

- Practice **problem definition**: choose a real user/problem where AI might help.
- Design a **multi-step AI system** (not just a single prompt).
- Apply course concepts:
  - RAG and/or simple agents/tool use
  - Safety, guardrails, and trust-aware UX
  - Evaluation, metrics, and basic monitoring/drift thinking
- Run at least **one real experiment**:
  - Formulate a hypothesis
  - Change something in your system
  - Measure the impact
  - Reflect on the results
- Communicate your work to a non-expert audience via a demo and short writeup.

---

## 3. High-Level Requirements

Your final project must:

1. **Target a clear problem and user.**  
   - Example: “Help new Tulane students find campus information more easily”  
   - Example: “Help residents understand local environmental data”

2. Be implemented as a **Colab + Gradio app**.  
   - A user should be able to:
     - Open your notebook
     - Run the setup cell
     - Launch a Gradio interface
     - Interact with your system

3. Use both of:
   - A **RAG-style retrieval component** (e.g., over a small document set or dataset), and/or  
   - A **simple agent/tool-using component** (e.g., choosing between tools like calculator vs RAG).

4. Include **safety & trust considerations**:
   - Identify key risks or failure modes.
   - Implement at least one basic **guardrail** or **trust signal**:
     - Refusals for unsafe prompts
     - Warnings on low-confidence answers
     - Restrictions on certain categories of outputs

5. Include **evaluation / GenAI Ops elements**:
   - At least three small experiments (PM3) using:
     - A clear **Question**: “If we change X, what happens to Y?”
     - A **Hypothesis**: “We expect that… because…”
     - A controlled **Experiment** (vary exactly one component, e.g., chunk size or prompt style)
     - A **Metric** to measure (accuracy, precision/recall, hallucination rate, human ratings, etc.)
     - A **Conclusion** comparing results to your hypothesis.

6. Produce final **deliverables**:
   - Working Colab + Gradio app
   - Final written report
   - In-class presentation/demo

---

## 4. Milestones & Due Dates

**Note:** The exact due dates will be posted in the syllabus/LMS. Here is the structure:

### PM1 – Project Proposal (Week 3)

**Deliverable (~1 page, markdown or PDF):**

- **Team members** (1-2 students; grad students are solo)
- **Title** (tentative is fine)
- **Problem statement & user**  
  - What problem are you solving, and for whom?
- **Why AI / LLM?**  
  - Why is an LLM-based system appropriate here?
- **Example interactions**  
  - 3–5 example user queries + “ideal” responses.
- **Possible data sources**  
  - E.g., university docs, city open data, your own curated texts, etc.
- **Initial risks / concerns**  
  - What could go wrong? (hallucinations, bias, misuse, etc.)

---

### PM2 – MVP Prototype (Week 6)

**Deliverable: runnable notebook + short notes**

Your **Minimum Viable Product (MVP)** should:

- Run in Colab.
- Include a **working Gradio interface**.
- Show at least a **two-step AI pipeline**, for example:
  - preprocessing → LLM call  
  - or retrieval → generation  
  - or classify intent → answer
- It does *not* need to be polished or robust yet.

Include a short markdown cell in the notebook:

- **What works right now?**
- **One obvious failure mode** you have already observed.

---

### PM3 – GenAI Ops Experiment (Week 11)

**Deliverable: experiment + 5–6 paragraph writeup**

You will run **three small experiments** on your system using the scientific process:

1. **Question**  
   - Example: “If we add RAG, how does answer correctness change?”  
   - Example: “If we introduce guardrails, how many harmful outputs are blocked?”

2. **Hypothesis**  
   - “We expect that RAG will reduce hallucinations on factual questions about our knowledge base.”  
   - “We expect that guardrails will block at least half of our previously unsafe responses, with minimal false positives.”

3. **Experiment design**  
   - Define what you will change (X): e.g.,  
     - RAG vs no-RAG  
     - chunk size 100 vs 300  
     - prompt style A vs B  
     - guardrails on vs off
   - Hold everything else as constant as you can.
   - Create a small test set (e.g., 10–20 inputs).

4. **Measurement (metrics)**  
   - Choose at least **one metric**, for example:
     - % of answers judged correct  
     - precision/recall for retrieval  
     - fraction of responses that violate a safety rule  
     - average user rating on a 1–5 scale

5. **Conclusion**  
   - Summarize what you found:
     - Did the metric change in the way you predicted?
     - If not, why do you think that is?

You will include:

- A section in your notebook where you run this experiment.
- A short written summary (5–6 paragraphs) in your project report draft or a separate markdown cell.

---

### PM4 – Final System, Report, & Presentation (Week 15)

**Final deliverables:**

1. **Working Colab + Gradio app**
   - Clear instructions at the top:
     - How to run setup
     - How to launch the app
   - The app should demonstrate your **full system**, including:
     - Multi-step pipeline
     - RAG and/or agent/tool use
     - Any guardrails or trust mechanisms
   - It should be reasonably robust for a live demo.

2. **Final report (4–6 pages or equivalent markdown/slides)**  
   Recommended structure:

   - **1. Introduction**
     - Problem, user, and motivation.
   - **2. System Overview**
     - High-level description.
     - Diagram: your system mapped onto the course’s **unifying system diagram** (User → Prompt → … → Output).
   - **3. Components**
     - RAG, agents/tool use, or both.
     - Guardrails / safety checks.
     - Any monitoring / logging.
   - **4. GenAI Ops Experiment(s)**
     - PM3 experiment (and any follow-ups).
     - Question, hypothesis, experimental design, metric, results, and interpretation.
   - **5. Limitations & Risks**
     - Failure cases you observed.
     - Risks you would worry about in real deployment.
   - **6. Future Work**
     - If you had 4 more weeks, what would you improve?

3. **In-class presentation/demo (5 minutes)**

   - Introduce the **problem & user** in plain language.
   - Show your **system diagram** and main components.
   - Give a short **live or recorded demo** (Gradio app).
   - Highlight **one experiment**:
     - What you tried, what changed, what you learned.
   - Briefly discuss at least one **failure** and how you thought about it.

---

## 5. Example Project Ideas

You can choose from these or propose your own:

- **Campus Info Assistant**
  - Uses RAG over campus websites / FAQs.
  - Guardrails to avoid giving medical/legal advice.
  - Experiment: RAG vs no-RAG on a set of common student questions.

- **Study Helper for a Specific Course**
  - Ingests lecture notes and problem sets.
  - Helps students by answering questions and suggesting practice problems.
  - Experiment: different prompt styles (Socratic vs direct answer).

- **Local Civic Data Explorer**
  - Uses city open data + documents to answer questions about neighborhoods, environment, or transit.
  - Experiment: different chunk sizes or retrieval strategies.

- **Writing Feedback Assistant**
  - Gives feedback on short essays or resumes.
  - Guardrails against offensive content.
  - Experiment: baseline vs enhanced prompt (rubric-based) on feedback quality.

- **Domain-Specific Safety Coach**
  - For example, “AI etiquette coach” (email tone), or “AI sustainability advisor” on simple lifestyle choices.
  - Experiment: comparing two versions of guardrails or trust signals.

You are encouraged to pick **something you actually care about**.

---

## 6. Constraints & Logistics

- **Team size:** 2 students (grad students solo)
- **Tech stack:**
  - Code: Python in Colab.
  - UI: Gradio.
  - LLMs: via course-approved APIs/proxies or local models.
- **Reuse of lab code:**
  - Strongly encouraged.
  - You should adapt lab patterns (RAG, agents, eval, guardrails) to your project.
- **Data:**
  - Use only data that is legal and appropriate to use.
  - No sensitive personal data.

If you are unsure whether your idea is in scope or allowed, ask the instructor early.

---

## 7. Grading Rubric

Total project score: **100 points**, broken down as follows.

### 1. Problem Definition & Design (20 points)

| Criterion                                          | Excellent (18–20)                                                                 | Good (14–17)                                                        | Needs Work (0–13)                                                                 |
|---------------------------------------------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Clear problem & user                              | Problem and target users are clearly defined and realistic.                        | Problem is understandable but somewhat vague or too broad.           | Problem is unclear, trivial, or not well scoped.                                  |
| Motivation for AI / LLM                           | Explains convincingly why an AI system is appropriate and what it adds.            | Some justification is given but not very detailed.                   | Weak or missing justification; could have been solved easily without AI.          |
| System design & diagram                           | Diagram is clear, accurate, and maps well to the course’s unifying diagram.        | Diagram is present but missing some components or is confusing.      | No diagram, or diagram does not reflect the actual system.                        |

---

### 2. Technical Implementation (30 points)

| Criterion                                          | Excellent (27–30)                                                                 | Good (20–26)                                                        | Needs Work (0–19)                                                                 |
|---------------------------------------------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Functionality & robustness                        | System runs smoothly; Gradio app is usable; few obvious bugs.                      | System mostly works; minor bugs or rough edges.                       | System is very fragile or does not run without major instructor intervention.     |
| Use of course techniques (RAG/agents/pipelines)   | Thoughtful use of multi-step pipelines and RAG and/or tool-using agents.           | At least one of RAG or agents is present but basic.                   | Only single-prompt calls; minimal or no integration of course techniques.         |
| Integration & modularity                          | Code is reasonably modular and readable; good use of helpers; reuse from labs.     | Code is a bit messy but understandable; some reuse of lab patterns.   | Code is hard to follow, heavily copy-pasted, or not consistent with course tools. |

---

### 3. GenAI Ops & Evaluation (25 points)

| Criterion                                          | Excellent (23–25)                                                                 | Good (17–22)                                                        | Needs Work (0–16)                                                                 |
|---------------------------------------------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Scientific framing (Question & Hypothesis)        | Clear question and thoughtful hypothesis grounded in course concepts.              | Question/hypothesis present but somewhat vague or shallow.            | Missing or very unclear question and/or hypothesis.                               |
| Experiment design (PM3)                           | Controlled experiments (change 1 variable at a time), appropriate test set, clear procedure.  | Somewhat controlled experiment; test set is small or uneven.         | Uncontrolled comparison; unclear how results were produced.                       |
| Metrics & measurement                             | At least one meaningful metric is used; appropriate for the question.              | Metric is used but could be better chosen or explained.              | No real metric, or metric does not match the question.                            |
| Analysis & conclusion                             | Results are interpreted thoughtfully; discrepancies with hypothesis are discussed. | Results are stated but analysis is brief or superficial.             | Little or no interpretation beyond “X is higher than Y”.                          |

---

### 4. Safety, Risk, & Trust (15 points)

| Criterion                                          | Excellent (14–15)                                                                 | Good (10–13)                                                       | Needs Work (0–9)                                                                   |
|---------------------------------------------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Identification of risks & failure modes           | Clear discussion of main risks and failure types (hallucinations, misuse, etc.).   | Some key risks mentioned but not deeply discussed.                   | Risks barely considered or not specific to the project.                           |
| Guardrails & trust signals                        | At least one guardrail and one trust mechanism (e.g., warnings, refusals) present and tested. | Some guardrails or trust features, but not clearly integrated.       | No meaningful guardrails or trust cues implemented.                               |

---

### 5. Communication & Presentation (10 points)

| Criterion                                          | Excellent (9–10)                                                                  | Good (7–8)                                                         | Needs Work (0–6)                                                                   |
|---------------------------------------------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Written report clarity                             | Report is clear, well-structured, and easy for a non-expert to follow.             | Report is understandable but somewhat disorganized.                  | Report is confusing, incomplete, or missing sections.                             |
| In-class presentation & demo                      | Presentation is engaging, on time, and shows the system and experiment clearly.    | Presentation covers most elements but is rushed or hard to follow.   | Presentation is missing key elements or the demo does not work at all.            |

---

### Late Policy & Academic Integrity

- **Late milestones** may receive partial credit depending on how late they are and prior communication.
- You must **cite any external code or resources** you use.
- You may use AI tools to help, but:
  - You must understand and be able to explain your code.
  - You must note in your report where AI tools were used for code/text.

---

If you have questions at any point about scope, feasibility, or grading, **please ask early**.  
I’m happy to help you shape an idea into something that’s both **interesting** and **doable** within the semester.



