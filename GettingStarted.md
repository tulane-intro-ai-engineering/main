## 🧭 Getting Started with Google Colab and the Course Materials

Welcome to **Introduction to AI Engineering**!  
This short guide will help you get started using Google Colab, running lecture and lab notebooks, and setting up your OpenAI access.

---

### 🎒 1. What You’ll Need

Before you begin, please make sure you have:

- A **Google Account** (for Colab access)  
- A free **OpenAI API key** (see below)  
- Internet access and your Tulane GitHub Classroom credentials  

---

### 🚀 2. Using Google Colab

All lectures and labs in this course run entirely in **Google Colab** — a cloud-based Python notebook environment.  
That means you don’t need to install anything on your computer.

#### To open a notebook in Colab:

1. Go to the course GitHub repository:  
   👉 [https://github.com/tulane-intro-ai-engineering/main](https://github.com/tulane-intro-ai-engineering/main)

2. Find the notebook you want to open:  
   - **`lectures/`** → in-class lecture demos  
   - **`labs/`** → hands-on, graded lab notebooks  

3. Click on a notebook file (for example, `labs/intro_lab.ipynb`).

4. At the top of the notebook preview, click the **Open in Colab** badge:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tulane-intro-ai-engineering/main/blob/main/labs/intro_lab.ipynb)

5. Once the notebook opens in Colab, click  
   **File → Save a copy in Drive**  
   to create your own editable version.  
   This ensures your work is saved to your Google Drive.

6. After saving your copy, select **Runtime → Run all** to execute the notebook cells.

---

### ⚡ 3. Useful Colab Tips & Shortcuts

Colab is very similar to Jupyter notebooks, but runs in your browser.  
Here are a few handy keyboard shortcuts that will save you time:

| Action | Shortcut |
|---------|-----------|
| **Run a single cell** | `Shift` + `Enter` |
| **Run a cell and move to the next one** | `Ctrl` (or `Cmd` on Mac) + `Enter` |
| **Add a new cell below** | `Alt` + `Enter` |
| **Delete a cell** | `Ctrl` + `M` then `D` |
| **Undo delete cell** | `Ctrl` + `M` then `Z` |
| **Open command palette** | `Ctrl` + `Shift` + `P` |

👉 *You can also click in a cell and press “Play” (▶️) on the left to run it.*

---

### 🧰 4. Setting Up Your Lab Environment

Each lab begins with a **Setup** cell that prepares your environment in Colab.  
It will look something like this:

```python
# @title Setup
from course_utils import lab11_setup
lab11_setup()
```

When you run this cell, it will:

- Clone or update the course repository in your Colab environment  
- Add helper functions and utilities for the lab  
- Prompt you to **paste in your OpenAI API key** (you do *not* need to set it manually)  

You’ll only need to provide your API key the first time you run a lab each session.

---

### 🔑 5. Getting an OpenAI API Key

You’ll need an **OpenAI API key** to use GPT models in the labs.  
To request your student key, follow the instructions in:  
📄 [`OpenAI-Students.md`](./OpenAI-Students.md)

That document explains how to:

- Create (or sign in to) your OpenAI account  
- Retrieve your API key from your OpenAI dashboard  
- Copy and paste your key when prompted in the setup cell  

⚠️ **Never share or commit your API key** — it’s unique to you and provides API access.

---

### 🧠 6. How Lectures and Labs Fit Together

Each week follows a pattern:

- **Lectures**: Introduce new ideas with live, runnable demos  
- **Labs**: Let you test and explore those ideas hands-on  

Every notebook (lecture or lab) includes a **“Setup”** section with a function like:

```python
from course_utils import lab4_setup
lab4_setup()
```

Just run it once at the start of the notebook to prepare your workspace.

---

### 💾 7. Turning In Labs

When you finish editing your lab notebook in Colab:

1. Make sure all your cells run without errors.  
   You can click **Runtime → Run all** to double-check.  

2. Save your work by choosing:  
   **File → Save** (this updates your Google Drive copy).  

3. Then download your notebook as a `.ipynb` file:  
   **File → Download → Download .ipynb**  

4. Finally, upload the downloaded `.ipynb` file to **Canvas** under the corresponding lab assignment.  

✅ **Tip:** You can reopen your saved copy later by visiting your Google Drive — it will appear under the folder `Colab Notebooks`.

---

### 💡 8. Troubleshooting Tips

If something isn’t working in Colab:

- Re-run the **Setup** cell (Colab environments reset automatically after inactivity).  
- If you’re prompted again for your OpenAI key, paste it in when asked.  
- Restart the runtime (**Runtime → Restart runtime**) and re-run all cells.  
- If the issue persists, ask a TA or instructor for help.

---

### 🎉 You’re Ready!

You now have everything you need to begin your journey into AI engineering.  
Start with your first lecture notebook:  
👉 [`lectures/intro_lecture.ipynb`](https://colab.research.google.com/github/tulane-intro-ai-engineering/main/blob/main/lectures/intro_lecture.ipynb)  

Click the **Open in Colab** badge, **Save a copy in Drive**, run the setup cell — and you’re ready to build your first generative AI system!
