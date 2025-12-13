# Instructor Workflow – Editing & Pushing Notebooks from Colab

This document explains how to use **Google Colab + Google Drive + git** to edit lecture/lab notebooks and push changes back to GitHub **without** using Colab’s “Save a copy in GitHub” dialog or pasting a token every time.

The main class repo lives here:

> https://github.com/tulane-intro-ai-engineering/main

We will:

- Keep a **persistent clone** of the repo in your Google Drive.
- Configure the git **remote once** with your Personal Access Token (PAT).
- Use normal `git pull`, `git add`, `git commit`, `git push` commands from Colab.

---

## 0. Prerequisites

1. GitHub repo:  
   `https://github.com/tulane-intro-ai-engineering/main`
2. A GitHub **Personal Access Token (PAT)** with `repo` scope (for private repos).  
   - GitHub → Settings → Developer settings → Personal access tokens.
   - Keep this token secret. Do **not** commit it to the repo or share your Drive folder.
3. A Google Drive account (same one you use in Colab).

---

## 1. One-Time Setup: Clone the Repo into Drive & Configure Remote

You only need to do this **once per machine** (per Drive clone). Afterward, the token is stored in the repo’s local git config in your Drive, and you won’t need to paste it again.

Create or open an **instructor-only** notebook (e.g., `instructor_dev.ipynb`) in Colab. Run this cell:

```python
# 🔧 ONE-TIME SETUP: mount Drive, clone repo into it, and set remote with PAT

from google.colab import drive
from pathlib import Path
import getpass

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Choose where in Drive to store the repo
BASE_DIR = Path("/content/drive/MyDrive/Teaching/IntroAIEngineering/github")
BASE_DIR.mkdir(parents=True, exist_ok=True)

REPO_OWNER = "tulane-intro-ai-engineering"
REPO_NAME = "main"
REPO_DIR = BASE_DIR / REPO_NAME
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

%cd {BASE_DIR}

# 3. Clone the repo if it doesn't exist yet
if REPO_DIR.exists():
    print("Repo already exists at:", REPO_DIR)
else:
    print("Cloning repo...")
    !git clone {REPO_URL} {REPO_DIR.name}
    print("Cloned repo to:", REPO_DIR)

# 4. Configure git remote with your PAT (only needed once per clone)
%cd {REPO_DIR}

print("\nCurrent remotes:")
!git remote -v

token = getpass.getpass("GitHub Personal Access Token (will be stored in local git config): ")

remote_with_token = f"https://{token}@github.com/{REPO_OWNER}/{REPO_NAME}.git"
!git remote set-url origin {remote_with_token}

print("\nUpdated remotes:")
!git remote -v

print("\nOne-time setup complete. Future sessions only need the start-of-session cell.")
