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

- Open the **instructor-only** notebook `instructor_dev.ipynb` in Colab. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/tulane-intro-ai-engineering/main/blob/main/instructor_dev.ipynb)
- Make a copy to your google drive (anywhere; you'll need this copy outside the repo)
- Edit the top cell to determine where to clone the repo into your google drive, set your name, etc.
- Run the first two cells to clone the repo to google drive.

---

## 2. Editing class files

To begin editing, launch your copy of `instructor_dev.ipynb` and run the section `🔧 Start of each session: mount & pull`. 
This will mount the drive and pull latest updates.

Once you’ve pulled the latest changes, open notebooks directly from the Drive-backed clone:

1. From drive, navigate to /content/drive/MyDrive/Teaching/IntroAIEngineering/github/main
2. Click a notebook to open it in a new Colab tab.
3. Edit cells, run, and test — you are editing the real file in the Git clone.
4. When Colab autosaves (or when you press Ctrl+S), it writes directly into your Drive-backed repo.

---

## 3. Committing & Pushing Changes Back to GitHub

When you’re done editing (could be multiple notebooks plus course_utils.py, etc.), go back to instructor_dev.ipynb and run the 
`💾 Commit and push changes` block.

