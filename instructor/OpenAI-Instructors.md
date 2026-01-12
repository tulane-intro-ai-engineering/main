## 1) Instructor guide (setup: 1 project per student)

**Goal:** Each student has their *own* project + project-scoped API key; all usage bills to your org.

1. **Create/confirm your Organization**

* Log into the OpenAI Platform and ensure you have an **Organization** where you are **Owner** (needed to create projects, set limits, manage billing).

2. **Add funding (recommended: prepaid credits)**

* Go to **Billing** → **Add to balance** (and optionally set **auto-recharge** + a recharge cap).

3. **Create 40 projects**

* **Projects** → **Create project**
* Naming convention: `ClassName-Student-01` … `ClassName-Student-40`

4. **Set per-project limits (repeat or copy pattern)**

* In each project: **Limits**

  * Allow only the models you want students to use
  * Set reasonable **rate limits**
  * Set **budget alerts** (note: alerts/soft thresholds are for monitoring)
 
* You can also use [setup_student_openai.ipynb](https://github.com/tulane-intro-ai-engineering/main/blob/main/instructor/setup_student_openai.ipynb) to set some of the rate limits.

5. **Invite students**

* **Organization settings** → **Members** → Invite student emails
* Role: typically **Reader** is fine for students

6. **Assign each student to exactly one project**

* Open the student’s project → **Members** → Add that student
* Double-check each student only appears in their assigned project.

7. **Day-of checklist**

* Students log in, switch to your org, open their project, create a key, save it locally.
* Monitor usage in **Usage/Cost** views and be ready to remove a student from a project if a key leaks.



