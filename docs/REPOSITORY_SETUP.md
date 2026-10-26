# LOCAL GIT REPOSITORY SETUP & GITHUB PUBLICATION GUIDE
## Chicago Crime Intelligence System

---

**Project Title:** Chicago Crime Intelligence System  
**Document Purpose:** Local Git Setup, Staging Verification, and Remote GitHub Deployment Guide  
**Supervisor:** Dr. S. Johanan Joysingh (`johananj`)  
**Student Authors:** A. Jai Aakaash (23MIA1092) & Tharunika Ravi Kumar (23MIA1090)  
**Target Repository:** `chicago-crime-intelligence-gnn` (GitHub)  

---

## 1. Local Git Repository Initialization Protocol

The local Git repository is prepared and staged locally for presentation to the course supervisor prior to remote publication.

### Step 1: Navigate to Project Root
```powershell
cd crime_gnn_project
```

### Step 2: Initialize Git Repository
```powershell
git init -b main
```

### Step 3: Configure Local Author Identity (If Not Configured Globally)
```powershell
git config user.name "A. Jai Aakaash"
git config user.email "jaiaakaash@vitstudent.ac.in"
```

---

## 2. Staging Inspection & Security Verification

Before staging, verify that sensitive and prohibited files are strictly ignored by Git.

### 2.1 Pre-Commit Safety Checklist
Verify the following rules:
- [x] **No Unpublished Research Papers / Manuscripts:** All `*.pdf`, `*.docx`, `*.tex`, and manuscript folders are excluded by `.gitignore`.
- [x] **No Large Raw Datasets:** Raw Chicago crime archive (`Crimes_-_2001_to_Present.csv`, 1.7+ GB) and zip archives are excluded.
- [x] **No Credentials / API Keys:** `.env`, `.pem`, tokens, and private files are excluded.
- [x] **Lightweight Artifacts Only:** Pre-trained model weights (`best_graphsage_model.pth`, 120 KB) and embeddings (`graphsage_embeddings.csv`, 19.8 MB) are within GitHub's 50 MB threshold.

### 2.2 Testing `.gitignore` Rules
Execute `git check-ignore` to verify rule enforcement:
```powershell
git check-ignore -v "draft_paper.pdf" "Crimes_-_2001_to_Present.csv" ".env"
```
*Expected Output:* Confirms that all matching patterns are ignored.

### 2.3 Staging Approved Source & Documentation Files
```powershell
git add .gitignore LICENSE CITATION.cff requirements.txt README.md PROJECT_DOCUMENTATION.md
git add run_pipeline.py verify_project.py build_academic_deck.py generate_academic_assets.py
git add config/ data/ figures/ models/ presentation_assets/ reports/ results/ dashboard/ tests/ docs/ notebooks/
```

### 2.4 Verifying Staged Files
```powershell
git status
git diff --cached --stat
```

### 2.5 Creating the Initial Review 3 Commit
```powershell
git commit -m "Initial commit: Chicago Crime Intelligence System (Review 3 Release)"
```

---

## 3. Remote GitHub Publication Protocol (When Approved)

> [!IMPORTANT]
> Do NOT push to remote GitHub until the local repository has been reviewed and explicit approval is provided.

When approval is granted to publish to GitHub:

### Step 1: Create a New Repository on GitHub
1. Log into your GitHub account: [github.com/new](https://github.com/new).
2. Repository name: `chicago-crime-intelligence-gnn`.
3. Description: `Chicago Crime Intelligence System: GraphSAGE-based crime classification, candidate link prediction, and network analytics for Forensic Analytics (CSE4074).`
4. Set visibility: **Public** (or **Private** as required by your instructor).
5. **Do NOT** initialize with a README, .gitignore, or License (these are already configured locally).

### Step 2: Link Remote and Push
```powershell
git remote add origin https://github.com/<your-username>/chicago-crime-intelligence-gnn.git
git branch -M main
git push -u origin main
```

---

## 4. Repository Health & Maintenance Commands

| Operation | Command |
| :--- | :--- |
| **Check Local Git Status** | `git status` |
| **View Commit History** | `git log --oneline --graph --decorate` |
| **Verify Repository File Sizes** | `git ls-files | ForEach-Object { Get-Item $_ } | Sort-Object Length -Descending | Select-Object -First 10` |
| **Run Regression Suite** | `python verify_project.py` |
| **Run Dashboard Unit Tests** | `python tests/test_dashboard.py` |
