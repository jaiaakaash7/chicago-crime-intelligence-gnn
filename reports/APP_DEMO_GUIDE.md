# Chicago Crime Intelligence System: Application Launch & Demo Guide
## Windows Execution Instructions, Troubleshooting & Offline Backup Demo Plan

**System URL**: `http://127.0.0.1:7860` (Strictly Localhost Web Server, `share=False`)  
**Target Environment**: Windows 10/11 / Windows Server, Python 3.14 / 3.11+, PyTorch 2.6  
**Primary Launch Batch**: `START_HERE.bat` or `run_app.bat`  

---

## Part 1: Standard Windows Launch Instructions

### Method A: One-Click Batch Launch (Recommended for Presentations)
1. Open Windows File Explorer and navigate to:
   ```
   C:\Users\jaiaa\.gemini\antigravity\scratch\crime_gnn_project
   ```
2. Double-click **`START_HERE.bat`** (or `run_app.bat`).
3. A terminal window will open, verify environment packages, check pre-computed artifacts, and initialize the Gradio server.
4. When you see:
   ```
   * Running on local URL:  http://127.0.0.1:7860
   ```
5. Open Google Chrome, Microsoft Edge, or Firefox and go to:
   ```
   http://127.0.0.1:7860
   ```

---

### Method B: Manual PowerShell / Command Prompt Launch
If you prefer launching from an existing terminal:
```powershell
# 1. Navigate to project root directory
cd C:\Users\jaiaa\.gemini\antigravity\scratch\crime_gnn_project

# 2. Verify Python environment
python --version
python -c "import torch, torch_geometric, gradio; print('Environment OK!')"

# 3. Launch Gradio Dashboard
python app/main.py
```

Expected Terminal Output:
```
============================================================
CHICAGO CRIME INTELLIGENCE SYSTEM
GraphSAGE + Network Analytics + Crime Link Prediction
============================================================
[INFO] Loading preprocessed data and graph...
[INFO] Graph loaded: 30,000 nodes, 189,474 directed edges.
[INFO] Loading trained GraphSAGE checkpoint...
[INFO] Initializing 8-Tab Gradio Interface...
* Running on local URL:  http://127.0.0.1:7860
To create a public link, set `share=True` in `launch()`.
```

---

## Part 2: Port & Connection Troubleshooting

### Problem 1: Port 7860 Already in Use (`Address already in use`)
If a previous instance of Gradio is running in the background:
```powershell
# Find process using port 7860
Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue | Select-Object OwningProcess

# Or find and terminate existing python process:
Get-Process python | Stop-Process -Force
```
*Alternative*: Specify an alternate port when running `python app/main.py --port 7861` or adjust `demo.launch(server_port=7861)` in `app/main.py`.

---

### Problem 2: Slow Initial Tab Loading
- **Cause**: Generating high-dimensional t-SNE or computing all-pairs similarity dynamically.
- **Solution**: The system uses pre-computed embeddings (`outputs/graphsage_embeddings.pt`) and pre-scored candidate links (`outputs/candidate_crime_links.csv`), allowing queries to respond in $<200\text{ ms}$. If memory is constrained, ensure background heavy applications are closed.

---

### Problem 3: Browser Cache Issues
If the dashboard appears with older CSS or blank containers:
- Press `Ctrl + F5` (Hard Reload) in Chrome/Edge to clear cached UI assets.

---

## Part 3: Offline Backup Demo Plan (Contingency Protocol)

If an unexpected technical failure occurs during an in-person defense (e.g., laptop battery glitch, port lock, or missing graphics driver), **do NOT panic**. Follow this structured 3-tier contingency plan:

### Contingency Tier 1: Static Pre-Computed Visual Figures
All 10 analytical figures are pre-rendered at 300 DPI in the `figures/` directory:
- Open `figures/` in Windows Photos or Image Viewer.
- **Tab 1 & 2 Backup**: Display `figures/01_crime_trend_by_year.png` and `figures/03_hourly_crime_distribution.png`.
- **Tab 3 Backup**: Display `figures/07_network_degree_distribution.png`.
- **Tab 4 & 5 Backup**: Display `figures/06_graphsage_vs_mlp_comparison.png` and `figures/10_gnn_similarity_distribution.png`.
- **Tab 6 Backup**: Display `figures/09_crime_link_score_distribution.png`.
- **Tab 7 Backup**: Display `figures/08_community_size_chart.png`.

---

### Contingency Tier 2: Slide 10 & Architecture Deck
- Switch directly to **Slide 10** in `Chicago_Crime_Intelligence_GNN_10_Slides.pptx`.
- Walk the committee through the card architecture of the 8 tabs, explaining the exact inputs, mathematical transformations, and outputs of each tab.

---

### Contingency Tier 3: Direct Inspection of Verified Output Files
Show the examiners the raw pre-computed artifacts to prove zero fabrication:
1. **Model Checkpoints**: `models/graphsage_best.pt` and `models/mlp_baseline_best.pt`.
2. **Evaluated Test Results**: Open `reports/FINAL_REPORT.md` or `outputs/test_metrics.csv` to show exact numbers (GraphSAGE Accuracy: 0.1511, MLP Accuracy: 0.1222).
3. **Candidate Crime Links Table**: Open `outputs/candidate_crime_links.csv` in Excel to show the top scored links for Node 587 (Node 11356, sim 0.9935).
4. **Community Membership**: Open `outputs/community_assignments.csv` showing 272 Louvain clusters across 30,000 nodes.
