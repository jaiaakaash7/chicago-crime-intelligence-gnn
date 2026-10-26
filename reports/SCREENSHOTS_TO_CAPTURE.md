# Chicago Crime Intelligence System: Dashboard Screenshot Capture Guide
## Comprehensive Manual for Capturing 8 Publication-Grade Dashboard Screenshots

This guide provides exact step-by-step instructions for capturing high-resolution screenshots of all 8 interactive tabs in the Gradio dashboard (`http://127.0.0.1:7860`). These screenshots are formatted for direct inclusion into project reports, review slide decks, and academic viva presentations.

---

## 1. Capture Environment Setup

1. **Browser Recommended**: Google Chrome or Microsoft Edge on Windows 10/11.
2. **Display Resolution**: Set display scaling to 100% at $1920 \times 1080$ (or higher).
3. **Full Screen**: Press `F11` in your browser to hide address bar and tabs for clean captures.
4. **Capture Tool**: Use Windows Snipping Tool (`Win + Shift + S`) or a screenshot utility like ShareX.
5. **Output Directory**: Save captures into `reports/screenshots/` or `figures/`.

---

## 2. Tab-by-Tab Capture Checklist & Exact Inputs

### Tab 1: System Overview & Telemetry
- **Target File**: `screenshot_tab1_overview.png`
- **Navigation**: Click on the first tab labeled **"System Overview"**.
- **Visual Elements to Verify Before Capture**:
  - System telemetry banner displaying: `30,000 Nodes`, `189,474 Directed Edges`, `40 Features`, `30 Classes`.
  - Benchmark performance cards showing: `GraphSAGE: 15.11% Acc, 0.1623 W-F1` vs `MLP: 12.22% Acc, 0.1222 W-F1`.
  - System status indicator: `OPERATIONAL (CPU)`.
- **Capture Framing**: Capture the full page width including the header title and the KPI card grid.

---

### Tab 2: Crime Pattern Analysis
- **Target File**: `screenshot_tab2_patterns.png`
- **Navigation**: Click on **"Crime Pattern Analysis"**.
- **Action**: Ensure the longitudinal crime trend plot and hourly distribution charts are fully rendered.
- **Visual Elements to Verify Before Capture**:
  - Annual trend line graph showing Chicago crime volumes from 2006 to 2023.
  - Circadian hourly bar chart demonstrating the late-night crime peak between 22:00 and 02:00.
  - Pareto bar chart of top crime categories (Theft, Battery, Criminal Damage dominating).
- **Capture Framing**: Capture the central visual chart area and interactive filter dropdowns.

---

### Tab 3: Network Topology
- **Target File**: `screenshot_tab3_topology.png`
- **Navigation**: Click on **"Network Topology"**.
- **Visual Elements to Verify Before Capture**:
  - Degree distribution histogram showing average degree of 6.32 and power-law decay.
  - Sparsity metric box ($2.11 \times 10^{-4}$) and connected components count (152).
  - Graph construction parameters box (1.5 km radius, 7-day window, same district).
- **Capture Framing**: Capture the full topology statistics grid and the degree histogram.

---

### Tab 4: GraphSAGE Model Inference
- **Target File**: `screenshot_tab4_inference.png`
- **Navigation**: Click on **"Model Inference"**.
- **Input Action**:
  1. In the **Node ID** input field, type: `587`.
  2. Click the **"Run Prediction"** / **"Classify Incident"** button.
- **Visual Elements to Verify Before Capture**:
  - Incident ground truth: `ASSAULT (AGGRAVATED: HANDGUN)` | District 9.
  - Predicted category: `ASSAULT` (Rank #1 prediction with high probability).
  - Top-3 class probability distribution bar chart.
- **Capture Framing**: Capture the input card and the prediction probability output chart side-by-side.

---

### Tab 5: Latent Embedding Explorer
- **Target File**: `screenshot_tab5_embeddings.png`
- **Navigation**: Click on **"Embedding Explorer"**.
- **Input Action**:
  1. In the **Query Node ID** field, type: `587`.
  2. Click **"Explore Latent Embeddings"**.
- **Visual Elements to Verify Before Capture**:
  - 64-dimensional embedding vector preview (numerical array).
  - Cosine similarity distribution chart with prominent peak.
  - Top-5 most similar incident nodes table with similarity scores exceeding `0.99`.
- **Capture Framing**: Capture the similarity ranking table and the latent distribution plot.

---

### Tab 6: Crime Link Discovery & Cold Case Linkage
- **Target File**: `screenshot_tab6_link_discovery.png`
- **Navigation**: Click on **"Crime Link Discovery"**.
- **Input Action**:
  1. Set **Focus Node ID** to: `587`.
  2. Ensure default composite weights are set: `GNN=0.35, Spatial=0.25, Temporal=0.15, Type=0.15, District=0.10`.
  3. Click **"Discover Links"**.
- **Visual Elements to Verify Before Capture**:
  - Top Candidate Link: `Node 11356` (Motor Vehicle Theft, District 8, Cosine Sim: `0.9935`).
  - Secondary Links: `Node 3283` (sim: `0.9918`) and `Node 959` (sim: `0.9914`).
  - Link score breakdown table showing spatial, temporal, and GNN components.
- **Capture Framing**: Capture the ranked link table and scoring breakdown.

---

### Tab 7: Community Analysis & Criminal Syndicates
- **Target File**: `screenshot_tab7_communities.png`
- **Navigation**: Click on **"Community Analysis"**.
- **Input Action**:
  1. Select a high-activity community or top cluster from the dropdown.
  2. Click **"Inspect Community"**.
- **Visual Elements to Verify Before Capture**:
  - Overview banner showing: `272 Louvain Communities Detected`.
  - Geographic distribution showing cross-district membership (Districts 7, 8, 9).
  - Hub incident identifier within the selected community.
- **Capture Framing**: Capture the community summary card and cluster breakdown chart.

---

### Tab 8: Incident Deep Dive & Dossier Generator
- **Target File**: `screenshot_tab8_dossier.png`
- **Navigation**: Click on **"Incident Deep Dive"**.
- **Input Action**:
  1. In the **Target Node ID** field, type: `587`.
  2. Click **"Generate Forensic Dossier"**.
- **Visual Elements to Verify Before Capture**:
  - Case Number: `HZ102757`.
  - Primary Type: `ASSAULT`, Description: `AGGRAVATED: HANDGUN`.
  - Date & Time: `2016-01-03 15:36:00`.
  - Location Details: District 9, Ward 11/16, Community Area 61 (New City).
  - Model Verification: Predicted Class `ASSAULT`.
  - Cross-Linked Incidents: Summary of linked Node 11356 (Motor Vehicle Theft).
  - Community Cluster ID assignment.
- **Capture Framing**: Capture the full generated dossier card.

---

## 3. Image Optimization & Inclusion in Documents

- **Format**: Save as PNG with 24-bit RGB color.
- **Compression**: Do not resize below $1200\text{ px}$ width to ensure text and small charts remain sharp in printed reports.
- **Report Insertion**: Use standard Markdown syntax:
  ```markdown
  ![Tab 4: GraphSAGE Live Model Inference](screenshots/screenshot_tab4_inference.png)
  ```
