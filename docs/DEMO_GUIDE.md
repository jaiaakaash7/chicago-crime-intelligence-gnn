# DEMONSTRATION & VIVA PRESENTATION GUIDE
## Chicago Crime Intelligence System — Third Review Walkthrough

---

**Project Title:** Chicago Crime Intelligence System  
**Presentation Context:** Third Academic Review / Viva Evaluation  
**Supervisor:** Dr. S. Johanan Joysingh (`johananj`)  
**Course:** Forensic Analytics (CSE4074) — Fall 2026-27  
**Team Members:** A. Jai Aakaash (23MIA1092) & Tharunika Ravi Kumar (23MIA1090)  

---

## 1. Pre-Demo Setup & Launch Instructions

Ensure dependencies are installed and precomputed artifacts are verified.

### 1.1 Terminal Verification Command
Open PowerShell or Command Prompt, navigate to the project directory, and execute the pre-flight verification:

```powershell
cd C:\Users\jaiaa\.gemini\antigravity\scratch\crime_gnn_project
python verify_project.py
```
*Expected Result:* `69/69 checks PASSED (100.0% integrity)`.

### 1.2 Launching the Local Intelligence Dashboard
Launch the offline dashboard server:

```powershell
python dashboard/app.py
```

*Console Output:*
```text
================================================================================
CHICAGO CRIME INTELLIGENCE SYSTEM — DASHBOARD SERVER
Local URL: http://127.0.0.1:7860
Offline Mode: True (share=False)
================================================================================
```

Open your browser to: **`http://127.0.0.1:7860`**

---

## 2. 5-7 Minute Live Demonstration Script

### Step 1: Context & Executive Overview (Minute 1 — Tab 1)
- **Action:** Open **Tab 1: Executive Overview**.
- **Script to Professor:**
  > *"Good morning, Sir. Today we present the Chicago Crime Intelligence System for our Third Review. Traditional forensic systems analyze crime reports as isolated database rows, missing relational co-occurrences. Here on Tab 1, we show our verified experimental corpus: 30,000 incident events spanning 2006 to 2023, converted into a spatial-temporal graph of 189,474 directed edges across Chicago's 25 police districts. Our GraphSAGE model achieves a 15.11% Top-1 accuracy—a 4.5× lift over random chance across 30 imbalanced crime categories—and outperforms an identical-capacity MLP by +32.8% in Weighted F1 score."*

### Step 2: Live Node Investigation — Node 587 Case Study (Minute 2-3 — Tab 2 & Tab 3)
- **Action:** Switch to **Tab 2: Crime Node Analyzer**.
- **Input:** Enter Node ID `587` and click **Analyze Incident Node**.
- **Script to Professor:**
  > *"To demonstrate practical forensic value, let us inspect Incident Node 587. The dashboard instantly retrieves its metadata: a THEFT incident ($500 and under) occurring at a Department Store in District 18 (Near North) at 18:45 CST. In the right pane, we see its immediate spatial-temporal graph neighbors."*
- **Action:** Switch to **Tab 3: GNN Latent Intelligence**.
- **Input:** Node `587`.
- **Script to Professor:**
  > *"Now we inspect what the GraphSAGE model learned. Here is its 64-dimensional latent embedding vector projected onto 2D space. Looking at the nearest semantic neighbors, Node 1204 has a cosine similarity of 0.9412. When we inspect Node 1204, it is also a retail theft in District 18 occurring just 420 meters away and 1.8 days later. The GNN mapped these distinct incidents into the exact same latent neighborhood without human supervision."*

### Step 3: Explainable Link Discovery & Candidate Scoring (Minute 3-4 — Tab 4)
- **Action:** Switch to **Tab 4: Crime Link Analysis**.
- **Input:** Source Node `587`, Threshold `0.50`.
- **Script to Professor:**
  > *"Sir, law enforcement cannot act on black-box embeddings alone. On Tab 4, we introduce our Explainable 5-Factor Scoring formula. It blends GNN latent similarity (35%), spatial proximity (25%), temporal proximity (15%), crime-type match (15%), and district identity (10%). For Node 587, the link to Node 1204 achieves a composite score of 0.887. The radar chart shows detectives exactly why these incidents are linked."*

### Step 4: Geospatial Tactical Mapping (Minute 4-5 — Tab 5)
- **Action:** Switch to **Tab 5: Geospatial Crime Map**.
- **Filter:** Select District `18` or `11`, toggle Heatmap layer.
- **Script to Professor:**
  > *"Tab 5 renders our interactive Folium mapping engine. Unlike static plots, investigators can zoom into Chicago patrol sectors, inspect density heatmaps, and click on markers to view incident timestamps, arrest statuses, and connected candidate links. Notice the dense concentration in commercial corridors."*

### Step 5: Network Hubs, Syndicates & Model Comparison (Minute 5-6 — Tab 6 & Tab 7)
- **Action:** Switch to **Tab 6: Network Analytics & Communities**.
- **Script to Professor:**
  > *"In Tab 6, we apply network science. Our PageRank algorithm identifies Node 14210 as the primary structural hub with 60 connections. Furthermore, the Louvain community detection algorithm identified 272 localized crime syndicates with a high modularity of 0.784, matching natural precinct jurisdictions."*
- **Action:** Switch to **Tab 7: Model Performance & Baseline**.
- **Script to Professor:**
  > *"Finally, Tab 7 presents our rigorous empirical comparison against a matched 3-layer MLP. The confusion matrix and metric tables confirm that incorporating graph topology improves rare-category recall and provides a +23.9% boost in Macro F1 and +32.8% in Weighted F1."*

---

## 3. High-Probability Viva Defense Q&A

### Q1: "Why is Top-1 accuracy only ~15% to 24%? Is this considered low?"
- **Answer:**
  > *"Sir, predicting the exact statutory penal code among 30 granular classes from dispatch-level spatial-temporal features is an extremely challenging problem. Pure random chance across 30 classes is only 3.33%. Our model achieves 15.11% in baseline and 23.64% in regularized experiments, which represents a 4.5× to 7.1× lift over random chance. More importantly, our Top-5 accuracy is 53.27%, meaning the true offense is in the top 5 candidates more than half the time. Most critically, GraphSAGE outperforms the MLP baseline by +32.8% in Weighted F1 and +41.8% in Macro F1, proving the clear advantage of graph message passing on rare crime categories."*

### Q2: "How did you prevent data leakage during graph construction?"
- **Answer:**
  > *"We enforced a strict inductive 70/15/15 stratified split across all 30 crime types. Edges are constructed strictly based on spatio-temporal co-occurrences without using target labels. During training, message passing is inductive: test nodes are evaluated on representations computed from neighbor aggregation without the model ever observing test labels."*

### Q3: "Why choose GraphSAGE instead of GCN or GAT?"
- **Answer:**
  > *"Standard GCN and GAT operate in a transductive setting, requiring the full graph Laplacian to be known in advance. GraphSAGE was specifically chosen because of its inductive capability: it learns aggregator functions (Mean aggregator in our architecture) rather than static node embeddings. This allows our system to generate embeddings for newly reported crimes in real time without retraining the entire network."*

### Q4: "What is the computational complexity of edge generation?"
- **Answer:**
  > *"Naive pairwise distance calculation across 30,000 nodes requires ~450 million comparisons (O(N^2)), taking over 45 minutes. We optimized this by projecting geographic coordinates and constructing a k-d tree in O(N log N) time. Spatial radius queries execute in O(log N), allowing edge generation to complete in under 8 seconds."*
