# Chicago Crime Intelligence System: Figure Verification & Source Provenance
## Complete Audit Trail of Analytical Figures, Data Sources & Report Placements

**Project Repository**: `C:\Users\jaiaa\.gemini\antigravity\scratch\crime_gnn_project`  
**Source Document**: `C:\Users\jaiaa\OneDrive\Desktop\Chicago_Crime_Intelligence_System_Report.pdf` (and `.docx`)  
**Target Report**: `Chicago_Crime_Intelligence_System_Final_With_Graphs.docx` / `.pdf`  
**Verification Standard**: Zero Fabrication, Strictly Empirical Project Outputs  

---

### Master Figure Provenance & Audit Table

| Fig # | Figure Title | Verified Data Source | Underlying File(s) Used | Report Section & Page Placement | Type / Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **Fig 1** | **Crime Incidents by Year** | `year` column in the 30,000 incident sample | `data/cleaned_crime_data.csv` (or `figures/01_crime_trend_by_year.png`) | **Section 4: Dataset** (Page 3)<br>*Placed immediately after: "The experimental dataset contains 30 crime categories..."* | Extracted & Formatted |
| **Fig 2** | **Crime Category Distribution** | All 30 primary crime categories with incident frequencies | `data/node_mapping.csv` (or `figures/02_top_crime_categories.png`) | **Section 4: Dataset** (Page 3)<br>*Placed after dataset description and before Section 5* | Extracted & Formatted (Horizontal bar chart) |
| **Fig 3** | **Crime Incidents by Hour** | Diurnal hourly distribution across 24-hour cycle | `data/cleaned_crime_data.csv` (or `figures/03_hourly_crime_distribution.png`) | **Section 5: Data Preprocessing** (Page 3)<br>*Placed after preprocessing bullet list and before Section 6* | Extracted & Formatted |
| **Fig 4** | **Crime Incidents by Police District** | Incident frequency across Chicago's 25 police districts | `data/cleaned_crime_data.csv` (or `figures/04_district_distribution.png`) | **Section 6: Feature Engineering** (Page 3)<br>*Placed immediately after feature-engineering table* | Extracted & Formatted (Horizontal bar chart) |
| **Fig 5** | **Crime Graph Construction** | Tri-criteria multiplex adjacency logic and topology stats | `results/graph_statistics.csv`, `presentation_assets/project_architecture_diagram.png` | **Section 7: Crime Graph Construction** (Page 4)<br>*Replaces simple text-only diagram* | Diagram (30k nodes, 189,474 edges, 94,737 undirected pairs) |
| **Fig 6** | **GraphSAGE Architecture** | 2-layer inductive GraphSAGE neural network structure | `presentation_assets/model_architecture_diagram.png` | **Section 8: Proposed GraphSAGE Model** (Page 4)<br>*Replaces plain text architecture* | Architecture Schematic (40 -> SAGEConv -> 128 -> LayerNorm/ReLU/Dropout -> SAGEConv -> 64 -> 30) |
| **Fig 7** | **GraphSAGE vs MLP Model Comparison** | Verified test metrics on 4,500 held-out incidents | `results/test_metrics.csv` (or `figures/06_graphsage_vs_mlp_comparison.png`) | **Section 12: Experimental Results** (Page 5)<br>*Placed immediately after experimental results table* | Benchmark Bar Chart (Acc: 15.11% vs 12.22%, W-F1: 0.1623 vs 0.1222, Macro F1: 0.0731 vs 0.0590) |
| **Fig 8** | **Crime Graph Degree Distribution** | Node degree values across all 30,000 incident nodes | `results/centrality_results.csv` (or `figures/07_network_degree_distribution.png`) | **Section 16: Network Analytics** (Page 7)<br>*Placed immediately after network statistics discussion* | Degree Histogram (Avg Degree = 6.32, Max Degree = 60) |
| **Fig 9** | **Graph Community Size Distribution** | Community size counts across all 272 Louvain clusters | `results/community_statistics.csv` (or `figures/08_community_size_chart.png`) | **Section 17: Community Detection** (Page 7)<br>*Placed immediately after: "Louvain community detection identified 272 graph communities."* | Community Distribution (Exactly 272 communities) |
| **Fig 10** | **Candidate Link Score Distribution** | Composite multi-factor scores for 71,915 candidate pairs | `results/crime_links.csv` (or `figures/09_crime_link_score_distribution.png`) | **Section 15: Candidate Crime Link Prediction** (Page 6)<br>*Placed immediately after: "The system scored 71,915 unique candidate incident pairs."* | Score Histogram (71,915 unique canonical pairs) |
| **Fig 11** | **64-Dimensional Crime Embedding Visualization** | Cosine similarity profile and latent manifold geometry | `results/graphsage_embeddings.csv` (or `figures/10_gnn_similarity_distribution.png`) | **Section 13: 64-Dimensional Embeddings** (Page 6)<br>*Placed immediately after: "The second GraphSAGE layer produces a 64-dimensional representation..."* | Embedding Geometry Visualization |
| **Fig 12** | **Top Similar Crime Incidents for Node 587** | Query Case HZ102757 and top cosine similar incident links | `results/crime_links.csv` and `data/node_mapping.csv` | **Section 14: Crime Similarity** (Page 6)<br>*Placed after Node 587 description* | Forensic Dossier Card (Node 11356: 0.9935, Node 3283: 0.9918, Node 959: 0.9914) |
| **Fig 13** | **Candidate Crime Link Scoring Framework** | 5-factor composite weighting formula flow | Mathematical formula in Section 15 | **Section 15: Candidate Crime Link Prediction** (Page 6)<br>*Placed immediately before paragraph describing 71,915 pairs* | Weighted Flow Diagram (35% GNN, 25% Spatial, 15% Temporal, 15% Type, 10% District) |
| **Fig 14** | **Crime Intelligence Dashboard** | Real hydrated screenshot of local Gradio web UI | Local server `http://127.0.0.1:7860` (`presentation_assets/final_report_figures/test_dash.png`) | **Section 18: Dashboard** (Page 7)<br>*Placed in Section 18* | Real Local Dashboard Screenshot (Executive Overview & Telemetry) |
| **Fig 15** | **Complete System Architecture** | 8-stage end-to-end data, ML, and deployment pipeline | `presentation_assets/project_architecture_diagram.png` | **Section 19: System Architecture** (Page 7)<br>*Replaces plain text pipeline* | End-to-End System Pipeline Diagram |

---

### Detailed Verification Notes by Section

#### Section 4: Dataset (Figures 1 & 2)
- **Data Source**: `data/cleaned_crime_data.csv` and `data/node_mapping.csv`.
- **Verified Values**:
  - Sample size: Exactly 30,000 records.
  - Date bounds: July 2006 to March 2023.
  - Number of classes: Exactly 30 unique primary crime categories.
  - Top categories: THEFT (6,639; 22.1%), BATTERY (5,652; 18.8%), CRIMINAL DAMAGE (3,477; 11.6%), ASSAULT (2,134; 7.1%), OTHER OFFENSE (2,070; 6.9%), DECEPTIVE PRACTICE (1,714; 5.7%), ROBBERY (1,299; 4.3%), BURGLARY (1,272; 4.2%), MOTOR VEHICLE THEFT (1,234; 4.1%), NARCOTICS (1,154; 3.8%).

#### Section 5 & 6: Preprocessing & Feature Engineering (Figures 3 & 4)
- **Data Source**: `data/cleaned_crime_data.csv`.
- **Verified Values**:
  - 24 hourly bins (peak at 22:00–02:00).
  - 40 features: 11 temporal, 4 spatial, 5 contextual, 20 location indicators.
  - Police districts: Districts 1 through 25.

#### Section 7 & 8: Graph Construction & GraphSAGE (Figures 5 & 6)
- **Data Source**: `results/graph_statistics.csv` and model specifications.
- **Verified Values**:
  - 30,000 nodes, 189,474 directed edges, 94,737 canonical undirected pairs.
  - Density: $2.1053 \times 10^{-4}$. Average degree: 6.32, Maximum degree: 60.
  - Connected components: 152, Isolated nodes: 5, Self-loops: 0.
  - GraphSAGE architecture: 40 -> 128 (mean aggr, LayerNorm, ReLU, Dropout 0.2) -> 64 (mean aggr, LayerNorm) -> Linear 30.

#### Section 12: Experimental Results (Figure 7)
- **Data Source**: `results/test_metrics.csv`.
- **Verified Values (4,500 Test Nodes)**:
  - Accuracy: GraphSAGE = 15.11%, Baseline MLP = 12.22% (Delta: +2.89 percentage points, +23.6% relative lift).
  - Weighted F1: GraphSAGE = 0.1623, Baseline MLP = 0.1222 (Delta: +0.0401, +32.8% relative lift).
  - Macro F1: GraphSAGE = 0.0731, Baseline MLP = 0.0590 (Delta: +0.0141, +23.9% relative lift).
  - Weighted Precision: GraphSAGE = 0.3060, Baseline MLP = 0.2893 (Delta: +0.0167, +5.8% relative lift).
  - Weighted Recall: GraphSAGE = 0.1511, Baseline MLP = 0.1222 (Delta: +0.0289, +23.6% relative lift).
  - Macro Recall: GraphSAGE = 0.1209, Baseline MLP = 0.1127 (Delta: +0.0082, +7.3% relative lift).
  - Macro Precision: Baseline MLP = 0.1068, GraphSAGE = 0.1046 (Delta: -0.0022; MLP conservative).

#### Section 13, 14 & 15: Embeddings, Similarity & Link Prediction (Figures 10, 11, 12, 13)
- **Data Source**: `results/graphsage_embeddings.csv` and `results/crime_links.csv`.
- **Verified Values**:
  - Latent dimension: 64 dimensions, $L_2$-normalized.
  - Candidate crime links: Exactly 71,915 unique canonical incident pairs.
  - Node 587 Case Study:
    - Case `HZ102757` | `ASSAULT` (`AGGRAVATED: HANDGUN`) | District 9 | `2016-01-03 15:36:00` | Prediction: `ASSAULT`.
    - #1 Link: Node 11356 (Case `JA158767`, Motor Vehicle Theft, District 8, Cosine Similarity: 0.9935).
    - #2 Link: Node 3283 (Case `JA118510`, Deceptive Practice, District 9, Cosine Similarity: 0.9918).
    - #3 Link: Node 959 (Case `HZ482430`, Criminal Damage, District 8, Cosine Similarity: 0.9914).
  - Multi-Factor Formula: $0.35 \cdot S_{\text{GNN}} + 0.25 \cdot S_{\text{Spatial}} + 0.15 \cdot S_{\text{Temporal}} + 0.15 \cdot S_{\text{Type}} + 0.10 \cdot S_{\text{District}}$.

#### Section 16 & 17: Network Analytics & Community Detection (Figures 8 & 9)
- **Data Source**: `results/centrality_results.csv` and `results/community_statistics.csv`.
- **Verified Values**:
  - Average node degree: 6.32, Maximum degree: 60.
  - Connected components: 152.
  - Discovered Louvain communities: Exactly 272 communities.

#### Section 18 & 19: Dashboard & Complete System Architecture (Figures 14 & 15)
- **Data Source**: Local Gradio application at `http://127.0.0.1:7860` (`share=False`).
- **Verified Values**:
  - 8 tabs: 1. Executive Overview, 2. Crime Node Analyzer, 3. GNN Intelligence, 4. Crime Link Analysis, 5. Crime Map, 6. Network Analytics, 7. Model Performance, 8. About & Methodology.
  - Complete end-to-end pipeline: Ingestion -> Bounding -> 30k Sampling -> 40 Features -> Graph -> GraphSAGE vs MLP -> Embeddings -> Links -> Communities -> Gradio.
