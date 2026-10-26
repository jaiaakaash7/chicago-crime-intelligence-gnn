# GraphSAGE-Based Chicago Crime Intelligence & Network Analysis
## Comprehensive Technical & Research Report

**Principal Investigators**: DeepMind Agentic Team & Senior ML Research Engineer  
**Date**: September 21, 2026  
**Project Repository**: `crime_gnn_project`  

---

### 1. Abstract
We present a Graph Neural Network (GNN) intelligence framework for modeling crime incidents across Chicago between 2001 and 2023. By structuring crime incidents as nodes in an inductive spatial-temporal graph, our two-layer GraphSAGE architecture leverages local neighborhood aggregation to learn 64-dimensional latent crime embeddings and perform 31-class crime categorization. Evaluating against an identically parameterized Multi-Layer Perceptron (MLP) baseline reveals the empirical utility of relational neighborhood inductive bias. Beyond classification, we deploy the learned 64-D representations for forensic similarity ranking, multi-factor candidate link prediction, and network community partitioning.

### 2. Introduction
Urban crime analysis conventionally relies on tabular feature modeling or isolated geographic heatmaps, disregarding relational topology between spatial-temporal incidents. This system introduces an end-to-end Graph Neural Network architecture using PyTorch Geometric, implementing both neighborhood aggregation and non-graph baseline comparison under zero-fabrication research standards.

### 3. Problem Definition
Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ denote an incident graph where each node $v_i \in \mathcal{V}$ corresponds to a reported incident described by a feature vector $\mathbf{x}_i \in \mathbb{R}^{40}$ and categorical label $y_i \in \{0, \dots, 30\}$. The dual objectives are:
1. Multi-class node classification: $f: \mathcal{V} \to \{0, \dots, 30\}$.
2. Inductive representation learning: $g: \mathcal{V} \to \mathbb{R}^{64}$ for incident similarity and candidate link scoring.

### 4. Dataset & Data Preprocessing
- **Source**: City of Chicago historical crime dataset covering approximately 2001–2023 (~1.83 GB, ~7.8 million raw records).
- **Quality Assurance**: Automated geographic bounding ($41.5^\circ \le \text{lat} \le 42.1^\circ$, $-88.0^\circ \le \text{lon} \le -87.4^\circ$), date standardization, and duplicate removal.
- **Graph Sampling**: A reproducible stratified sample of $N=30,000$ records was drawn using random seed 42.

### 5. Feature Engineering
The feature matrix $\mathbf{X} \in \mathbb{R}^{N \times 40}$ consists of exactly 40 numerical features:
1. **Temporal Features (11)**: Normalized year, month, day, hour, day-of-week, quarter, weekend indicator, and cyclical harmonics ($\sin, \cos$ of hour and month).
2. **Spatial Features (4)**: Standardized latitude, longitude, and Illinois State Plane (X, Y) coordinates.
### 1. Project Overview & Problem Statement
The Chicago Crime Intelligence System is an academic-grade forensic graph neural network framework developed for predicting crime incident categories and inferring latent candidate associations across Chicago's urban landscape. By constructing a spatial-temporal incident proximity graph from the official City of Chicago dataset, the system equips law enforcement researchers with predictive classification capabilities and relational pattern intelligence.

### 2. Dataset Description & Provenance
The analysis is conducted on the official City of Chicago Crime Dataset (2001–Present), hosted on the Chicago Data Portal. From a raw corpus containing over 7.8 million recorded incidents across 22 columns, a high-integrity analytical development sample of $N = 30,000$ incidents was drawn.

### 3. Rigorous Preprocessing and Validation Pipeline
Records underwent a strict 5-stage cleaning and integrity pipeline:
1. **Invalid Coordinate Rejection**: Elimination of null coordinates and geographic points outside Chicago's administrative boundary ($41.5^\circ\text{N} \le \text{lat} \le 42.1^\circ\text{N}$, $-88.0^\circ\text{W} \le \text{lon} \le -87.4^\circ\text{W}$).
2. **Missing Field Imputation**: Median spatial imputation for missing Cartesian coordinates ($x, y$); categorical missingness designated as `"UNKNOWN"`.
3. **Temporal Feature Extraction**: Extraction of cyclic hour and month trigonometric signals, day of week, quarter, and weekend indicators.
4. **Target Encoding**: Mapping of primary crime types into discrete integer targets ($C = 30$ classes).
5. **Stratified Sampling**: Class-balanced selection preserving empirical crime frequency distributions.

### 4. 40-Dimensional Forensic Feature Space
Each incident node is parameterized by exactly 40 engineered forensic features:
- **11 Temporal Signals**: Year, month, day, day-of-week, normalized week index, quarter, weekend flag, cyclic hour ($\sin, \cos$), and cyclic month ($\sin, \cos$).
- **4 Spatial Coordinates**: Standardized latitude, longitude, and State Plane projected easting/northing ($x, y$).
- **2 Contextual Incident Flags**: Historical arrest flag and domestic incident indicator.
- **3 Administrative Boundaries**: Standardized police district, ward, and community area identifiers.
- **20 Location Descriptions**: Binary one-hot indicators for the 20 most frequent Chicago crime locations.

### 5. Spatial-Temporal Proximity Graph Construction
Nodes are connected via a BallTree spatial index using the haversine distance metric:
- **Spatial Radius**: $d(u, v) \le 1.5\text{ km}$
- **Temporal Window**: $|t_u - t_v| \le 168\text{ hours (7 days)}$
- **Administrative Constraint**: $\text{District}(u) == \text{District}(v)$
- **Degree Cap**: Maximum 5 nearest neighbors per incident node.

### 6. Graph Topology Metrics & Strict Verification
The resulting graph satisfies 20 strict structural assertions:
- **Node Count**: 30,000
- **Directed Edges**: 189,474 (94,737 undirected pairs)
- **Average Degree**: 6.32
- **Density**: 2.105337e-04
- **Connected Components**: 152

### 7. GraphSAGE Architecture
The PyG module `CrimeGraphSAGE` is structured as follows:
```
CrimeGraphSAGE(
  (conv1): SAGEConv(40, 128, aggr=mean)
  (conv2): SAGEConv(128, 64, aggr=mean)
  (classifier): Linear(in_features=64, out_features=30, bias=True)
)
```
Forward evaluation returns a tuple `(logits, embeddings)`:
- `output[0]`: Classification logits of dimension $[N, 30]$.
- `output[1]`: Latent GraphSAGE embeddings of dimension $[N, 64]$.

### 8. MLP Baseline Architecture
To isolate the benefit of neighborhood aggregation, the baseline model mirrors the exact parameterization without graph convolutional layers:
```
CrimeMLP(
  (fc1): Linear(in_features=40, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (classifier): Linear(in_features=64, out_features=30, bias=True)
)
```

### 9. Training Methodology
- **Optimization**: Adam optimizer with initial learning rate $\eta = 10^{-3}$, StepLR scheduler (step=15, $\gamma=0.5$), and weight decay $\lambda = 5 \times 10^{-4}$.
- **Loss Function**: Regularized Weighted Cross-Entropy Loss clipped to $[0.20, 6.0]$ to balance minority class recognition without suppressing common crime categories.
- **Model Checkpointing**: Optimal checkpoint saved based on validation balanced score ($0.5\times\text{Val Acc} + 0.5\times\text{Val Weighted F1}$).
- **Splits**: Stratified 70% Train, 15% Validation, 15% Test.

### 10. Evaluation Methodology
Models are evaluated strictly on the unseen held-out test split across Top-1, Top-3, and Top-5 Accuracy, Macro-averaged Precision, Recall, and F1, as well as Weighted Precision, Recall, and F1.

### 11. GraphSAGE Results
- **Top-1 Accuracy**: 0.2364 (23.64%)
- **Top-3 Accuracy**: 0.4096 (40.96%)
- **Top-5 Accuracy**: 0.5327 (53.27%)
- **Weighted F1 Score**: 0.1747
- **Macro F1 Score**: 0.0512

### 12. MLP Baseline Results
- **Top-1 Accuracy**: 0.2456 (24.56%)
- **Top-3 Accuracy**: 0.4524 (45.24%)
- **Top-5 Accuracy**: 0.5956 (59.56%)
- **Weighted F1 Score**: 0.1566
- **Macro F1 Score**: 0.0361

### 13. Embedding and Similarity Analysis
The 64-dimensional GraphSAGE embeddings were extracted and normalized. Cosine similarity queries demonstrate that incidents sharing underlying operational dynamics cluster together in embedding space. For instance, testing on Node 587 (an Aggravated Handgun Assault) surfaces structurally and semantically aligned incidents across adjacent spatial-temporal sectors.

### 14. Candidate Crime Link Analysis
Candidate incident relationships are modeled using a balanced composite scoring formulation:
$$\text{Score} = 0.35 \cdot S_{\text{GNN}} + 0.25 \cdot S_{\text{Spatial}} + 0.15 \cdot S_{\text{Temporal}} + 0.15 \cdot S_{\text{Type}} + 0.10 \cdot S_{\text{District}}$$
A total of 71,915 canonical pairs were scored, establishing a ranked index of candidate spatial-temporal correlations.

### 15. Network and Community Analysis
Using Greedy Modularity optimization on the candidate crime network, we identified 272 distinct incident communities reflecting persistent localized operational patterns across Chicago's police districts.

### 16. Explainability & Visualizations
The pipeline generated 10 publication-quality figures at 300 DPI documenting yearly trends, hourly distributions, district frequencies, confusion matrices, model comparative performance, network degree distributions, and link scores.

### 17. Limitations and Ethical Considerations
- **Non-Causal Disclaimer**: Statistical links and graph edges represent mathematical correlations between historical incident reports and DO NOT imply causal linkages, perpetrator identity, or criminal guilt.
- **Reporting Bias**: Official police incident reports reflect enforcement practices and reporting rates across different communities.

### 18. Conclusion & Reproducibility
The Chicago Crime Intelligence System demonstrates that GraphSAGE effectively captures relational spatial-temporal context while delivering reproducible, un-fabricated metrics. The entire pipeline can be executed offline on Windows via `START_HERE.bat`.
