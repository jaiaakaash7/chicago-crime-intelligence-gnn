# TECHNICAL ARCHITECTURE SPECIFICATION
## Chicago Crime Intelligence System

---

**Project Title:** Chicago Crime Intelligence System  
**Component:** System Architecture & Algorithmic Blueprint  
**Authors:** A. Jai Aakaash (23MIA1092) & Tharunika Ravi Kumar (23MIA1090)  
**Supervisor:** Dr. S. Johanan Joysingh (`johananj`)  
**Course:** Forensic Analytics (CSE4074) — Fall 2026-27  

---

## 1. System Architecture Overview

The Chicago Crime Intelligence System is designed as an end-to-end relational graph intelligence pipeline. It transitions from raw spatial-temporal municipal crime records to an inductive graph representation, trains Graph Neural Network architectures, computes network centrality and modular community structures, and serves forensic analysts through an offline, privacy-preserving interactive dashboard.

```
+---------------------------------------------------------------------------------------------------+
|                                  END-TO-END PIPELINE ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------------------+
|  1. INGESTION & PREPROCESSING                                                                     |
|     * Raw Chicago Crime Portal Data (7.8M+ incidents filtered to 30,000 cohort, 2006-2023)        |
|     * Spatial coordinate bounding box validation & temporal UTC parsing                           |
|     * 40-dimensional feature engineering (cyclical time, calendar, spatial, legal, 26 env flags)  |
|                                                                                                   |
|  2. SPATIO-TEMPORAL GRAPH GENERATION                                                              |
|     * k-d Tree spatial acceleration in projected coordinate space                                 |
|     * Triple constraints: Distance <= 1.5 km, Delta t <= 7.0 days, Police District Match          |
|     * Result: 30,000 nodes, 189,474 directed edges (94,737 canonical pairs), 0 self-loops         |
|                                                                                                   |
|  3. INDUCTIVE DEEP LEARNING (GraphSAGE vs MLP)                                                    |
|     * Stratified Inductive Split: Train (20,994), Val (4,500), Test (4,500) across 30 categories  |
|     * GraphSAGE: 2-layer Mean Aggregator with LayerNorm, ReLU, Dropout (40 -> 128 -> 64 -> 30)    |
|     * MLP Baseline: Matched 3-layer feedforward network without topological message passing       |
|     * Optimization: Adam (lr=0.005, weight decay=1e-4), CrossEntropyLoss, early stopping          |
|                                                                                                   |
|  4. FORENSIC INTELLIGENCE ENGINES                                                                 |
|     * 64-D Latent Embedding Extraction (results/graphsage_embeddings.csv)                         |
|     * Explainable 5-Factor Link Scoring Engine (71,915 candidate links in results/crime_links.csv)|
|     * Network Science: PageRank (alpha=0.85), Degree centrality, Louvain Community (272 clusters) |
|                                                                                                   |
|  5. AIR-GAPPED INTERACTIVE DASHBOARD                                                              |
|     * Gradio 4.0 + Folium Geospatial Map Engine                                                   |
|     * 8 Modular Analytical Tabs, running at 127.0.0.1:7860 (share=False)                          |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Spatio-Temporal Graph Mathematical Formulation

Let $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X})$ be an undirected attributed graph where:
- $\mathcal{V} = \{v_1, v_2, \dots, v_N\}$ is the set of $N = 30,000$ crime incident nodes.
- $\mathbf{X} \in \mathbb{R}^{N \times 40}$ represents the node feature matrix.
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ is the set of physical relational edges.

### 2.1 Co-Occurrence Linking Criterion
An edge $e_{ij} = (v_i, v_j)$ exists if and only if all three spatial, temporal, and administrative constraints are simultaneously satisfied:

$$e_{ij} \in \mathcal{E} \iff \left[ \mathcal{D}_{\text{Haversine}}(v_i, v_j) \le 1.50\text{ km} \right] \land \left[ |t_i - t_j| \le 7.00\text{ days} \right] \land \left[ \text{District}(v_i) = \text{District}(v_j) \right]$$

### 2.2 Great-Circle Haversine Distance
The spatial distance $\mathcal{D}_{\text{Haversine}}$ between incident coordinates $(\phi_i, \lambda_i)$ and $(\phi_j, \lambda_j)$ is computed via the Haversine formula with mean Earth radius $R = 6,371.0\text{ km}$:

$$\Delta \phi = \phi_j - \phi_i, \quad \Delta \lambda = \lambda_j - \lambda_i$$

$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_i) \cos(\phi_j) \sin^2\left(\frac{\Delta \lambda}{2}\right)$$

$$c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1 - a}\right), \quad \mathcal{D}_{\text{Haversine}} = R \cdot c$$

### 2.3 Computational Acceleration with k-d Tree
Naive pairwise comparison of $N = 30,000$ points requires $\frac{N(N-1)}{2} = 449,985,000$ distance evaluations ($\mathcal{O}(N^2)$). To scale construction, latitude and longitude coordinates are projected into Cartesian coordinates:

$$x = R \cos(\phi) \cos(\lambda), \quad y = R \cos(\phi) \sin(\lambda), \quad z = R \sin(\phi)$$

A balanced **k-d tree** is constructed in $\mathcal{O}(N \log N)$ time. Spatial radius queries are executed for each point within radius $r = 1.5\text{ km}$ in $\mathcal{O}(\log N)$ expected time, reducing graph construction runtime from over 45 minutes to under 8 seconds.

---

## 3. Inductive GraphSAGE Deep Learning Pipeline

### 3.1 Layer-Wise Mathematical Updates
GraphSAGE generates low-dimensional vector representations through iterative local neighborhood aggregation:

**Step 1: Neighbor Aggregation**
$$\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)}$$

**Step 2: Concat & Linear Projection**
$$\mathbf{z}_v^{(k)} = \mathbf{W}^{(k)} \cdot \left[ \mathbf{h}_v^{(k-1)} \,\|\, \mathbf{h}_{\mathcal{N}(v)}^{(k)} \right]$$

**Step 3: Normalization & Activation**
$$\mathbf{h}_v^{(k)} = \text{Dropout}\left(\text{ReLU}\left(\text{LayerNorm}\left(\mathbf{z}_v^{(k)}\right)\right), p=0.3\right)$$

Where:
- $\mathbf{h}_v^{(0)} = \mathbf{x}_v \in \mathbb{R}^{40}$ (initial node feature vector).
- $\mathbf{W}^{(1)} \in \mathbb{R}^{128 \times (40 + 40)} = \mathbb{R}^{128 \times 80}$.
- $\mathbf{W}^{(2)} \in \mathbb{R}^{64 \times (128 + 128)} = \mathbb{R}^{64 \times 256}$.
- $\mathbf{z}_v \in \mathbb{R}^{64}$ represents the final inductive latent embedding.

### 3.2 Classification Head
$$\hat{\mathbf{y}}_v = \text{Softmax}\left(\mathbf{W}_{\text{cls}} \mathbf{z}_v + \mathbf{b}_{\text{cls}}\right), \quad \mathbf{W}_{\text{cls}} \in \mathbb{R}^{30 \times 64}, \, \mathbf{b}_{\text{cls}} \in \mathbb{R}^{30}$$

### 3.3 Inductive Stratified Partitioning Matrix

```
Total Graph Nodes: 30,000
├── Training Partition (Inductive Mask = Train):    20,994 nodes (70.0%)
├── Validation Partition (Inductive Mask = Val):      4,500 nodes (15.0%)
└── Test Partition (Inductive Mask = Test):           4,500 nodes (15.0%)
```

---

## 4. Multi-Factor Link Prediction Scoring Engine

### 4.1 Scoring Formula
Candidate links between incidents $v_i$ and $v_j$ are assigned an explainable composite link score $\mathcal{S}_{\text{Total}} \in [0, 1]$:

$$\mathcal{S}_{\text{Total}}(v_i, v_j) = 0.35 \cdot \mathcal{S}_{\text{GNN}} + 0.25 \cdot \mathcal{S}_{\text{Space}} + 0.15 \cdot \mathcal{S}_{\text{Time}} + 0.15 \cdot \mathcal{S}_{\text{Crime}} + 0.10 \cdot \mathcal{S}_{\text{District}}$$

### 4.2 Constituent Component Formulations

1. **GNN Latent Similarity ($w_1 = 0.35$):**
   $$\mathcal{S}_{\text{GNN}}(v_i, v_j) = \frac{1 + \frac{\mathbf{z}_i \cdot \mathbf{z}_j}{\|\mathbf{z}_i\|_2 \|\mathbf{z}_j\|_2}}{2}$$

2. **Spatial Proximity ($w_2 = 0.25$):**
   $$\mathcal{S}_{\text{Space}}(v_i, v_j) = \max\left(0, \, 1 - \frac{\mathcal{D}_{\text{Haversine}}(v_i, v_j)}{1.50\text{ km}}\right)$$

3. **Temporal Proximity ($w_3 = 0.15$):**
   $$\mathcal{S}_{\text{Time}}(v_i, v_j) = \max\left(0, \, 1 - \frac{|t_i - t_j|}{7.00\text{ days}}\right)$$

4. **Crime Type Concordance ($w_4 = 0.15$):**
   $$\mathcal{S}_{\text{Crime}}(v_i, v_j) = \begin{cases} 1.0 & \text{if } \text{Class}(v_i) = \text{Class}(v_j) \\ 0.0 & \text{otherwise} \end{cases}$$

5. **Police District Concordance ($w_5 = 0.10$):**
   $$\mathcal{S}_{\text{District}}(v_i, v_j) = \begin{cases} 1.0 & \text{if } \text{District}(v_i) = \text{District}(v_j) \\ 0.0 & \text{otherwise} \end{cases}$$

---

## 5. Network Analytics & Community Detection

### 5.1 PageRank Centrality ($\alpha = 0.85$)
Incident influence is computed via stationary distribution vectors satisfying:

$$\mathbf{p} = \left(\frac{1 - \alpha}{N}\right) \mathbf{1} + \alpha \mathbf{A}^\top \mathbf{D}^{-1} \mathbf{p}$$

Identifies structural nexus points where localized crime clusters concentrate.

### 5.2 Louvain Modularity Maximization
Partitions the graph $\mathcal{G}$ into $K = 272$ discrete communities by maximizing modularity $Q$:

$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Where $m = 94,737$ undirected edges, $k_i$ is node degree, and $\delta(c_i, c_j) = 1$ if nodes belong to the same community.

---

## 6. Software Architecture & Execution Topology

```
crime_gnn_project/
├── config/config.yaml                 <- Declarative system hyperparameters
├── data/                              <- Preprocessed clean dataset & node mapping
├── models/                            <- PyTorch checkpoints (.pth) & training histories
├── results/                           <- Serialized embeddings, metrics, and links
├── figures/                           <- 10 high-resolution publication plots
├── presentation_assets/               <- Diagrams, comparative charts & slide figures
├── dashboard/app.py                   <- Self-contained Gradio 4.0 + Folium application
├── run_pipeline.py                    <- Automated end-to-end orchestration pipeline
└── verify_project.py                  <- 69-step automated verification suite
```

The system is fully decoupled, deterministic, and executes in standard Python 3.10-3.14 environments without external cloud services.
