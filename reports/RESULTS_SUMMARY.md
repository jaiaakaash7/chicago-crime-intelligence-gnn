# Experimental Results Summary: Chicago Crime Intelligence System

**Execution Date**: 2026-09-21 22:40:45  
**Model Architecture**: GraphSAGE (40 $\to$ 128 $\to$ 64 $\to$ 31) vs. MLP Baseline (40 $\to$ 128 $\to$ 64 $\to$ 31)  
**Dataset**: Chicago Crimes (2001–2023 Sample, $N=30,000$)  

---

## 1. Primary Model Performance Comparison (Test Split)

| Evaluation Metric | GraphSAGE (GNN) | MLP Baseline | Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **0.2364** | 0.2456 | -0.0091 |
| **Macro F1** | **0.0512** | 0.0361 | +0.0151 |
| **Weighted F1** | **0.1747** | 0.1566 | +0.0181 |
| **Macro Precision** | 0.0680 | 0.0390 | +0.0290 |
| **Macro Recall** | 0.0722 | 0.0503 | +0.0219 |
| **Weighted Precision** | 0.1996 | 0.1401 | +0.0595 |
| **Weighted Recall** | 0.2364 | 0.2456 | -0.0091 |

---

## 2. Graph & Analytics Summary

- **Total Incident Nodes ($N$)**: 30,000
- **Total Directed Graph Edges ($E$)**: 189,474 (94,737 undirected)
- **Graph Density**: 2.105337e-04
- **Average Degree**: 6.32
- **Maximum Degree**: 60
- **Connected Components**: 152
- **Number of Communities Detected**: 272
- **Candidate Crime Links Generated**: 71,915
- **Feature Dimension**: 40
- **Embedding Dimension**: 64
- **Target Classes**: 30

*Note: All values above are generated directly from execution.*
