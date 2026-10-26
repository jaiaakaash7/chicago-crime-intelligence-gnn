import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "presentation_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Project Architecture Diagram
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#0f172a')

# Boxes
boxes = [
    ("Raw Chicago Crime Data\n(500,000 Development Slice)", 0.5, 0.92, 0.38, 0.07, '#1e293b', '#3b82f6'),
    ("Data Cleaning & Geographic Validation\n(495,969 valid records, 4,031 removed)", 0.5, 0.81, 0.44, 0.07, '#1e293b', '#38bdf8'),
    ("Stratified Sampling & Contiguous Node Mapping\n(N = 30,000 nodes, node_id: 0..29,999)", 0.5, 0.70, 0.46, 0.07, '#1e293b', '#38bdf8'),
    ("Feature Engineering (40 Dims)\n11 Temporal | 4 Spatial | 25 Contextual", 0.5, 0.59, 0.44, 0.07, '#1e293b', '#60a5fa'),
    ("Spatial-Temporal Graph Construction\nBallTree: d <= 1.5 km, dt <= 7 days, same district", 0.5, 0.48, 0.46, 0.07, '#1e293b', '#818cf8'),
    ("20-Point Strict Graph Validation\n189,474 Edges | 0 Self-Loops | Bounded IDs", 0.5, 0.37, 0.44, 0.07, '#1e293b', '#a855f7'),
    ("GraphSAGE Model (40 -> 128 -> 64 -> 30)\nNeighborhood Mean Aggregation", 0.28, 0.24, 0.38, 0.08, '#1e1b4b', '#6366f1'),
    ("MLP Baseline (40 -> 128 -> 64 -> 30)\nIndependent Node Features (No Graph)", 0.72, 0.24, 0.38, 0.08, '#3b0764', '#c084fc'),
    ("Latent 64-D Embeddings\nCosine Similarity (Node 587)", 0.28, 0.11, 0.34, 0.07, '#064e3b', '#10b981'),
    ("Candidate Crime Link Scoring\n5-Factor Formula (71,915 Links)", 0.50, 0.02, 0.36, 0.06, '#701a75', '#f43f5e'),
    ("Test Evaluation & Comparison\nAcc: 15.11% vs 12.22% (+2.89 pp)", 0.72, 0.11, 0.34, 0.07, '#831843', '#fb7185')
]

for text, x, y, w, h, bg, border in boxes:
    rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.015",
                                  facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', color='#f8fafc', fontsize=9, fontweight='bold')

# Connecting Arrows
arrows = [
    ((0.5, 0.885), (0.5, 0.845)),
    ((0.5, 0.775), (0.5, 0.735)),
    ((0.5, 0.665), (0.5, 0.625)),
    ((0.5, 0.555), (0.5, 0.515)),
    ((0.5, 0.445), (0.5, 0.405)),
    ((0.4, 0.335), (0.28, 0.28)),
    ((0.6, 0.335), (0.72, 0.28)),
    ((0.28, 0.20), (0.28, 0.145)),
    ((0.72, 0.20), (0.72, 0.145)),
    ((0.28, 0.075), (0.42, 0.035)),
    ((0.72, 0.075), (0.58, 0.035)),
]

for (x1, y1), (x2, y2) in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2, mutation_scale=15))

ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.0)
ax.axis('off')
plt.title("Chicago Crime Intelligence System — End-to-End System Architecture",
          color="#38bdf8", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "project_architecture_diagram.png"))
plt.close()

# -------------------------------------------------------------
# 2. Model Architecture Diagram
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#0f172a')

layers = [
    ("Incident Input Features\nX in R^(N x 40)\n[Temporal, Spatial, Contextual]", 0.12, 0.5, 0.20, 0.35, '#1e293b', '#3b82f6'),
    ("SAGEConv Layer 1\n40 -> 128 (aggr=mean)\nReLU + Dropout(0.3)", 0.37, 0.5, 0.18, 0.45, '#1e1b4b', '#6366f1'),
    ("SAGEConv Layer 2\n128 -> 64 (aggr=mean)\nLatent Embedding h_v in R^64", 0.62, 0.5, 0.18, 0.55, '#064e3b', '#10b981'),
    ("Classifier Head\nLinear(64 -> 30)\nSoftmax Probabilities", 0.88, 0.5, 0.18, 0.35, '#831843', '#ec4899'),
]

for text, x, y, w, h, bg, border in layers:
    rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02",
                                  facecolor=bg, edgecolor=border, linewidth=2.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', color='#f8fafc', fontsize=9, fontweight='bold')

conn_arrows = [
    ((0.22, 0.5), (0.28, 0.5)),
    ((0.46, 0.5), (0.53, 0.5)),
    ((0.71, 0.5), (0.79, 0.5)),
]
for (x1, y1), (x2, y2) in conn_arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3, mutation_scale=20))

# Branch arrow from Layer 2 to Downstream Analytics
rect_down = patches.FancyBboxPatch((0.62 - 0.18/2, 0.15 - 0.10/2), 0.18, 0.10, boxstyle="round,pad=0.015",
                                   facecolor='#1e293b', edgecolor='#f59e0b', linewidth=2)
ax.add_patch(rect_down)
ax.text(0.62, 0.15, "Downstream Tasks:\nCosine Sim (Node 587)\nCandidate Link Scoring", ha='center', va='center', color='#f8fafc', fontsize=8, fontweight='bold')
ax.annotate('', xy=(0.62, 0.20), xytext=(0.62, 0.225),
            arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=2, mutation_scale=15))

ax.set_xlim(0, 1)
ax.set_ylim(0.05, 0.95)
ax.axis('off')
plt.title("CrimeGraphSAGE Neural Network Architecture (40 -> 128 -> 64 -> 30)",
          color="#38bdf8", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "model_architecture_diagram.png"))
plt.close()

# -------------------------------------------------------------
# 3. Results Comparison Table Image
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#0f172a')
ax.axis('off')

table_data = [
    ["Metric", "GraphSAGE (GNN)", "MLP Baseline", "Difference (Delta)"],
    ["Test Accuracy", "15.11%", "12.22%", "+2.89 percentage points"],
    ["Macro F1 Score", "0.0731", "0.0590", "+0.0141 (+23.9% rel)"],
    ["Weighted F1 Score", "0.1623", "0.1222", "+0.0401 (+32.8% rel)"],
    ["Macro Precision", "0.1046", "0.1068", "-0.0022 (MLP marginally higher)"],
    ["Macro Recall", "0.1209", "0.1127", "+0.0082"],
    ["Weighted Precision", "0.3060", "0.2893", "+0.0167"],
    ["Weighted Recall", "0.1511", "0.1222", "+0.0289"]
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.7)

for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1e293b')
        cell.set_text_props(color='#38bdf8', weight='bold', size=9.5)
        cell.set_edgecolor('#475569')
    else:
        cell.set_facecolor('#0f172a' if r % 2 == 0 else '#1e293b')
        cell.set_text_props(color='#f8fafc', size=8.5)
        cell.set_edgecolor('#334155')
        if c == 1:
            cell.set_text_props(color='#60a5fa', weight='bold')
        elif c == 3:
            if "+" in table_data[r][c]:
                cell.set_text_props(color='#4ade80', weight='bold')
            else:
                cell.set_text_props(color='#f87171')

plt.title("Held-Out Test Set Performance: GraphSAGE vs. Non-Graph MLP Baseline",
          color="#38bdf8", fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "results_comparison_table.png"))
plt.close()

print("Presentation asset diagrams successfully generated in presentation_assets/")
