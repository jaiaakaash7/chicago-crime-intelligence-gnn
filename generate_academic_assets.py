"""
generate_academic_assets.py
Generates clean, professional, academic diagram assets for the 10-slide PowerPoint presentation.
Follows dark navy (#0B132B / #0F172A), slate (#1E293B), sky blue (#38BDF8), and orange (#FB923C) theme.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = "presentation_assets"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']

def draw_box(ax, x, y, w, h, title, subtexts, bg='#1E293B', ec='#334155', tc='#38BDF8', stc='#94A3B8', lw=1.2, r=0.1):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.03,rounding_size={r}", facecolor=bg, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    if title:
        ax.text(x + w/2, y + h - 0.22, title, ha='center', va='top', fontsize=9.5, fontweight='bold', color=tc)
    if subtexts:
        y_off = y + h - (0.48 if title else 0.25)
        for st in subtexts:
            ax.text(x + w/2, y_off, st, ha='center', va='top', fontsize=8, color=stc)
            y_off -= 0.22

def draw_arrow(ax, x1, y1, x2, y2, color='#38BDF8', lw=1.5, label=''):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", color=color, lw=lw))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label, ha='center', va='bottom', fontsize=8, fontweight='bold', color='#CBD5E1')

# ------------------------------------------------------------------------------
# 1. Slide 1: Simple Visual Pipeline Flow
# Crime Data -> Graph -> GraphSAGE -> Intelligence Dashboard
# ------------------------------------------------------------------------------
def make_slide1_flow():
    fig, ax = plt.subplots(figsize=(10, 1.8), dpi=300, facecolor='#0F172A')
    ax.set_facecolor('#0F172A')
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.8)

    steps = [
        ("Crime Data", ["30,000 Records", "40 Features"]),
        ("Graph Topology", ["Spatial-Temporal", "189,474 Edges"]),
        ("GraphSAGE Model", ["Neighborhood Aggr.", "64-D Embeddings"]),
        ("Intelligence UI", ["Interactive Gradio", "Local 127.0.0.1"])
    ]
    x_coords = [0.4, 2.9, 5.4, 7.9]
    for i, (t, subs) in enumerate(steps):
        x = x_coords[i]
        draw_box(ax, x, 0.25, 1.8, 1.3, t, subs, bg='#1E293B', ec='#38BDF8' if i==2 else '#334155', tc='#38BDF8', stc='#94A3B8')
        if i < 3:
            draw_arrow(ax, x + 1.85, 0.9, x_coords[i+1] - 0.05, 0.9, color='#FB923C', lw=2.0)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "slide1_pipeline_flow.png")
    fig.savefig(p, facecolor='#0F172A')
    plt.close()
    print("Created:", p)

# ------------------------------------------------------------------------------
# 2. Slide 2: Tabular vs Graph Analysis Comparison
# ------------------------------------------------------------------------------
def make_slide2_comparison():
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=300, facecolor='#0F172A')
    ax.set_facecolor('#0F172A')
    ax.axis('off')
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 4.2)

    # Left: Traditional Tabular Analysis
    draw_box(ax, 0.4, 0.5, 4.0, 3.4, "Traditional Tabular Analysis", [
        "Independent Table Rows (i.i.d. Assumption)",
        "-------------------------------------------",
        "• Each crime report treated in isolation",
        "• Ignores spatial continuity between events",
        "• Ignores temporal clustering & recurrence",
        "• Misses boundary patterns across districts",
        "• Limited feature interaction modeling"
    ], bg='#1E293B', ec='#EF4444', tc='#F87171', stc='#CBD5E1', lw=1.5)

    # VS badge
    ax.text(4.75, 2.2, "VS", ha='center', va='center', fontsize=12, fontweight='bold', color='#FB923C',
            bbox=dict(boxstyle="circle,pad=0.3", fc="#334155", ec="#FB923C", lw=1.5))

    # Right: Graph-Based Analysis
    draw_box(ax, 5.1, 0.5, 4.0, 3.4, "Proposed Graph-Based Analysis", [
        "Relational Incident Multiplex Graph",
        "-------------------------------------------",
        "• Crime incident modeled as a Node",
        "• Spatial-temporal proximity forms Edges",
        "• GraphSAGE aggregates neighbor context",
        "• Discovers cross-district communities",
        "• Embeddings power candidate link discovery"
    ], bg='#1E293B', ec='#38BDF8', tc='#38BDF8', stc='#CBD5E1', lw=1.5)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "slide2_tabular_vs_graph.png")
    fig.savefig(p, facecolor='#0F172A')
    plt.close()
    print("Created:", p)

# ------------------------------------------------------------------------------
# 3. Slide 4: Graph Construction Flow
# ------------------------------------------------------------------------------
def make_slide4_construction():
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=300, facecolor='#0F172A')
    ax.set_facecolor('#0F172A')
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)

    draw_box(ax, 0.5, 2.7, 2.6, 1.5, "1. Crime Incident", ["Single Reported Crime", "40 Engineered Features", "N = 30,000 Incidents"], bg='#1E293B', ec='#38BDF8', tc='#38BDF8')
    draw_box(ax, 3.7, 2.7, 2.6, 1.5, "2. Graph Node", ["Unique ID: 0 to 29,999", "Feature Vector x_i in R^40", "30 Target Crime Classes"], bg='#1E293B', ec='#34D399', tc='#34D399')
    draw_box(ax, 6.9, 2.7, 2.6, 1.5, "3. Graph Edge", ["Connection Formed if:", "• Spatial dist <= 1.5 km", "• Time delta <= 7 days", "• Same Police District"], bg='#1E293B', ec='#FB923C', tc='#FB923C')

    draw_arrow(ax, 3.15, 3.45, 3.65, 3.45, color='#38BDF8')
    draw_arrow(ax, 6.35, 3.45, 6.85, 3.45, color='#34D399')

    # Resulting topology box
    draw_box(ax, 1.0, 0.3, 8.0, 1.9, "Verified Graph Topology Statistics", [
        "30,000 Nodes  |  189,474 Directed Edges  |  94,737 Undirected Pairs",
        "Average Degree: 6.32  |  Maximum Degree: 60  |  Network Density: 2.11 x 10^-4",
        "152 Connected Components  |  5 Isolated Nodes  |  0 Self-Loops",
        "---------------------------------------------------------------------------------",
        "*Note: Graph communities represent structurally connected incident groups, not confirmed organizations."
    ], bg='#0F172A', ec='#38BDF8', tc='#F8FAFC', stc='#94A3B8', lw=1.5)

    draw_arrow(ax, 8.2, 2.65, 8.2, 2.25, color='#FB923C')

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "slide4_graph_construction.png")
    fig.savefig(p, facecolor='#0F172A')
    plt.close()
    print("Created:", p)

# ------------------------------------------------------------------------------
# 4. Slide 5: GraphSAGE Architecture vs MLP
# ------------------------------------------------------------------------------
def make_slide5_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300, facecolor='#0F172A')
    ax.set_facecolor('#0F172A')
    ax.axis('off')
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.8)

    # Top GraphSAGE Flow
    draw_box(ax, 0.4, 2.4, 1.7, 2.0, "Input", ["40 Features", "Spatial-Temporal", "Adjacency A"], bg='#1E293B', ec='#334155', tc='#F8FAFC')
    draw_box(ax, 2.5, 2.2, 2.2, 2.4, "SAGEConv Layer 1", ["Mean Aggregation", "40 -> 128 Hidden", "LayerNorm + ReLU", "Dropout (p = 0.2)", "1-Hop Neighbors"], bg='#1E293B', ec='#38BDF8', tc='#38BDF8')
    draw_box(ax, 5.1, 2.2, 2.2, 2.4, "SAGEConv Layer 2", ["Mean Aggregation", "128 -> 64 Channels", "LayerNorm", "64-D Embedding", "2-Hop Neighbors"], bg='#1E293B', ec='#34D399', tc='#34D399')
    draw_box(ax, 7.7, 2.4, 2.4, 2.0, "Classifier Head", ["Linear(64 -> 30)", "Softmax Probabilities", "30 Crime Classes"], bg='#1E293B', ec='#FB923C', tc='#FB923C')

    draw_arrow(ax, 2.15, 3.4, 2.45, 3.4)
    draw_arrow(ax, 4.75, 3.4, 5.05, 3.4)
    draw_arrow(ax, 7.35, 3.4, 7.65, 3.4)

    # Bottom Comparison with MLP
    draw_box(ax, 0.4, 0.3, 9.7, 1.5, "Comparison with Parameter-Matched Baseline MLP", [
        "Baseline MLP Architecture: Linear(40 -> 128) -> LayerNorm + ReLU + Dropout -> Linear(128 -> 64) -> Linear(64 -> 30)",
        "Key Distinction: MLP receives the identical 40 features but uses NO graph adjacency and NO neighborhood aggregation.",
        "Experimental Principle: Isolates the specific empirical advantage provided by graph relational inductive bias."
    ], bg='#0F172A', ec='#475569', tc='#F8FAFC', stc='#94A3B8', lw=1.2)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "slide5_graphsage_arch.png")
    fig.savefig(p, facecolor='#0F172A')
    plt.close()
    print("Created:", p)

# ------------------------------------------------------------------------------
# 5. Slide 6: Model Evaluation Comparison Chart
# ------------------------------------------------------------------------------
def make_slide6_evaluation():
    metrics = ["Accuracy (Top-1)", "Top-3 Accuracy", "Top-5 Accuracy", "Weighted F1", "Macro F1"]
    mlp_scores = [24.56, 45.24, 59.56, 15.66, 3.61]
    sage_scores = [23.64, 40.96, 53.27, 17.47, 5.12]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300, facecolor='#0F172A')
    ax.set_facecolor('#1E293B')

    x = np.arange(len(metrics))
    width = 0.35

    r1 = ax.bar(x - width/2, mlp_scores, width, label='Baseline MLP (No Graph)', color='#64748B', edgecolor='#334155', linewidth=0.8)
    r2 = ax.bar(x + width/2, sage_scores, width, label='GraphSAGE (GNN)', color='#38BDF8', edgecolor='#0284C7', linewidth=0.8)

    for r in r1:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#94A3B8', fontweight='bold')

    for r in r2:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#38BDF8', fontweight='bold')

    ax.set_ylabel("Score Percentage (%)", fontsize=10, fontweight='bold', color='#F8FAFC')
    ax.set_title("Experimental Evaluation: GraphSAGE vs. Baseline MLP (4,500 Test Incidents)", fontsize=11, fontweight='bold', pad=12, color='#F8FAFC')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9, fontweight='semibold', color='#CBD5E1')
    ax.legend(loc='upper left', frameon=True, facecolor='#0F172A', edgecolor='#334155', fontsize=8.5, labelcolor='#F8FAFC')
    ax.set_ylim(0, 70)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='#94A3B8')

    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors='#CBD5E1')

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "slide6_eval_chart.png")
    fig.savefig(p, facecolor='#0F172A')
    plt.close()
    print("Created:", p)

# ------------------------------------------------------------------------------
# 6. Slide 7: Similarity Diagram for Node 587
# ------------------------------------------------------------------------------
def make_slide7_similarity():
    fig, ax = plt.subplots(figsize=(9.8, 4.4), dpi=300, facecolor='#0F172A')
    ax.set_facecolor('#0F172A')
    ax.axis('off')
    ax.set_xlim(0, 9.8)
    ax.set_ylim(0, 4.4)

    # Target Node Box
    draw_box(ax, 0.4, 1.4, 3.6, 2.6, "Query Demonstration: Node 587", [
        "Case Number: HZ102757",
        "Primary Crime: ASSAULT",
        "Description: AGGRAVATED: HANDGUN",
        "Police District: 9",
        "Incident Date: 2016-01-03 15:36:00",
        "------------------------------------",
        "GraphSAGE Ranks ASSAULT as Top Pred.",
        "Learned Representation: 64-D Vector"
    ], bg='#1E293B', ec='#38BDF8', tc='#38BDF8', stc='#CBD5E1', lw=1.5)

    # Top 3 similar incidents
    similar_nodes = [
        ("1. Node 11356", "Case JA158767 | Motor Vehicle Theft", "District 8 | Cosine Sim = 0.9935", 2.9, '#34D399'),
        ("2. Node 3283", "Case JA118510 | Deceptive Practice", "District 9 | Cosine Sim = 0.9918", 1.9, '#38BDF8'),
        ("3. Node 959", "Case HZ482430 | Criminal Damage", "District 8 | Cosine Sim = 0.9914", 0.9, '#FB923C')
    ]

    for title, line1, line2, y, col in similar_nodes:
        draw_box(ax, 4.8, y, 4.6, 0.85, title, [line1, line2], bg='#1E293B', ec=col, tc=col, stc='#CBD5E1')
        draw_arrow(ax, 4.05, 2.7, 4.75, y + 0.42, color='#64748B', lw=1.2)

    # Disclaimer banner
    ax.text(4.9, 0.35, "*Note: Cosine similarity indicates mathematical closeness in the learned representation.\nIt does not establish causation, offender identity, or a confirmed relationship.",
            ha='center', va='center', fontsize=7.5, fontstyle='italic', color='#94A3B8')

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "slide7_similarity_node587.png")
    fig.savefig(p, facecolor='#0F172A')
    plt.close()
    print("Created:", p)

# ------------------------------------------------------------------------------
# 7. Slide 8: Multi-Factor Candidate Link Scoring Formula
# ------------------------------------------------------------------------------
def make_slide8_link_formula():
    fig, ax = plt.subplots(figsize=(9.8, 4.4), dpi=300, facecolor='#0F172A')
    ax.set_facecolor('#0F172A')
    ax.axis('off')
    ax.set_xlim(0, 9.8)
    ax.set_ylim(0, 4.4)

    factors = [
        ("GNN Latent Similarity", "35% Weight", "Cosine similarity between 64-D GraphSAGE embeddings", 3.4, '#38BDF8'),
        ("Spatial Proximity", "25% Weight", "Haversine distance decay within 1.5 km", 2.65, '#34D399'),
        ("Temporal Proximity", "15% Weight", "Time difference decay within 7-day window", 1.9, '#FB923C'),
        ("Crime-Type Match", "15% Weight", "Categorical match of primary crime type", 1.15, '#A78BFA'),
        ("District Match", "10% Weight", "Coincidence in police administrative district", 0.4, '#F43F5E')
    ]

    for title, weight, desc, y, col in factors:
        draw_box(ax, 0.4, y, 5.0, 0.65, f"{title} ({weight})", [desc], bg='#1E293B', ec=col, tc=col, stc='#CBD5E1')
        draw_arrow(ax, 5.45, y + 0.32, 6.35, 2.2, color='#64748B', lw=1.0)

    # Result Box
    draw_box(ax, 6.4, 1.2, 3.0, 2.0, "Candidate Link Score", [
        "Score Range: 0.0 to 1.0",
        "-----------------------",
        "71,915 Unique Pairs",
        "Evaluated in System",
        "Filters incidents for",
        "further analytical review"
    ], bg='#0F172A', ec='#38BDF8', tc='#38BDF8', stc='#F8FAFC', lw=1.5)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "slide8_link_formula.png")
    fig.savefig(p, facecolor='#0F172A')
    plt.close()
    print("Created:", p)

# ------------------------------------------------------------------------------
# 8. Slide 9: Dashboard Flow Diagram
# ------------------------------------------------------------------------------
def make_slide9_flow():
    fig, ax = plt.subplots(figsize=(10, 1.8), dpi=300, facecolor='#0F172A')
    ax.set_facecolor('#0F172A')
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.8)

    tabs = [
        ("1. Node Analysis", "Tab 2 Analyzer"),
        ("2. Embeddings", "64-D Latent Rep."),
        ("3. Similarity", "Cosine Queries"),
        ("4. Candidate Links", "71,915 Scored Pairs"),
        ("5. Network Analytics", "272 Communities")
    ]
    x_coords = [0.3, 2.3, 4.3, 6.3, 8.3]
    for i, (t, s) in enumerate(tabs):
        x = x_coords[i]
        draw_box(ax, x, 0.3, 1.5, 1.2, t, [s], bg='#1E293B', ec='#38BDF8' if i==3 else '#334155', tc='#38BDF8', stc='#CBD5E1')
        if i < 4:
            draw_arrow(ax, x + 1.55, 0.9, x_coords[i+1] - 0.05, 0.9, color='#FB923C', lw=1.5)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "slide9_dashboard_flow.png")
    fig.savefig(p, facecolor='#0F172A')
    plt.close()
    print("Created:", p)

if __name__ == "__main__":
    make_slide1_flow()
    make_slide2_comparison()
    make_slide4_construction()
    make_slide5_architecture()
    make_slide6_evaluation()
    make_slide7_similarity()
    make_slide8_link_formula()
    make_slide9_flow()
    print("All academic assets generated successfully.")
