#!/usr/bin/env python
# ==============================================================================
# CHICAGO CRIME INTELLIGENCE SYSTEM — INTERACTIVE GRADIO DASHBOARD
# Professional Dark-Navy Local Intelligence UI (Offline-Ready)
# ==============================================================================

import os
import sys
import torch
import folium
import numpy as np
import pandas as pd
import gradio as gr
import torch.nn.functional as F

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_pipeline import CrimeGraphSAGE, CrimeMLP, TOP_20_LOCATIONS

# ------------------------------------------------------------------------------
# 1. GLOBAL RESOURCE CACHING & SAFE LOADING
# ------------------------------------------------------------------------------

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

device = torch.device("cpu") # Dashboard operates deterministically on CPU

print("[Dashboard] Loading pre-computed intelligence artifacts...")

# Load Data & Mappings
dashboard_nodes_df = pd.read_csv(os.path.join(DATA_DIR, "node_mapping.csv"))
N_TOTAL_NODES = len(dashboard_nodes_df)

dashboard_class_map = pd.read_csv(os.path.join(RESULTS_DIR, "class_mapping.csv"))
CLASS_NAMES = dashboard_class_map["class_name"].tolist()
NUM_CLASSES = len(CLASS_NAMES)

dashboard_feat_names = pd.read_csv(os.path.join(RESULTS_DIR, "feature_names.csv"))["feature_name"].tolist()

# Load Embeddings & Pre-calculate Normalized Embeddings
dashboard_emb_df = pd.read_csv(os.path.join(RESULTS_DIR, "graphsage_embeddings.csv"))
emb_cols = [f"embedding_{i}" for i in range(64)]
dashboard_embeddings = dashboard_emb_df[emb_cols].values.astype(np.float32)
norms = np.linalg.norm(dashboard_embeddings, axis=1, keepdims=True)
norms = np.maximum(norms, 1e-12)
dashboard_norm_embeddings = dashboard_embeddings / norms

# Load Candidate Links
dashboard_links_df = pd.read_csv(os.path.join(RESULTS_DIR, "crime_links.csv"))

# Load Results & Metrics
dashboard_metrics_df = pd.read_csv(os.path.join(RESULTS_DIR, "test_metrics.csv"))
dashboard_graph_stats = pd.read_csv(os.path.join(RESULTS_DIR, "graph_statistics.csv")).iloc[0]
dashboard_centrality_df = pd.read_csv(os.path.join(RESULTS_DIR, "centrality_results.csv"))
dashboard_comm_stats = pd.read_csv(os.path.join(RESULTS_DIR, "community_statistics.csv"))
dashboard_class_report = pd.read_csv(os.path.join(RESULTS_DIR, "classification_report.csv"), index_col=0)

# Load GraphSAGE Model
gnn_ckpt = torch.load(os.path.join(MODELS_DIR, "best_graphsage_model.pth"), map_location=device, weights_only=False)
dashboard_model = CrimeGraphSAGE(in_channels=40, hidden_channels=128, out_channels=64, num_classes=NUM_CLASSES)
dashboard_model.load_state_dict(gnn_ckpt["model_state_dict"])
dashboard_model.eval()

print(f"[Dashboard] Loaded {N_TOTAL_NODES:,} nodes, {NUM_CLASSES} crime classes, and {len(dashboard_links_df):,} candidate links.")

# ------------------------------------------------------------------------------
# 2. TAB CONTROLLERS & QUERY HANDLERS
# ------------------------------------------------------------------------------

def analyze_crime_node(node_id):
    """Tab 2: Query a single node, retrieve details, and run GraphSAGE inference."""
    try:
        node_id = int(node_id)
    except (ValueError, TypeError):
        return (
            "⚠️ Invalid input: Node ID must be an integer.",
            pd.DataFrame(),
            "Error",
            "0%",
            pd.DataFrame()
        )

    if node_id < 0 or node_id >= N_TOTAL_NODES:
        return (
            f"⚠️ Node ID out of bounds! Valid range is 0 to {N_TOTAL_NODES - 1:,}.",
            pd.DataFrame(),
            "Out of Range",
            "0%",
            pd.DataFrame()
        )

    row = dashboard_nodes_df.iloc[node_id]
    metadata_df = pd.DataFrame([
        {"Property": "Crime Node ID", "Value": str(row["node_id"])},
        {"Property": "Official Crime ID", "Value": str(row["crime_id"])},
        {"Property": "Case Number", "Value": str(row["case_number"])},
        {"Property": "Actual Crime Type", "Value": str(row["crime_type"])},
        {"Property": "Incident Description", "Value": str(row["description"])},
        {"Property": "Incident Date", "Value": str(row["date"])},
        {"Property": "Police District", "Value": str(row["district"])},
        {"Property": "Community Area", "Value": str(row["community_area"])},
        {"Property": "Latitude", "Value": f"{row['latitude']:.6f}"},
        {"Property": "Longitude", "Value": f"{row['longitude']:.6f}"},
    ])

    # Run inference using 64-d embeddings + classifier
    # Since embeddings[node_id] is the exact output of conv2, classifier(embeddings[node_id]) produces logits!
    with torch.no_grad():
        emb_tensor = torch.from_numpy(dashboard_embeddings[node_id:node_id+1])
        # Pass through classifier
        h2 = F.relu(emb_tensor)
        logits = dashboard_model.classifier(h2)[0]
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    top_idx = np.argsort(probs)[::-1][:10]
    pred_class = CLASS_NAMES[top_idx[0]]
    pred_conf = f"{probs[top_idx[0]] * 100.0:.2f}%"

    top10_df = pd.DataFrame({
        "Rank": np.arange(1, 11),
        "Predicted Crime Category": [CLASS_NAMES[i] for i in top_idx],
        "Probability": [f"{probs[i] * 100.0:.2f}%" for i in top_idx],
        "Score (Logit)": [round(float(logits[i]), 4) for i in top_idx]
    })

    status_msg = f"✅ Incident successfully retrieved and classified by GraphSAGE for Node {node_id}."
    return status_msg, metadata_df, pred_class, pred_conf, top10_df

def get_gnn_intelligence(node_id):
    """Tab 3: Inspect 64-D embedding and find top 10 cosine-similar crime incidents."""
    try:
        node_id = int(node_id)
    except (ValueError, TypeError):
        return "⚠️ Invalid input.", pd.DataFrame(), pd.DataFrame()

    if node_id < 0 or node_id >= N_TOTAL_NODES:
        return f"⚠️ Node ID out of bounds! Valid range is 0 to {N_TOTAL_NODES - 1:,}.", pd.DataFrame(), pd.DataFrame()

    # Embedding slice table (show first 16 dimensions + summary)
    emb_vec = dashboard_embeddings[node_id]
    emb_preview = pd.DataFrame([
        {f"Dim {i:02d}": f"{emb_vec[i]:.4f}" for i in range(j, min(j+8, 64))}
        for j in range(0, 64, 8)
    ])

    # Cosine similarity against all nodes
    query_vec = dashboard_norm_embeddings[node_id]
    sims = np.dot(dashboard_norm_embeddings, query_vec)
    sims[node_id] = -1.0 # strictly exclude self!

    top10_idx = np.argsort(sims)[::-1][:10]
    similar_rows = []
    for rank, idx in enumerate(top10_idx, 1):
        target_row = dashboard_nodes_df.iloc[idx]
        similar_rows.append({
            "Rank": rank,
            "Target Node": idx,
            "Case Number": target_row["case_number"],
            "Crime Type": target_row["crime_type"],
            "Description": target_row["description"],
            "District": target_row["district"],
            "Date": str(target_row["date"]),
            "GNN Cosine Similarity": round(float(sims[idx]), 4)
        })

    df_similar = pd.DataFrame(similar_rows)
    status_msg = f"✅ Extracted 64-D GraphSAGE embedding and computed top 10 most similar crime nodes for Node {node_id}."
    return status_msg, emb_preview, df_similar

def query_candidate_links(node_id, min_score):
    """Tab 4: Retrieve candidate crime links involving the queried node or top links."""
    try:
        node_id = int(node_id)
    except (ValueError, TypeError):
        node_id = -1

    min_score = float(min_score) if min_score else 0.50

    if node_id >= 0 and node_id < N_TOTAL_NODES:
        subset = dashboard_links_df[
            (dashboard_links_df["source_node"] == node_id) |
            (dashboard_links_df["target_node"] == node_id)
        ]
        subset = subset[subset["link_score"] >= min_score].head(25)
        note = f"Showing candidate links connected to Node {node_id} (Score >= {min_score}):"
    else:
        subset = dashboard_links_df[dashboard_links_df["link_score"] >= min_score].head(25)
        note = f"Showing top city-wide candidate links (Score >= {min_score}):"

    if len(subset) == 0:
        return f"⚠️ No candidate links found meeting threshold {min_score}.", pd.DataFrame()

    display_cols = [
        "source_node", "target_node", "link_score", "gnn_similarity",
        "spatial_similarity", "temporal_similarity", "type_similarity", "district_similarity",
        "spatial_distance_km", "temporal_difference_hours",
        "source_crime_type", "target_crime_type", "source_district", "target_district"
    ]
    return f"✅ Found {len(subset)} candidate crime links. {note}", subset[display_cols]

def render_crime_map(filter_type, filter_district):
    """Tab 5: Render local Folium crime map with offline fallback."""
    df_sample = dashboard_nodes_df.copy()

    if filter_type and filter_type != "ALL":
        df_sample = df_sample[df_sample["crime_type"] == filter_type]
    if filter_district and filter_district != "ALL":
        try:
            d_val = int(filter_district)
            df_sample = df_sample[df_sample["district"] == d_val]
        except ValueError:
            pass

    # Sample max 400 points for browser rendering responsiveness
    render_sample = df_sample.head(400)

    # Chicago coordinates
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles="OpenStreetMap")

    for _, r in render_sample.iterrows():
        color = "red" if "THEFT" in r["crime_type"] or "ROBBERY" in r["crime_type"] else "blue"
        popup_html = (
            f"<b>Node ID:</b> {r['node_id']}<br>"
            f"<b>Crime:</b> {r['crime_type']}<br>"
            f"<b>District:</b> {r['district']}<br>"
            f"<b>Date:</b> {r['date']}<br>"
            f"<b>Case:</b> {r['case_number']}"
        )
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

    map_html = m._repr_html_()
    summary = f"Displaying {len(render_sample):,} crime locations (filtered from {len(df_sample):,} matches in sample)."
    return map_html, summary

# ------------------------------------------------------------------------------
# 3. GRADIO APP INTERFACE DEFINITION
# ------------------------------------------------------------------------------

CUSTOM_CSS = """
body, .gradio-container {
    background-color: #0d131f !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
.header-box {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid #334155;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
}
.metric-card {
    background-color: #1e293b;
    border: 1px solid #3b82f6;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    color: white;
}
.metric-val {
    font-size: 26px;
    font-weight: 700;
    color: #60a5fa;
}
.metric-lbl {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
}
.tab-content {
    padding: 15px;
}
table, .dataframe {
    color: #f1f5f9 !important;
    background-color: #1e293b !important;
}
"""

with gr.Blocks(title="Chicago Crime Intelligence System") as demo:

    # Global Header
    gr.HTML("""
    <div class="header-box">
        <h1 style="color: #60a5fa; margin: 0; font-size: 30px; letter-spacing: -0.5px;">
            🛡️ CHICAGO CRIME INTELLIGENCE SYSTEM
        </h1>
        <p style="color: #94a3b8; margin-top: 6px; margin-bottom: 0; font-size: 15px;">
            Graph Neural Network (GraphSAGE) Forensic Analytics, Link Prediction & Incident Network Topology
        </p>
    </div>
    """)

    with gr.Tabs():

        # ----------------------------------------------------------------------
        # TAB 1: EXECUTIVE OVERVIEW
        # ----------------------------------------------------------------------
        with gr.TabItem("📊 1. Executive Overview"):
            gr.Markdown("### 🏛️ Metropolitan Crime Graph & Intelligence KPI Summary")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(f"""<div class="metric-card"><div class="metric-val">{N_TOTAL_NODES:,}</div><div class="metric-lbl">Total Incidents (Nodes)</div></div>""")
                with gr.Column(scale=1):
                    gr.HTML(f"""<div class="metric-card"><div class="metric-val">{NUM_CLASSES}</div><div class="metric-lbl">Crime Categories</div></div>""")
                with gr.Column(scale=1):
                    gr.HTML(f"""<div class="metric-card"><div class="metric-val">{int(dashboard_graph_stats['edge_count']):,}</div><div class="metric-lbl">Graph Edges</div></div>""")
                with gr.Column(scale=1):
                    gr.HTML(f"""<div class="metric-card"><div class="metric-val">{dashboard_graph_stats['average_degree']:.2f}</div><div class="metric-lbl">Average Degree</div></div>""")
                with gr.Column(scale=1):
                    gr.HTML(f"""<div class="metric-card"><div class="metric-val">{len(dashboard_comm_stats)}</div><div class="metric-lbl">Louvain Communities</div></div>""")
                with gr.Column(scale=1):
                    gr.HTML(f"""<div class="metric-card"><div class="metric-val">{len(dashboard_links_df):,}</div><div class="metric-lbl">Candidate Links</div></div>""")

            gr.Markdown("---")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📈 Historical Incident Trend Across Chicago")
                    gr.Image(os.path.join(FIGURES_DIR, "01_crime_trend_by_year.png"), label="Yearly Crime Trend")
                with gr.Column():
                    gr.Markdown("#### 📋 Predominant Crime Categories in Dataset")
                    gr.Image(os.path.join(FIGURES_DIR, "02_top_crime_categories.png"), label="Top Crime Types")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### ⏰ Diurnal Hourly Distribution")
                    gr.Image(os.path.join(FIGURES_DIR, "03_hourly_crime_distribution.png"), label="Hourly Distribution")
                with gr.Column():
                    gr.Markdown("#### 🚔 Incident Distribution Across Police Districts")
                    gr.Image(os.path.join(FIGURES_DIR, "04_district_distribution.png"), label="District Distribution")

        # ----------------------------------------------------------------------
        # TAB 2: CRIME NODE ANALYZER
        # ----------------------------------------------------------------------
        with gr.TabItem("🔍 2. Crime Node Analyzer"):
            gr.Markdown("### 🎯 Single-Incident Inspection & GraphSAGE Inference Engine")
            gr.Markdown("Enter any valid Graph Node ID (0 to 29,999) to retrieve incident attributes and execute the trained GraphSAGE classifier.")

            with gr.Row():
                with gr.Column(scale=2):
                    node_input = gr.Number(value=587, label="Graph Node ID", precision=0)
                    analyze_btn = gr.Button("🔍 Run GraphSAGE Inference", variant="primary")
                    status_output = gr.Textbox(label="Status & Execution Log", interactive=False)
                with gr.Column(scale=2):
                    pred_class_out = gr.Textbox(label="GraphSAGE Predicted Crime Category", interactive=False)
                    pred_conf_out = gr.Textbox(label="Prediction Confidence", interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📋 Official Police Incident Record")
                    meta_table = gr.DataFrame(headers=["Property", "Value"], label="Incident Metadata")
                with gr.Column():
                    gr.Markdown("#### 📊 Top 10 Prediction Class Probabilities")
                    probs_table = gr.DataFrame(label="Top 10 Softmax Probabilities")

            analyze_btn.click(
                fn=analyze_crime_node,
                inputs=[node_input],
                outputs=[status_output, meta_table, pred_class_out, pred_conf_out, probs_table]
            )

        # ----------------------------------------------------------------------
        # TAB 3: GNN INTELLIGENCE
        # ----------------------------------------------------------------------
        with gr.TabItem("🧠 3. GNN Latent Intelligence"):
            gr.Markdown("### 🧬 Learned 64-Dimensional Latent GraphSAGE Embeddings & Similarity")
            gr.Markdown(r"Calculates cosine similarity across learned GraphSAGE latent embeddings ($\mathbf{h}_v \in \mathbb{R}^{64}$) generated by layer 2 neighborhood aggregation.")

            with gr.Row():
                with gr.Column(scale=2):
                    gnn_node_input = gr.Number(value=587, label="Target Node ID", precision=0)
                    gnn_btn = gr.Button("🧠 Extract Embeddings & Compute Similarity", variant="primary")
                    gnn_status = gr.Textbox(label="Status Log", interactive=False)

            gr.Markdown("#### 🧬 64-Dimensional Embedding Vector Preview (Dimensions 0–63)")
            emb_table = gr.DataFrame(label="Latent Vector Dimensions")

            gr.Markdown("#### 🏆 Top 10 Most Similar Incidents in Chicago (Cosine Similarity on 64-D GraphSAGE Embeddings)")
            similar_table = gr.DataFrame(label="Top 10 Cosine Similar Incidents (Self Excluded)")

            gnn_btn.click(
                fn=get_gnn_intelligence,
                inputs=[gnn_node_input],
                outputs=[gnn_status, emb_table, similar_table]
            )

        # ----------------------------------------------------------------------
        # TAB 4: CRIME LINK ANALYSIS
        # ----------------------------------------------------------------------
        with gr.TabItem("🔗 4. Crime Link Analysis"):
            gr.Markdown("### 🔗 Multi-Factor Candidate Crime Link Prediction")
            gr.Markdown("""
            **Scoring Formula**:
            $$\\text{Link Score} = 0.35 \\cdot S_{\\text{GNN}} + 0.25 \\cdot S_{\\text{Spatial}} + 0.15 \\cdot S_{\\text{Temporal}} + 0.15 \\cdot S_{\\text{Type}} + 0.10 \\cdot S_{\\text{District}}$$
            *Disclaimer: Candidate incident links represent spatial-temporal and operational correlations. They do not constitute legal proof or causal identity.*
            """)

            with gr.Row():
                with gr.Column():
                    link_node_input = gr.Number(value=587, label="Source Node ID (or -1 for all)", precision=0)
                    score_slider = gr.Slider(minimum=0.40, maximum=0.98, value=0.60, step=0.02, label="Minimum Candidate Link Score Threshold")
                    link_btn = gr.Button("🔗 Query Candidate Links", variant="primary")
                    link_status = gr.Textbox(label="Link Search Status", interactive=False)

            link_results_table = gr.DataFrame(label="Candidate Linked Incidents")

            link_btn.click(
                fn=query_candidate_links,
                inputs=[link_node_input, score_slider],
                outputs=[link_status, link_results_table]
            )

        # ----------------------------------------------------------------------
        # TAB 5: CRIME MAP
        # ----------------------------------------------------------------------
        with gr.TabItem("🗺️ 5. Geospatial Crime Map"):
            gr.Markdown("### 🗺️ Interactive Chicago Incident Map (Offline-Compatible)")
            gr.Markdown("Inspect geospatial patterns, hotspot clusters, and filter by crime category or police district.")

            with gr.Row():
                crime_filter = gr.Dropdown(choices=["ALL"] + CLASS_NAMES[:15], value="ALL", label="Filter by Crime Category")
                district_filter = gr.Dropdown(choices=["ALL"] + [str(i) for i in sorted(dashboard_nodes_df['district'].unique()) if i > 0], value="ALL", label="Filter by Police District")
                map_btn = gr.Button("🗺️ Update Map View", variant="primary")

            map_status = gr.Textbox(label="Map Render Info", interactive=False)
            map_html_view = gr.HTML()

            map_btn.click(
                fn=render_crime_map,
                inputs=[crime_filter, district_filter],
                outputs=[map_html_view, map_status]
            )

        # ----------------------------------------------------------------------
        # TAB 6: NETWORK ANALYTICS
        # ----------------------------------------------------------------------
        with gr.TabItem("🕸️ 6. Network Analytics & Communities"):
            gr.Markdown("### 🕸️ Graph Topology, Centrality Distributions & Community Structure")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📊 Incident Degree Distribution")
                    gr.Image(os.path.join(FIGURES_DIR, "07_network_degree_distribution.png"), label="Degree Distribution")
                with gr.Column():
                    gr.Markdown("#### 👥 Top 15 Louvain Communities by Size")
                    gr.Image(os.path.join(FIGURES_DIR, "08_community_size_chart.png"), label="Community Sizes")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📈 Network Centrality Statistics (Top Nodes by PageRank)")
                    top_cent = dashboard_centrality_df.sort_values("pagerank", ascending=False).head(15)
                    gr.DataFrame(top_cent, label="PageRank Centrality Leaders")
                with gr.Column():
                    gr.Markdown("#### 🏘️ Top Community Partitions Summary")
                    gr.DataFrame(dashboard_comm_stats.head(15), label="Community Statistics")

        # ----------------------------------------------------------------------
        # TAB 7: MODEL PERFORMANCE & BASELINE
        # ----------------------------------------------------------------------
        with gr.TabItem("📈 7. Model Performance & Baseline"):
            gr.Markdown("### ⚖️ GraphSAGE vs. Non-Graph MLP Baseline Comparison (Real Test Metrics)")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📊 Test Metric Evaluation Table")
                    gr.DataFrame(dashboard_metrics_df, label="Evaluation Metrics (Test Set)")
                with gr.Column():
                    gr.Markdown("#### 📊 Performance Comparison Chart")
                    gr.Image(os.path.join(FIGURES_DIR, "06_graphsage_vs_mlp_comparison.png"), label="Model Comparison")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎯 GraphSAGE Confusion Matrix (Top 15 Classes)")
                    gr.Image(os.path.join(FIGURES_DIR, "05_confusion_matrix.png"), label="Confusion Matrix")
                with gr.Column():
                    gr.Markdown("#### 📋 Detailed Per-Class Classification Report (Excerpt)")
                    gr.DataFrame(dashboard_class_report.head(12), label="Classification Report")

        # ----------------------------------------------------------------------
        # TAB 8: ABOUT & METHODOLOGY
        # ----------------------------------------------------------------------
        with gr.TabItem("ℹ️ 8. About & Methodology"):
            gr.Markdown("""
            ### 📖 Chicago Crime Intelligence System — Technical & Research Documentation

            #### 1. Architecture Pipeline
            ```
            Incident Features (X: 40 dims) 
               │
               ▼
            SAGEConv Layer 1 (40 → 128, aggr=mean) + ReLU + Dropout(0.3)
               │
               ▼
            SAGEConv Layer 2 (128 → 64, aggr=mean) ────► 64-D Latent Crime Embedding (Cosine Similarity)
               │
               ▼
            Linear Classifier (64 → 31) ────► 31-Class Logits & Probabilities
            ```

            #### 2. Feature Vector Design (Exactly 40 Features)
            - **Temporal Features (11)**: Normalized Year, Month, Day, Hour, Day of Week, Quarter, Is Weekend, $\\sin/\\cos$ harmonics for hour and month.
            - **Spatial Features (4)**: Standardized Latitude, Longitude, State Plane X, State Plane Y.
            - **Contextual & Location (25)**: Binary Arrest, Domestic flags, frequency-encoded District, Ward, Community Area, plus 20 top location indicators (Street, Residence, Apartment, Sidewalk, etc.).

            #### 3. Spatial-Temporal Edge Construction
            - Incident proximity is parameterized with Haversine distance $d \\le 1.5\\text{ km}$ via a BallTree index, temporal constraint $\\Delta t \\le 7\\text{ days}$, and same police district constraint.

            #### 4. Candidate Crime Link Formula
            $$\\text{Link Score} = 0.35 \\cdot S_{\\text{GNN}} + 0.25 \\cdot S_{\\text{Spatial}} + 0.15 \\cdot S_{\\text{Temporal}} + 0.15 \\cdot S_{\\text{Type}} + 0.10 \\cdot S_{\\text{District}}$$

            #### 5. Academic & Ethical Disclaimer
            *This analytics system is designed strictly for research, retrospective analysis, and educational demonstration. The candidate link predictions and similarity scores do not represent legal evidence, proof of guilt, or criminal responsibility. Correlation between reported historical incidents does not establish causation.*
            """)

    # Initial trigger on load
    demo.load(
        fn=analyze_crime_node,
        inputs=[node_input],
        outputs=[status_output, meta_table, pred_class_out, pred_conf_out, probs_table]
    )
    demo.load(
        fn=render_crime_map,
        inputs=[crime_filter, district_filter],
        outputs=[map_html_view, map_status]
    )

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LAUNCHING LOCAL-ONLY GRADIO DASHBOARD")
    print("URL: http://127.0.0.1:7860 (Local Only, share=False)")
    print("=" * 60 + "\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CUSTOM_CSS
    )
