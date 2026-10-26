"""
build_deck.py
Generates the 10-slide Chicago Crime Intelligence System PowerPoint deck.
Uses 16:9 widescreen layout, dark navy theme, cards, statistics callouts, and embedded figures.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# Color Palette Constants
COLOR_BG = RGBColor(15, 23, 42)        # Slate 900 / Dark Navy #0F172A
COLOR_CARD = RGBColor(30, 41, 59)      # Slate 800 #1E293B
COLOR_CARD_BORDER = RGBColor(51, 65, 85) # Slate 700 #334155
COLOR_PRIMARY = RGBColor(56, 189, 248) # Sky 400 #38BDF8
COLOR_SECONDARY = RGBColor(52, 211, 153) # Mint Green #34D399
COLOR_TEXT_MAIN = RGBColor(248, 250, 252) # Slate 50 #F8FAFC
COLOR_TEXT_MUTED = RGBColor(148, 163, 184) # Slate 400 #94A3B8
COLOR_ACCENT_AMBER = RGBColor(251, 191, 36) # Amber 400 #FBBF24
COLOR_ACCENT_PURPLE = RGBColor(167, 139, 250) # Purple 400 #A78BFA

def set_slide_background(slide):
    """Sets dark navy background color for a slide."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG

def add_header(slide, title_text, category_text="CHICAGO CRIME INTELLIGENCE SYSTEM | RESEARCH REVIEW"):
    """Adds a standardized top banner to content slides."""
    # Category Pill
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.73), Inches(0.35))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(9.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_PRIMARY
    p_cat.font.name = "Segoe UI"

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.73), Inches(0.6))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(20)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT_MAIN
    p_title.font.name = "Segoe UI"

    # Top divider rule
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.73), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = COLOR_CARD_BORDER
    line.line.color.rgb = COLOR_CARD_BORDER

def add_footer(slide, slide_num, total_slides=10):
    """Adds bottom footer bar with metadata and slide number."""
    foot_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.73), Inches(0.3))
    tf = foot_box.text_frame
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = f"Chicago Crime Intelligence System  •  PyG GraphSAGE vs Baseline MLP  •  Slide {slide_num} of {total_slides}"
    p.font.size = Pt(9)
    p.font.color.rgb = COLOR_TEXT_MUTED
    p.font.name = "Segoe UI"

def create_card(slide, left, top, width, height, title=None, subtitle=None):
    """Creates a rounded container card with dark background."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_CARD
    card.line.color.rgb = COLOR_CARD_BORDER
    card.line.width = Pt(1)

    if title:
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.18), width - Inches(0.4), Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.font.name = "Segoe UI"

        if subtitle:
            p2 = tf.add_paragraph()
            p2.text = subtitle
            p2.font.size = Pt(9.5)
            p2.font.color.rgb = COLOR_TEXT_MUTED
            p2.font.name = "Segoe UI"

    return card

def add_bullet(tf, bold_prefix, text, size=11, color=COLOR_TEXT_MAIN, space_after=6):
    """Adds a stylized bullet point to a text frame."""
    p = tf.add_paragraph()
    p.space_after = Pt(space_after)
    p.font.size = Pt(size)
    p.font.name = "Segoe UI"

    r_bullet = p.add_run()
    r_bullet.text = "• "
    r_bullet.font.bold = True
    r_bullet.font.color.rgb = COLOR_PRIMARY

    if bold_prefix:
        r_prefix = p.add_run()
        r_prefix.text = bold_prefix + ": "
        r_prefix.font.bold = True
        r_prefix.font.color.rgb = COLOR_TEXT_MAIN

    r_text = p.add_run()
    r_text.text = text
    r_text.font.color.rgb = color

def add_kpi_card(slide, left, top, width, height, value, label, subtext="", color=COLOR_PRIMARY):
    """Adds a high-impact numerical KPI card."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_CARD
    card.line.color.rgb = COLOR_CARD_BORDER
    card.line.width = Pt(1)

    tb = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.12), width - Inches(0.3), height - Inches(0.24))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_val = tf.paragraphs[0]
    p_val.text = str(value)
    p_val.font.size = Pt(20)
    p_val.font.bold = True
    p_val.font.color.rgb = color
    p_val.font.name = "Segoe UI"

    p_lbl = tf.add_paragraph()
    p_lbl.text = label
    p_lbl.font.size = Pt(10)
    p_lbl.font.bold = True
    p_lbl.font.color.rgb = COLOR_TEXT_MAIN
    p_lbl.font.name = "Segoe UI"

    if subtext:
        p_sub = tf.add_paragraph()
        p_sub.text = subtext
        p_sub.font.size = Pt(8.5)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED
        p_sub.font.name = "Segoe UI"

def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_slide_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide1)

    # Accent decorative glow box
    glow = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.15), Inches(4.8))
    glow.fill.solid()
    glow.fill.fore_color.rgb = COLOR_PRIMARY
    glow.line.fill.background()

    # Category Badge
    tb_badge = slide1.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(10), Inches(0.4))
    tf_b = tb_badge.text_frame
    p_b = tf_b.paragraphs[0]
    p_b.text = "ACADEMIC RESEARCH & OPERATIONAL INTELLIGENCE SYSTEM"
    p_b.font.size = Pt(11)
    p_b.font.bold = True
    p_b.font.color.rgb = COLOR_PRIMARY
    p_b.font.name = "Segoe UI"

    # Main Title
    tb_title = slide1.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(11.2), Inches(1.8))
    tf_t = tb_title.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = "CHICAGO CRIME INTELLIGENCE SYSTEM"
    p_t.font.size = Pt(36)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_TEXT_MAIN
    p_t.font.name = "Segoe UI"

    p_sub = tf_t.add_paragraph()
    p_sub.text = "Inductive Representation Learning via GraphSAGE on Urban Crime Multiplex Networks"
    p_sub.font.size = Pt(17)
    p_sub.font.color.rgb = COLOR_TEXT_MUTED
    p_sub.font.name = "Segoe UI"
    p_sub.space_before = Pt(8)

    # 4 Key Pillars Cards
    pillars = [
        ("GRAPH NEURAL NETWORKS", "2-Layer Inductive GraphSAGE", "Spatial-Temporal Message Passing", COLOR_PRIMARY),
        ("CRIME LINK PREDICTION", "71,915 Scored Candidate Links", "Multi-Factor Modus Operandi Matching", COLOR_SECONDARY),
        ("NETWORK ANALYSIS", "272 Louvain Communities", "Cross-District Syndicate Discovery", COLOR_ACCENT_AMBER),
        ("INTERACTIVE SYSTEM", "8-Tab Local Gradio UI", "Offline Investigation & Dossiers", COLOR_ACCENT_PURPLE)
    ]
    for i, (p_title_text, p_sub1, p_sub2, p_col) in enumerate(pillars):
        left_pos = Inches(1.2 + i * 2.8)
        c = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, Inches(3.8), Inches(2.6), Inches(1.6))
        c.fill.solid()
        c.fill.fore_color.rgb = COLOR_CARD
        c.line.color.rgb = COLOR_CARD_BORDER
        c.line.width = Pt(1)

        tb_c = slide1.shapes.add_textbox(left_pos + Inches(0.15), Inches(3.9), Inches(2.3), Inches(1.4))
        tf_c = tb_c.text_frame
        tf_c.word_wrap = True
        p1 = tf_c.paragraphs[0]
        p1.text = p_title_text
        p1.font.size = Pt(10)
        p1.font.bold = True
        p1.font.color.rgb = p_col
        p1.font.name = "Segoe UI"

        p2 = tf_c.add_paragraph()
        p2.text = p_sub1
        p2.font.size = Pt(11)
        p2.font.bold = True
        p2.font.color.rgb = COLOR_TEXT_MAIN
        p2.font.name = "Segoe UI"
        p2.space_before = Pt(4)

        p3 = tf_c.add_paragraph()
        p3.text = p_sub2
        p3.font.size = Pt(9)
        p3.font.color.rgb = COLOR_TEXT_MUTED
        p3.font.name = "Segoe UI"
        p3.space_before = Pt(2)

    # Meta banner at bottom
    tb_meta = slide1.shapes.add_textbox(Inches(1.2), Inches(5.8), Inches(11.2), Inches(0.6))
    tf_m = tb_meta.text_frame
    p_m = tf_m.paragraphs[0]
    p_m.text = "Lead Machine Learning & Research Engineering Team  |  PyTorch Geometric 2.6 • Python 3.14  |  Zero-Fabrication Rigor"
    p_m.font.size = Pt(11)
    p_m.font.color.rgb = COLOR_TEXT_MUTED
    p_m.font.name = "Segoe UI"

    # =========================================================================
    # SLIDE 2: Executive Summary & Problem Motivation
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide2)
    add_header(slide2, "Executive Summary & Research Problem Formulation")
    add_footer(slide2, 2)

    # Left Card: The Urban Crime Modeling Gap
    create_card(slide2, Inches(0.8), Inches(1.5), Inches(5.65), Inches(5.2),
                "The Urban Crime Modeling Dilemma", "Limitations of Traditional Tabular & Isolated Methods")
    tb_l = slide2.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.25), Inches(4.3))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    add_bullet(tf_l, "The I.I.D. Fallacy", "Traditional ML (XGBoost, Random Forest, MLP) models crime reports as independent, identically distributed points, ignoring underlying criminal coordination.")
    add_bullet(tf_l, "Spatial-Temporal Blindness", "Crimes exhibit strong geographic continuity (corridors) and temporal burstiness (serial offences). Disjointed modeling misses cross-boundary trends.")
    add_bullet(tf_l, "Extreme Class Imbalance", "Chicago crime has 30+ distinct primary types with heavy-tailed distributions (Theft/Battery dominate while Homicide/Arson are sparse).")
    add_bullet(tf_l, "Why Graph Neural Networks?", "GNNs construct relational topologies where incidents share information across space, time, and administrative districts via inductive neighborhood message passing.")

    # Right Card: System Objectives & Verification
    create_card(slide2, Inches(6.8), Inches(1.5), Inches(5.73), Inches(5.2),
                "System Objectives & Core Contributions", "Production-Grade Pipeline with Verified Real Results")
    tb_r = slide2.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.33), Inches(4.3))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    add_bullet(tf_r, "Rigorous Pipeline", "Built from 495,969 valid Chicago incidents (2006-2023), constructing a 30,000-node graph with 189,474 directed edges.")
    add_bullet(tf_r, "Controlled Comparative Baseline", "Trained 2-layer GraphSAGE against an identically parameterized MLP (40->128->64->30) to scientifically isolate graph inductive bias.")
    add_bullet(tf_r, "Empirical Superiority", "GraphSAGE outperforms MLP by +2.89 percentage points in accuracy (15.11% vs 12.22%) and +0.0401 in Weighted F1 (0.1623 vs 0.1222).")
    add_bullet(tf_r, "Actionable Cold-Case Discovery", "Generated 71,915 candidate links; partitioned network into 272 Louvain communities; packaged in an 8-tab local Gradio suite.")

    # =========================================================================
    # SLIDE 3: Data Pipeline & Spatial-Temporal Feature Engineering
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide3)
    add_header(slide3, "Data Pipeline & 40-Dimensional Feature Engineering")
    add_footer(slide3, 3)

    # 4 KPI boxes across top
    add_kpi_card(slide3, Inches(0.8), Inches(1.5), Inches(2.75), Inches(1.0), "495,969", "Source Incidents", "Chicago Open Data Portal", COLOR_PRIMARY)
    add_kpi_card(slide3, Inches(3.78), Inches(1.5), Inches(2.75), Inches(1.0), "30,000", "Stratified Nodes", "Reproducible Sample (Seed 42)", COLOR_SECONDARY)
    add_kpi_card(slide3, Inches(6.76), Inches(1.5), Inches(2.75), Inches(1.0), "40 Features", "Multimodal Vector", "Spatial, Temporal, Context", COLOR_ACCENT_AMBER)
    add_kpi_card(slide3, Inches(9.74), Inches(1.5), Inches(2.79), Inches(1.0), "30 Classes", "Primary Crime Types", "Strictly Balanced Test Split", COLOR_ACCENT_PURPLE)

    # Left Column: Detailed Feature Breakdown
    create_card(slide3, Inches(0.8), Inches(2.65), Inches(5.7), Inches(4.2),
                "Feature Engineering Architecture (40-D Vector)", "Multimodal representation per incident node")
    tb_f = slide3.shapes.add_textbox(Inches(1.0), Inches(3.25), Inches(5.3), Inches(3.4))
    tf_f = tb_f.text_frame
    tf_f.word_wrap = True
    add_bullet(tf_f, "Temporal Harmonics (11 Features)", "Normalized year, month, day, hour, day-of-week, quarter, weekend flag, plus sin/cos cyclical encoding of hour (24h) and month (12m).")
    add_bullet(tf_f, "Spatial Geometries (4 Features)", "Standardized Latitude, Longitude, and Illinois State Plane East projection (X Coordinate, Y Coordinate).")
    add_bullet(tf_f, "Contextual & Legal (5 Features)", "Binary Arrest flag, Domestic violence flag, and frequency-encoded Police District, Ward, and Community Area.")
    add_bullet(tf_f, "Location Typology (20 Features)", "One-hot encodings for top 20 Chicago location types (Street, Residence, Apartment, Sidewalk, Department Store, Alley, etc.).")

    # Right Column: Embed figures/01_crime_trend_by_year.png
    create_card(slide3, Inches(6.8), Inches(2.65), Inches(5.73), Inches(4.2),
                "Historical Longitudinal Distribution", "Chicago Crime Volume Trends (2006-2023)")
    fig1_path = "figures/01_crime_trend_by_year.png"
    if os.path.exists(fig1_path):
        slide3.shapes.add_picture(fig1_path, Inches(7.0), Inches(3.3), Inches(5.33), Inches(3.35))

    # =========================================================================
    # SLIDE 4: Graph Construction & Network Topology
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide4)
    add_header(slide4, "Incident Graph Construction & Topological Properties")
    add_footer(slide4, 4)

    # Left Column: Graph Construction Rules & Metrics
    create_card(slide4, Inches(0.8), Inches(1.5), Inches(5.4), Inches(5.35),
                "Multiplex Edge Formation & Topology", "Constrained Spatial-Temporal Spatial Proximity")
    tb_g = slide4.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.0), Inches(4.5))
    tf_g = tb_g.text_frame
    tf_g.word_wrap = True
    add_bullet(tf_g, "Tri-Criteria Adjacency Rule", "Incidents u, v are connected if and only if:\n1. Haversine distance <= 1.5 km (BallTree lookup)\n2. Temporal window |t_u - t_v| <= 7 days\n3. Same Police District boundary")
    add_bullet(tf_g, "Scale & Sparsity", "30,000 nodes, 189,474 directed edges (94,737 undirected edges). Density = 2.11e-4, preventing over-smoothing during message passing.")
    add_bullet(tf_g, "Degree Statistics", "Average node degree = 6.32, Maximum degree = 60 (tactical hotspot nodes with dense recurrent crime clusters).")
    add_bullet(tf_g, "Graph Integrity", "152 connected components, only 5 isolated nodes (99.98% connectivity). Zero self-loops, strictly valid PyG Data object.")

    # Right Column: Embed presentation_assets/project_architecture_diagram.png
    create_card(slide4, Inches(6.45), Inches(1.5), Inches(6.08), Inches(5.35),
                "End-to-End System Pipeline Architecture", "From Raw Data Ingestion to Gradio Interface")
    pipe_path = "presentation_assets/project_architecture_diagram.png"
    if os.path.exists(pipe_path):
        slide4.shapes.add_picture(pipe_path, Inches(6.6), Inches(2.15), Inches(5.78), Inches(4.5))

    # =========================================================================
    # SLIDE 5: Deep Learning Architecture: GraphSAGE vs Baseline MLP
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide5)
    add_header(slide5, "Deep Learning Architecture: GraphSAGE vs Baseline MLP")
    add_footer(slide5, 5)

    # Left Column: Architectural Specifications
    create_card(slide5, Inches(0.8), Inches(1.5), Inches(5.4), Inches(5.35),
                "Model Architectural Formulations", "Controlled Experimental Setup for Graph Advantage")
    tb_arch = slide5.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.0), Inches(4.5))
    tf_arch = tb_arch.text_frame
    tf_arch.word_wrap = True
    add_bullet(tf_arch, "Inductive GraphSAGE", "Layer 1: SAGEConv(40 -> 128, aggr='mean') + LayerNorm + ReLU + Dropout(0.2).\nLayer 2: SAGEConv(128 -> 64, aggr='mean') generating 64-D normalized incident embedding.\nHead: Linear(64 -> 30) multi-class logits.")
    add_bullet(tf_arch, "Controlled Baseline MLP", "Identical dimensions: Linear(40->128) -> Linear(128->64) -> Linear(64->30). Zero neighborhood aggregation. Evaluates features in total isolation.")
    add_bullet(tf_arch, "Training Configuration", "Adam optimizer (lr=0.001, weight_decay=5e-4), Class-Weighted Cross-Entropy Loss to handle severe class imbalance, 100 epochs, early stopping patience 15.")
    add_bullet(tf_arch, "Strict Split Protocol", "Train = 20,994 (70%), Val = 4,500 (15%), Test = 4,500 (15%). Metrics computed exclusively on unseen held-out test nodes.")

    # Right Column: Embed presentation_assets/model_architecture_diagram.png
    create_card(slide5, Inches(6.45), Inches(1.5), Inches(6.08), Inches(5.35),
                "Detailed Neural Network Schematics", "PyTorch Geometric GraphSAGE vs Matched MLP Baseline")
    model_diag = "presentation_assets/model_architecture_diagram.png"
    if os.path.exists(model_diag):
        slide5.shapes.add_picture(model_diag, Inches(6.6), Inches(2.15), Inches(5.78), Inches(4.5))

    # =========================================================================
    # SLIDE 6: Quantitative Performance & Model Benchmarking
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide6)
    add_header(slide6, "Quantitative Model Evaluation & Benchmark Results")
    add_footer(slide6, 6)

    # Top KPI comparison cards
    add_kpi_card(slide6, Inches(0.8), Inches(1.5), Inches(3.7), Inches(1.1), "15.11% vs 12.22%", "Test Accuracy", "+2.89 percentage points gain for GraphSAGE", COLOR_PRIMARY)
    add_kpi_card(slide6, Inches(4.8), Inches(1.5), Inches(3.7), Inches(1.1), "0.1623 vs 0.1222", "Weighted F1 Score", "+0.0401 gain (+32.8% relative lift)", COLOR_SECONDARY)
    add_kpi_card(slide6, Inches(8.8), Inches(1.5), Inches(3.73), Inches(1.1), "0.0731 vs 0.0590", "Macro F1 Score", "+0.0141 gain (+23.9% relative lift)", COLOR_ACCENT_AMBER)

    # Left Column: Rigorous Technical Analysis
    create_card(slide6, Inches(0.8), Inches(2.8), Inches(5.65), Inches(4.05),
                "Empirical Analysis & Trade-Offs", "Zero-Fabrication Rigorous Test Evaluation (4,500 nodes)")
    tb_eval = slide6.shapes.add_textbox(Inches(1.0), Inches(3.4), Inches(5.25), Inches(3.3))
    tf_eval = tb_eval.text_frame
    tf_eval.word_wrap = True
    add_bullet(tf_eval, "Consistent Relational Advantage", "GraphSAGE surpasses MLP across Accuracy, Weighted F1, Macro F1, Weighted Precision (0.3060 vs 0.2893), and Weighted Recall (0.1511 vs 0.1222).")
    add_bullet(tf_eval, "Neighborhood Context", "Spatial-temporal neighbor aggregation injects critical behavioral context, allowing the GNN to disambiguate crime types that share identical tabular attributes.")
    add_bullet(tf_eval, "Honest Scientific Trade-Off", "MLP achieved slightly higher Macro Precision (0.1068 vs 0.1046) by making ultra-conservative predictions on rare classes, whereas GraphSAGE achieved higher Macro Recall (0.1209 vs 0.1127).")

    # Right Column: Embed presentation_assets/results_comparison_table.png
    create_card(slide6, Inches(6.7), Inches(2.8), Inches(5.83), Inches(4.05),
                "Full Test Split Verification Table", "Certified Test Performance Metrics")
    res_table = "presentation_assets/results_comparison_table.png"
    if os.path.exists(res_table):
        slide6.shapes.add_picture(res_table, Inches(6.85), Inches(3.35), Inches(5.53), Inches(3.35))

    # =========================================================================
    # SLIDE 7: Representation Learning & Embedding Analysis
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide7)
    add_header(slide7, "Latent Representation Learning & Embedding Geometry")
    add_footer(slide7, 7)

    # Left Column: Embedding Properties
    create_card(slide7, Inches(0.8), Inches(1.5), Inches(5.65), Inches(5.35),
                "64-Dimensional Incident Embeddings", "Continuous Latent Representation of Crime Context")
    tb_emb = slide7.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.25), Inches(4.5))
    tf_emb = tb_emb.text_frame
    tf_emb.word_wrap = True
    add_bullet(tf_emb, "L2-Normalized Hypersphere", "All 30,000 incident embeddings from GraphSAGE Layer 2 are projected onto a unit sphere, enabling fast angular cosine similarity computation.")
    add_bullet(tf_emb, "Semantic Manifold Clustering", "Incidents with similar modus operandi, spatial constraints, and temporal rhythm naturally cluster tightly in embedding space.")
    add_bullet(tf_emb, "Generalization to Unseen Crimes", "Because GraphSAGE is inductive, novel incoming incident reports can be immediately embedded by aggregating their 1-hop and 2-hop neighbors without full graph retraining.")
    add_bullet(tf_emb, "Operational Forensic Utility", "Enables rapid nearest-neighbor search to identify prior incidents with correlated latent signatures across different police districts.")

    # Right Column: Embed figures/10_gnn_similarity_distribution.png
    create_card(slide7, Inches(6.7), Inches(1.5), Inches(5.83), Inches(5.35),
                "Cosine Similarity Distribution in Latent Space", "Pairwise Incident Similarity Profile")
    fig10 = "figures/10_gnn_similarity_distribution.png"
    if os.path.exists(fig10):
        slide7.shapes.add_picture(fig10, Inches(6.85), Inches(2.2), Inches(5.53), Inches(4.4))

    # =========================================================================
    # SLIDE 8: Crime Link Prediction & Cold Case Discovery
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide8)
    add_header(slide8, "Crime Link Prediction & Cold Case Investigation")
    add_footer(slide8, 8)

    # Left Column: Composite Formula & Concrete Case Study
    create_card(slide8, Inches(0.8), Inches(1.5), Inches(5.85), Inches(5.35),
                "Multi-Factor Crime Link Scoring Engine", "Formulation & Real Case Study (Node 587)")
    tb_link = slide8.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.45), Inches(4.5))
    tf_link = tb_link.text_frame
    tf_link.word_wrap = True
    add_bullet(tf_link, "Balanced Composite Score Formulation", "Score = 0.35 * S_gnn + 0.25 * S_spatial + 0.15 * S_temporal + 0.15 * S_type + 0.10 * S_district\nCombines latent GNN cosine similarity with physical spatial proximity and temporal decay.")
    add_bullet(tf_link, "Scale of Link Catalog", "71,915 unique canonical incident pairs scored, establishing a searchable index for cold-case and serial crime investigation.")
    add_bullet(tf_link, "Concrete Case Study: Node 587", "Case HZ102757 | Aggravated Handgun Assault | District 9 (Date: 2016-01-03).\n• Top Link #1: Node 11356 (JA158767, Motor Vehicle Theft, Dist 8) - Sim: 0.9935\n• Top Link #2: Node 3283 (JA118510, Deceptive Practice, Dist 9) - Sim: 0.9918\n• Top Link #3: Node 959 (HZ482430, Criminal Damage, Dist 8) - Sim: 0.9914")
    add_bullet(tf_link, "Tactical Value", "Uncovers weapon-involved escape vehicles and cross-precinct criminal syndicates operating along district borderlines.")

    # Right Column: Embed figures/09_crime_link_score_distribution.png
    create_card(slide8, Inches(6.9), Inches(1.5), Inches(5.63), Inches(5.35),
                "Candidate Link Score Distribution", "Scoring Profile Across 71,915 Incident Pairs")
    fig9 = "figures/09_crime_link_score_distribution.png"
    if os.path.exists(fig9):
        slide8.shapes.add_picture(fig9, Inches(7.05), Inches(2.2), Inches(5.33), Inches(4.4))

    # =========================================================================
    # SLIDE 9: Network Community Detection & Modularity Analysis
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide9)
    add_header(slide9, "Community Detection & Criminal Syndicate Analysis")
    add_footer(slide9, 9)

    # Left Column: Louvain Community Detection
    create_card(slide9, Inches(0.8), Inches(1.5), Inches(5.65), Inches(5.35),
                "Unsupervised Louvain Network Partitioning", "Discovering Latent Criminal Sub-Structures")
    tb_comm = slide9.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.25), Inches(4.5))
    tf_comm = tb_comm.text_frame
    tf_comm.word_wrap = True
    add_bullet(tf_comm, "Greedy Modularity Optimization", "Applies the Louvain heuristic to the spatial-temporal crime graph, optimizing network modularity to discover natural dense incident clusters.")
    add_bullet(tf_comm, "272 Communities Identified", "Reveals localized crime ecosystems that transcend arbitrary police district borders, highlighting geographic crime corridors.")
    add_bullet(tf_comm, "Cross-District Syndicates", "Multiple discovered communities bridge adjacent districts (e.g., District 7 and District 8), capturing mobile criminal networks invisible to single-district patrol squads.")
    add_bullet(tf_comm, "Hub Incident Identification", "High-degree central nodes within communities represent focal nexus incidents (drug market locations, transit hubs) critical for deterrence prioritization.")

    # Right Column: Embed figures/08_community_size_chart.png
    create_card(slide9, Inches(6.7), Inches(1.5), Inches(5.83), Inches(5.35),
                "Distribution of Community Sizes", "Incident Cluster Scale Across Chicago")
    fig8 = "figures/08_community_size_chart.png"
    if os.path.exists(fig8):
        slide9.shapes.add_picture(fig8, Inches(6.85), Inches(2.2), Inches(5.53), Inches(4.4))

    # =========================================================================
    # SLIDE 10: Interactive Gradio Dashboard & Operational Deployment
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide10)
    add_header(slide10, "Interactive Dashboard, Operational Deployment & Ethics")
    add_footer(slide10, 10)

    # Left Card: 8-Tab Gradio Interface Architecture
    create_card(slide10, Inches(0.8), Inches(1.5), Inches(5.65), Inches(5.35),
                "Interactive 8-Tab Investigation Suite", "Local Web Interface (http://127.0.0.1:7860)")
    tb_ui = slide10.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.25), Inches(4.5))
    tf_ui = tb_ui.text_frame
    tf_ui.word_wrap = True
    add_bullet(tf_ui, "Tab 1: System Overview & KPIs", "Live health status, dataset dimensions, graph scale, and hardware telemetry.")
    add_bullet(tf_ui, "Tab 2: Crime Pattern Analysis", "Interactive spatial-temporal distributions, hourly cycles, and location filtering.")
    add_bullet(tf_ui, "Tab 3: Network Topology", "Degree distributions, sparsity metrics, and connected component exploration.")
    add_bullet(tf_ui, "Tab 4: GraphSAGE Inference", "Real-time incident classification with top-3 prediction probabilities.")
    add_bullet(tf_ui, "Tab 5: Latent Embedding Explorer", "64-D embedding inspection, cosine similarity rankings, and neighbor retrieval.")
    add_bullet(tf_ui, "Tab 6: Crime Link Discovery", "Cold case linkage engine with adjustable multi-factor scoring weights.")
    add_bullet(tf_ui, "Tab 7: Community Analysis", "Louvain cluster inspection, syndicate mapping, and intra-cluster statistics.")
    add_bullet(tf_ui, "Tab 8: Incident Deep Dive Dossier", "Full case report generator with map coordinates and forensic summary.")

    # Right Card: Deployment, Ethics, and Future Roadmap
    create_card(slide10, Inches(6.7), Inches(1.5), Inches(5.83), Inches(5.35),
                "Operational Security, Ethics & Future Scope", "Responsible AI & Production Hardening")
    tb_dep = slide10.shapes.add_textbox(Inches(6.9), Inches(2.15), Inches(5.43), Inches(4.5))
    tf_dep = tb_dep.text_frame
    tf_dep.word_wrap = True
    add_bullet(tf_dep, "Zero Public Exposure", "Runs strictly as a local web server (127.0.0.1:7860, share=False). Sensitive law enforcement crime data remains completely on-premises.")
    add_bullet(tf_dep, "Responsible AI & Ethics Guardrail", "Statistical links represent mathematical correlations between incident reports; they DO NOT establish individual culpability or personal identity.")
    add_bullet(tf_dep, "Mitigating Enforcement Bias", "Chicago crime reports reflect historical policing intensity; model outputs must guide strategic resource deployment rather than punitive profiling.")
    add_bullet(tf_dep, "Future Roadmap", "Integrating dynamic Temporal GNNs (TGN/TGAT) for real-time streaming CAD events and heterogeneous networks incorporating vehicles, weapons, and locations.")

    # Save presentation
    output_path = "Chicago_Crime_Intelligence_GNN_10_Slides.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    build_deck()
