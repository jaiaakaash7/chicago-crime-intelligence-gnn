"""
build_academic_deck.py
Builds the exact 10-slide academic presentation: Chicago_Crime_Intelligence_System_10_Slides.pptx
Adheres strictly to all user constraints, formatting rules, zero-fabrication metrics, and academic design.
"""

import os
import shutil
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# Professional Color Palette
BG_COLOR = RGBColor(11, 19, 43)          # Dark Navy #0B132B
CARD_BG = RGBColor(30, 41, 59)          # Dark Slate #1E293B
CARD_BORDER = RGBColor(51, 65, 85)      # Slate 700 #334155
ACCENT_BLUE = RGBColor(56, 189, 248)    # Sky 400 #38BDF8
ACCENT_ORANGE = RGBColor(251, 146, 60)  # Orange 400 #FB923C
ACCENT_MINT = RGBColor(52, 211, 153)    # Mint 400 #34D399
TEXT_WHITE = RGBColor(248, 250, 252)    # Slate 50 #F8FAFC
TEXT_MUTED = RGBColor(148, 163, 184)    # Slate 400 #94A3B8
TEXT_LIGHT = RGBColor(203, 213, 225)    # Slate 300 #CBD5E1

def set_slide_background(slide):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = BG_COLOR

def add_header(slide, title_text, category_text="CHICAGO CRIME INTELLIGENCE SYSTEM | ACADEMIC PROJECT PRESENTATION"):
    # Category Tag
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.73), Inches(0.3))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(9.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_BLUE
    p_cat.font.name = "Segoe UI"

    # Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.68), Inches(11.73), Inches(0.55))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(21)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE
    p_title.font.name = "Segoe UI"

    # Subtle divider rule
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.30), Inches(11.73), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = CARD_BORDER
    line.line.color.rgb = CARD_BORDER

def add_footer(slide, slide_num, total_slides=10):
    foot_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.02), Inches(11.73), Inches(0.3))
    tf = foot_box.text_frame
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = f"Chicago Crime Intelligence System  •  A. Jai Aakaash (23MIA1092) & Tharunika Ravikumar (23MIA1090)  •  Slide {slide_num} of {total_slides}"
    p.font.size = Pt(8.5)
    p.font.color.rgb = TEXT_MUTED
    p.font.name = "Segoe UI"

def create_card(slide, left, top, width, height, title=None, subtitle=None):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1)

    if title:
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.16), width - Inches(0.4), Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12.5)
        p.font.bold = True
        p.font.color.rgb = ACCENT_BLUE
        p.font.name = "Segoe UI"

        if subtitle:
            p2 = tf.add_paragraph()
            p2.text = subtitle
            p2.font.size = Pt(9)
            p2.font.color.rgb = TEXT_MUTED
            p2.font.name = "Segoe UI"

    return card

def add_bullet(tf, bold_prefix, text, size=10.5, color=TEXT_LIGHT, space_after=5):
    p = tf.add_paragraph()
    p.space_after = Pt(space_after)
    p.font.size = Pt(size)
    p.font.name = "Segoe UI"

    r_bullet = p.add_run()
    r_bullet.text = "• "
    r_bullet.font.bold = True
    r_bullet.font.color.rgb = ACCENT_BLUE

    if bold_prefix:
        r_prefix = p.add_run()
        r_prefix.text = bold_prefix + ": "
        r_prefix.font.bold = True
        r_prefix.font.color.rgb = TEXT_WHITE

    r_text = p.add_run()
    r_text.text = text
    r_text.font.color.rgb = color

def add_kpi_card(slide, left, top, width, height, value, label, subtext="", val_color=ACCENT_BLUE):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1)

    tb = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.12), width - Inches(0.3), height - Inches(0.24))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_val = tf.paragraphs[0]
    p_val.text = str(value)
    p_val.font.size = Pt(20)
    p_val.font.bold = True
    p_val.font.color.rgb = val_color
    p_val.font.name = "Segoe UI"

    p_lbl = tf.add_paragraph()
    p_lbl.text = label
    p_lbl.font.size = Pt(9.5)
    p_lbl.font.bold = True
    p_lbl.font.color.rgb = TEXT_WHITE
    p_lbl.font.name = "Segoe UI"

    if subtext:
        p_sub = tf.add_paragraph()
        p_sub.text = subtext
        p_sub.font.size = Pt(8)
        p_sub.font.color.rgb = TEXT_MUTED
        p_sub.font.name = "Segoe UI"

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1 — TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Accent decorative glow bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.1), Inches(0.12), Inches(4.9))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT_BLUE
    bar.line.fill.background()

    # Category Pill
    tb_badge = s1.shapes.add_textbox(Inches(1.1), Inches(1.1), Inches(11.4), Inches(0.35))
    tf_b = tb_badge.text_frame
    p_b = tf_b.paragraphs[0]
    p_b.text = "ACADEMIC PROJECT PRESENTATION  |  COLLEGE REVIEW & VIVA EVALUATION"
    p_b.font.size = Pt(10.5)
    p_b.font.bold = True
    p_b.font.color.rgb = ACCENT_BLUE
    p_b.font.name = "Segoe UI"

    # Main Title
    tb_title = s1.shapes.add_textbox(Inches(1.1), Inches(1.55), Inches(11.4), Inches(1.7))
    tf_t = tb_title.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = "CHICAGO CRIME INTELLIGENCE SYSTEM"
    p_t.font.size = Pt(34)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_WHITE
    p_t.font.name = "Segoe UI"

    p_sub = tf_t.add_paragraph()
    p_sub.text = "GraphSAGE-Based Crime Classification, Candidate Link Prediction and Network Analytics"
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = ACCENT_ORANGE
    p_sub.font.name = "Segoe UI"
    p_sub.space_before = Pt(6)

    # Authors Card
    c_auth = create_card(s1, Inches(1.1), Inches(3.4), Inches(11.4), Inches(1.2))
    tb_auth = s1.shapes.add_textbox(Inches(1.3), Inches(3.55), Inches(11.0), Inches(0.9))
    tf_a = tb_auth.text_frame
    tf_a.word_wrap = True
    p_a1 = tf_a.paragraphs[0]
    p_a1.text = "Project Investigators:"
    p_a1.font.size = Pt(10)
    p_a1.font.bold = True
    p_a1.font.color.rgb = ACCENT_BLUE
    p_a1.font.name = "Segoe UI"

    p_a2 = tf_a.add_paragraph()
    p_a2.text = "• A. JAI AAKAASH  –  23MIA1092        • THARUNIKA RAVIKUMAR  –  23MIA1090"
    p_a2.font.size = Pt(13)
    p_a2.font.bold = True
    p_a2.font.color.rgb = TEXT_WHITE
    p_a2.font.name = "Segoe UI"
    p_a2.space_before = Pt(4)

    # Visual Pipeline Flow
    flow_img = "presentation_assets/slide1_pipeline_flow.png"
    if os.path.exists(flow_img):
        s1.shapes.add_picture(flow_img, Inches(1.1), Inches(4.85), Inches(11.4), Inches(1.5))

    add_footer(s1, 1)

    # =========================================================================
    # SLIDE 2 — PROBLEM & OBJECTIVES
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Problem & Objectives")
    add_footer(s2, 2)

    # Left: Problem & Approach (Tabular vs Graph)
    create_card(s2, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.4),
                "Problem & Graph-Based Approach", "From Isolated Table Rows to Relational Networks")
    tb_p = s2.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.5))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True
    add_bullet(tf_p, "The Problem", "Traditional crime analysis often treats every incident as an independent table row, ignoring inherent spatial and temporal connections.")
    add_bullet(tf_p, "Incident Relationships", "Real-world crimes share latent relationships based on location, time, police district, and crime characteristics.")
    add_bullet(tf_p, "Our Approach", "Represent crime incidents as an attributed graph where each incident = Node, and spatial-temporal relationships = Edges.")
    add_bullet(tf_p, "GraphSAGE Mechanism", "Uses neighborhood message passing to aggregate information from connected neighboring incidents to learn better representations.")

    # Visual comparison box inside left
    comp_img = "presentation_assets/slide2_tabular_vs_graph.png"
    if os.path.exists(comp_img):
        s2.shapes.add_picture(comp_img, Inches(1.0), Inches(4.4), Inches(5.2), Inches(2.25))

    # Right: Objectives Card
    create_card(s2, Inches(6.6), Inches(1.45), Inches(5.93), Inches(5.4),
                "Project Objectives", "Concrete Milestones Executed in Pipeline")
    tb_o = s2.shapes.add_textbox(Inches(6.8), Inches(2.1), Inches(5.53), Inches(4.5))
    tf_o = tb_o.text_frame
    tf_o.word_wrap = True
    add_bullet(tf_o, "Data Pipeline", "Process and clean Chicago crime data; select a reproducible 30,000-incident sample.")
    add_bullet(tf_o, "Feature Engineering", "Engineer exactly 40 numerical features (temporal, spatial, contextual, location typology).")
    add_bullet(tf_o, "Graph Construction", "Construct spatial-temporal crime graph using distance, time window, and district rules.")
    add_bullet(tf_o, "Model Training", "Train a 2-layer GraphSAGE classifier and train an identical MLP baseline for fair comparison.")
    add_bullet(tf_o, "Representation Learning", "Generate 64-dimensional embeddings; query similar incidents using cosine similarity.")
    add_bullet(tf_o, "Link Prediction", "Generate multi-factor candidate incident links (71,915 unique pairs scored).")
    add_bullet(tf_o, "Network Analytics", "Analyze graph degree centrality, connected components, and 272 Louvain communities.")
    add_bullet(tf_o, "Interactive UI", "Develop an interactive local dashboard running on 127.0.0.1:7860.")

    # =========================================================================
    # SLIDE 3 — DATASET & FEATURE ENGINEERING
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Dataset & Feature Engineering")
    add_footer(s3, 3)

    # Top 4 KPI Cards
    add_kpi_card(s3, Inches(0.8), Inches(1.45), Inches(2.75), Inches(1.05), "7.8M+", "Raw Chicago Records", "City Open Data Portal", ACCENT_BLUE)
    add_kpi_card(s3, Inches(3.78), Inches(1.45), Inches(2.75), Inches(1.05), "30,000", "Experimental Incidents", "Reproducible Sample (Seed 42)", ACCENT_MINT)
    add_kpi_card(s3, Inches(6.76), Inches(1.45), Inches(2.75), Inches(1.05), "40", "Engineered Features", "11 Temp + 4 Spat + 25 Context", ACCENT_ORANGE)
    add_kpi_card(s3, Inches(9.74), Inches(1.45), Inches(2.79), Inches(1.05), "30", "Target Crime Classes", "Covering July 2006 to March 2023", TEXT_WHITE)

    # Left Column: Splits and Feature Breakdown
    create_card(s3, Inches(0.8), Inches(2.65), Inches(5.6), Inches(4.2),
                "Dataset Splits & Feature Breakdown", "Strict Experimental Protocol (No Model Trained on 7.8M+)")
    tb_ds = s3.shapes.add_textbox(Inches(1.0), Inches(3.25), Inches(5.2), Inches(3.4))
    tf_ds = tb_ds.text_frame
    tf_ds.word_wrap = True
    add_bullet(tf_ds, "Sampling Scope", "Raw Chicago dataset has 7.8M+ records. After cleaning and geographic validation, an experimental sample of 30,000 incidents was drawn (July 2006 – March 2023).")
    add_bullet(tf_ds, "Data Splits", "Training: 20,994 incidents (~70%)\nValidation: 4,500 incidents (15%)\nTesting: 4,500 incidents (15%)")
    add_bullet(tf_ds, "Feature Breakdown (40 Numerical)", "• Temporal = 11 (year, month, day, hour, day-of-week, quarter, weekend, cyclical sin/cos)\n• Spatial = 4 (standardized lat/lon, X/Y State Plane)\n• Contextual = 5 (arrest, domestic, district, ward, area)\n• Location Typology = 20 (one-hot location categories)")

    # Right Column: Crime Category Chart
    create_card(s3, Inches(6.6), Inches(2.65), Inches(5.93), Inches(4.2),
                "Crime Category Distribution", "Pareto Breakdown of 30 Incident Categories")
    cat_img = "figures/02_top_crime_categories.png"
    if os.path.exists(cat_img):
        s3.shapes.add_picture(cat_img, Inches(6.8), Inches(3.3), Inches(5.53), Inches(3.35))

    # =========================================================================
    # SLIDE 4 — CRIME GRAPH CONSTRUCTION
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Crime Graph Construction")
    add_footer(s4, 4)

    # Left Card: Adjacency Criteria & Topology
    create_card(s4, Inches(0.8), Inches(1.45), Inches(5.4), Inches(5.4),
                "Spatial-Temporal Edge Formation", "Connection Criteria & Network Topology")
    tb_gc = s4.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.0), Inches(4.5))
    tf_gc = tb_gc.text_frame
    tf_gc.word_wrap = True
    add_bullet(tf_gc, "Graph Formulation", "Each crime incident is modeled as a Graph Node with 40 numerical features.")
    add_bullet(tf_gc, "Tri-Criteria Connection Rule", "A graph edge is established if and only if:\n1. Spatial distance <= 1.5 km (BallTree lookup)\n2. Temporal difference <= 7 days (168 hours)\n3. Same police administrative district")
    add_bullet(tf_gc, "Verified Graph Statistics", "• 30,000 Nodes\n• 189,474 Directed Edges (94,737 Undirected Pairs)\n• Average Degree: 6.32 (Max Degree: 60)\n• 152 Connected Components\n• 5 Isolated Nodes (99.98% Connected)\n• 0 Self-Loops")
    add_bullet(tf_gc, "Scientific Note", "Graph communities represent structurally connected incident groups. They are NOT confirmed crime syndicates.")

    # Right Column: Diagram
    create_card(s4, Inches(6.45), Inches(1.45), Inches(6.08), Inches(5.4),
                "Multiplex Graph Construction Architecture", "Visualizing Incident-to-Edge Transformation")
    gc_diag = "presentation_assets/slide4_graph_construction.png"
    if os.path.exists(gc_diag):
        s4.shapes.add_picture(gc_diag, Inches(6.6), Inches(2.15), Inches(5.78), Inches(4.5))

    # =========================================================================
    # SLIDE 5 — GRAPHSAGE ARCHITECTURE
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "GraphSAGE Architecture")
    add_footer(s5, 5)

    # Left Card: GraphSAGE Details
    create_card(s5, Inches(0.8), Inches(1.45), Inches(5.4), Inches(5.4),
                "Neighborhood Aggregation Pipeline", "Inductive Graph Representation Learning")
    tb_sage = s5.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.0), Inches(4.5))
    tf_sage = tb_sage.text_frame
    tf_sage.word_wrap = True
    add_bullet(tf_sage, "Model Architecture", "40 Features → SAGEConv → 128 Hidden → LayerNorm + ReLU + Dropout(0.2) → SAGEConv → 64-Dimensional Embedding → 30-Class Classifier Head.")
    add_bullet(tf_sage, "Mean Aggregator", "Computes the element-wise mean of neighbor feature vectors, combining the node's own features with local context.")
    add_bullet(tf_sage, "Inductive Representation", "Node's own features + Neighbor information → GraphSAGE → Learned 64-D continuous representation.")
    add_bullet(tf_sage, "Baseline MLP Model", "Identical dimensions: 40 → 128 → 64 → 30.\nReceives the exact same 40 features but uses NO graph adjacency and NO neighborhood aggregation.")

    # Right Column: Architecture Diagram
    create_card(s5, Inches(6.45), Inches(1.45), Inches(6.08), Inches(5.4),
                "Neural Network Structural Diagram", "GraphSAGE Message Passing vs Baseline MLP")
    arch_diag = "presentation_assets/slide5_graphsage_arch.png"
    if os.path.exists(arch_diag):
        s5.shapes.add_picture(arch_diag, Inches(6.6), Inches(2.15), Inches(5.78), Inches(4.5))

    # =========================================================================
    # SLIDE 6 — MODEL EVALUATION (GRAPHSAGE VS MLP)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "GraphSAGE vs MLP")
    add_footer(s6, 6)

    # Left Column: Verified Results & Highlights
    create_card(s6, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.4),
                "Empirical Test Benchmark (4,500 Nodes)", "Rigorous Comparison Across Verified Metrics")
    tb_ev = s6.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.2), Inches(4.5))
    tf_ev = tb_ev.text_frame
    tf_ev.word_wrap = True

    add_bullet(tf_ev, "GraphSAGE Top-1 Accuracy", "23.64% (improved from 15.11%, 7.1x higher than random chance).")
    add_bullet(tf_ev, "Top-3 & Top-5 Accuracy", "Top-3: 40.96% | Top-5: 53.27% (operational multi-rank dispatch accuracy).")
    add_bullet(tf_ev, "Weighted F1 Score", "0.1747 vs MLP 0.1566 (+11.5% relative advantage for GraphSAGE).")
    add_bullet(tf_ev, "Macro F1 Score", "0.0512 vs MLP 0.0361 (+41.8% relative advantage for GraphSAGE).")
    add_bullet(tf_ev, "Weighted Precision", "0.1996 vs MLP 0.1401 (+42.5% advantage for GraphSAGE).")
    add_bullet(tf_ev, "Common Crime Recall", "Theft Recall: 68.67% (F1: 0.4196) | Battery Recall: 23.47% (F1: 0.2666).")
    add_bullet(tf_ev, "Task Context", "Across 30 classes, random guess is 3.33%. GraphSAGE achieves 23.64% Top-1 and 53.27% Top-5.")

    # Right Column: Chart
    create_card(s6, Inches(6.6), Inches(1.45), Inches(5.93), Inches(5.4),
                "Benchmark Comparison Chart", "Evaluated on Held-Out Test Data")
    eval_img = "presentation_assets/slide6_eval_chart.png"
    if os.path.exists(eval_img):
        s6.shapes.add_picture(eval_img, Inches(6.8), Inches(2.15), Inches(5.53), Inches(4.5))

    # =========================================================================
    # SLIDE 7 — 64-D EMBEDDINGS & CRIME SIMILARITY
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "64-Dimensional Crime Embeddings & Similarity")
    add_footer(s7, 7)

    # Left Column: Concept & Demonstration
    create_card(s7, Inches(0.8), Inches(1.45), Inches(5.4), Inches(5.4),
                "Representation Learning Workflow", "Incident Embeddings & Cosine Similarity")
    tb_sim = s7.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.0), Inches(4.5))
    tf_sim = tb_sim.text_frame
    tf_sim.word_wrap = True
    add_bullet(tf_sim, "Workflow", "Crime Incident → GraphSAGE → 64-Dimensional Embedding → Cosine Similarity → Similar Crime Incidents.")
    add_bullet(tf_sim, "Demonstration Incident (Node 587)", "• Case: HZ102757\n• Crime: ASSAULT (AGGRAVATED: HANDGUN)\n• Police District: 9\n• Date: 2016-01-03 15:36:00")
    add_bullet(tf_sim, "Top Similarity Incidents", "1. Node 7106: THEFT (Dist 9) | Sim = 0.9963\n2. Node 4394: WEAPONS VIOLATION (Dist 9) | Sim = 0.9950\n3. Node 11356: MOTOR VEHICLE THEFT (Dist 8) | Sim = 0.9949")
    add_bullet(tf_sim, "Mathematical Meaning", "Cosine similarity measures mathematical closeness in the learned 64-D representation space.")
    add_bullet(tf_sim, "Crucial Guardrail", "Similarity does NOT establish causation, offender identity, or a confirmed relationship between incidents.")

    # Right Column: Visual Diagram
    create_card(s7, Inches(6.45), Inches(1.45), Inches(6.08), Inches(5.4),
                "Visual Case Study: Node 587", "Demonstrating Latent Space Incident Retrieval")
    sim_diag = "presentation_assets/slide7_similarity_node587.png"
    if os.path.exists(sim_diag):
        s7.shapes.add_picture(sim_diag, Inches(6.6), Inches(2.15), Inches(5.78), Inches(4.5))

    # =========================================================================
    # SLIDE 8 — MULTI-FACTOR CANDIDATE LINK ANALYSIS
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Multi-Factor Candidate Crime Link Analysis")
    add_footer(s8, 8)

    # Left Column: Formula & Logic
    create_card(s8, Inches(0.8), Inches(1.45), Inches(5.4), Inches(5.4),
                "Candidate Link Scoring Framework", "Core Project Forensic Analytics Feature")
    tb_lk = s8.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.0), Inches(4.5))
    tf_lk = tb_lk.text_frame
    tf_lk.word_wrap = True
    add_bullet(tf_lk, "Weighted Formula", "Candidate Link Score =\n• 35% GNN Embedding Similarity\n• 25% Spatial Proximity (Haversine decay)\n• 15% Temporal Proximity (7-day decay)\n• 15% Crime-Type Match\n• 10% Police District Match")
    add_bullet(tf_lk, "Scale of Link Catalog", "71,915 Unique Candidate Incident Pairs evaluated and indexed.")
    add_bullet(tf_lk, "Purpose", "The system combines multiple similarity factors to identify incidents that are worth further comparison.")
    add_bullet(tf_lk, "Important Guardrail", "Candidate links are analytical relationships, NOT confirmed criminal relationships.")
    add_bullet(tf_lk, "Strict Scope", "Does NOT claim serial offender detection, cold case solving, or getaway vehicle identification.")

    # Right Column: Formula Diagram
    create_card(s8, Inches(6.45), Inches(1.45), Inches(6.08), Inches(5.4),
                "Multi-Factor Scoring Architecture", "Visualizing Composite Weight Flow")
    lk_diag = "presentation_assets/slide8_link_formula.png"
    if os.path.exists(lk_diag):
        s8.shapes.add_picture(lk_diag, Inches(6.6), Inches(2.15), Inches(5.78), Inches(4.5))

    # =========================================================================
    # SLIDE 9 — NETWORK ANALYTICS & DASHBOARD
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Network Analytics & Interactive Dashboard")
    add_footer(s9, 9)

    # Left Column: Network Analytics & Dashboard Flow
    create_card(s9, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.4),
                "Network Properties & Investigation UI", "Localhost Deployment (127.0.0.1:7860)")
    tb_net = s9.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.2), Inches(4.5))
    tf_net = tb_net.text_frame
    tf_net.word_wrap = True
    add_bullet(tf_net, "Verified Network Results", "• 30,000 Nodes\n• 189,474 Directed Edges\n• 152 Connected Components\n• 272 Graph Communities identified via Louvain modularity")
    add_bullet(tf_net, "Network Metrics", "Degree distribution, degree centrality, PageRank, connected components, and community structure.")
    add_bullet(tf_net, "Community Clarification", "Graph communities represent structurally connected incident groups. They are NOT confirmed criminal organizations.")
    add_bullet(tf_net, "Dashboard Architecture", "Runs strictly locally at http://127.0.0.1:7860 (no cloud deployment, ensuring local data security).")
    add_bullet(tf_net, "Investigation Flow", "Node Analysis → Embeddings → Similarity → Candidate Links → Network Analytics.")

    # Right Column: Dashboard Screenshot
    create_card(s9, Inches(6.6), Inches(1.45), Inches(5.93), Inches(5.4),
                "Interactive Local Dashboard Interface", "Localhost Web Server (127.0.0.1:7860)")
    dash_img = "presentation_assets/final_report_figures/test_dash.png"
    if os.path.exists(dash_img):
        s9.shapes.add_picture(dash_img, Inches(6.8), Inches(2.15), Inches(5.53), Inches(4.5))

    # =========================================================================
    # SLIDE 10 — CONCLUSION & FUTURE WORK
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Conclusion & Future Work")
    add_footer(s10, 10)

    # Left Column: Conclusion & Contributions
    create_card(s10, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.4),
                "Project Conclusion & Summary", "Key Technical Contributions")
    tb_c = s10.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.2), Inches(4.5))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True

    p_quote = tf_c.add_paragraph()
    p_quote.text = '"The system converts historical crime incidents into a graph where each incident is represented as a node and spatial-temporal relationships form graph edges. GraphSAGE combines each incident\'s features with information from neighboring incidents to learn 64-dimensional representations."'
    p_quote.font.size = Pt(10)
    p_quote.font.italic = True
    p_quote.font.color.rgb = ACCENT_BLUE
    p_quote.font.name = "Segoe UI"
    p_quote.space_after = Pt(8)

    add_bullet(tf_c, "30,000 Incidents", "Modeled as an attributed graph with 189,474 edges.")
    add_bullet(tf_c, "40 Features", "Engineered across temporal harmonics, spatial coords, and context.")
    add_bullet(tf_c, "64-D Embeddings", "Generated via GraphSAGE for incident similarity queries.")
    add_bullet(tf_c, "71,915 Candidate Pairs", "Evaluated via multi-factor link scoring.")
    add_bullet(tf_c, "272 Communities", "Identified using unsupervised Louvain modularity optimization.")
    add_bullet(tf_c, "Interactive Dashboard", "Built and deployed locally at 127.0.0.1:7860.")

    # Right Column: Future Work & Final Thought
    create_card(s10, Inches(6.6), Inches(1.45), Inches(5.93), Inches(5.4),
                "Future Work & Research Outlook", "Avenues for Next-Generation GNN Systems")
    tb_fw = s10.shapes.add_textbox(Inches(6.8), Inches(2.15), Inches(5.53), Inches(4.5))
    tf_fw = tb_fw.text_frame
    tf_fw.word_wrap = True

    add_bullet(tf_fw, "Dynamic Temporal GNNs", "Incorporate continuous-time models like TGN and TGAT for streaming 911 dispatch feeds.")
    add_bullet(tf_fw, "Larger-Scale Graph Scaling", "Expand graph construction and sampling across larger slices of the 7.8M+ historical dataset.")
    add_bullet(tf_fw, "Heterogeneous Graphs", "Incorporate multiple entity types (incidents, vehicles, weapons, locations) via HeteroData.")
    add_bullet(tf_fw, "Explainable AI (XGNN)", "Integrate GNNExplainer to pinpoint critical subgraph neighbors influencing predictions.")
    add_bullet(tf_fw, "Additional Context", "Incorporate environmental and socioeconomic context where ethically and legally appropriate.")

    p_fin = tf_fw.add_paragraph()
    p_fin.text = '"Graph-based modeling provides a way to incorporate relational information into historical crime-data analysis while keeping the resulting patterns as analytical relationships rather than evidence of causation or offender identity."'
    p_fin.font.size = Pt(10)
    p_fin.font.italic = True
    p_fin.font.color.rgb = ACCENT_ORANGE
    p_fin.font.name = "Segoe UI"
    p_fin.space_before = Pt(12)

    # Save presentation to root
    out_file = "Chicago_Crime_Intelligence_System_10_Slides.pptx"
    prs.save(out_file)
    print(f"Presentation saved successfully to: {out_file}")

    # Copy to user desktop project folder if exists
    desktop_dest = r"C:\Users\jaiaa\OneDrive\Desktop\Chicago_Crime_Intelligence_GNN"
    if os.path.exists(desktop_dest):
        shutil.copy2(out_file, os.path.join(desktop_dest, out_file))
        print(f"Copied to Desktop folder: {desktop_dest}")

if __name__ == "__main__":
    build_presentation()
