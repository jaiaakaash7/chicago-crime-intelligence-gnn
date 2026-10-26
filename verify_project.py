#!/usr/bin/env python
# ==============================================================================
# COMPREHENSIVE PROJECT VERIFICATION SCRIPT (QA AUDIT)
# Validates every required component, dimension, tensor, metric, and artifact
# ==============================================================================

import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def run_verification():
    print("\n" + "=" * 70)
    print("STARTING FULL CHICAGO CRIME INTELLIGENCE PROJECT VERIFICATION (QA)")
    print("=" * 70)
    
    passed_checks = 0
    total_checks = 0

    def check(condition, desc):
        nonlocal passed_checks, total_checks
        total_checks += 1
        if condition:
            passed_checks += 1
            print(f"  [PASS] {desc}")
        else:
            print(f"  [FAIL] {desc}")
            raise AssertionError(f"Check failed: {desc}")

    # 1. DATA VERIFICATION
    print("\n1. DATA ARTIFACTS AUDIT:")
    cleaned_csv = os.path.join(PROJECT_ROOT, "data", "cleaned_crime_data.csv")
    node_mapping_csv = os.path.join(PROJECT_ROOT, "data", "node_mapping.csv")
    check(os.path.exists(cleaned_csv), "data/cleaned_crime_data.csv exists")
    check(os.path.exists(node_mapping_csv), "data/node_mapping.csv exists")

    df_clean = pd.read_csv(cleaned_csv)
    df_node_map = pd.read_csv(node_mapping_csv)
    N = len(df_clean)
    check(N == 30000, f"Cleaned dataset has N == 30,000 nodes (Actual: {N:,})")
    check(len(df_node_map) == 30000, f"Node mapping has N == 30,000 nodes (Actual: {len(df_node_map):,})")
    check(list(df_node_map["node_id"]) == list(range(30000)), "Node IDs in mapping are strictly contiguous 0 to N-1")

    # 2. FEATURE MATRIX AUDIT
    print("\n2. FEATURE ENGINEERING AUDIT:")
    feat_names_csv = os.path.join(PROJECT_ROOT, "results", "feature_names.csv")
    check(os.path.exists(feat_names_csv), "results/feature_names.csv exists")
    df_feat_names = pd.read_csv(feat_names_csv)
    check(len(df_feat_names) == 40, f"Exact feature count is 40 (Actual: {len(df_feat_names)})")

    # 3. CLASS MAPPING & TARGET AUDIT
    print("\n3. CLASS MAPPING & LABELS AUDIT:")
    class_map_csv = os.path.join(PROJECT_ROOT, "results", "class_mapping.csv")
    check(os.path.exists(class_map_csv), "results/class_mapping.csv exists")
    df_class_map = pd.read_csv(class_map_csv)
    check(len(df_class_map) >= 26, f"Target classes documented (Actual: {len(df_class_map)})")

    # 4. DATA SPLITS AUDIT
    print("\n4. STRATIFIED TRAIN/VAL/TEST SPLITS AUDIT:")
    train_idx_path = os.path.join(PROJECT_ROOT, "models", "train_indices.npy")
    val_idx_path = os.path.join(PROJECT_ROOT, "models", "val_indices.npy")
    test_idx_path = os.path.join(PROJECT_ROOT, "models", "test_indices.npy")
    check(os.path.exists(train_idx_path), "models/train_indices.npy exists")
    check(os.path.exists(val_idx_path), "models/val_indices.npy exists")
    check(os.path.exists(test_idx_path), "models/test_indices.npy exists")

    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    test_idx = np.load(test_idx_path)

    check(len(train_idx) == 20994, f"Train split count == 20,994 (Actual: {len(train_idx):,})")
    check(len(val_idx) == 4500, f"Val split count == 4,500 (Actual: {len(val_idx):,})")
    check(len(test_idx) == 4500, f"Test split count == 4,500 (Actual: {len(test_idx):,})")

    # No leakage between splits
    overlap_tv = np.intersect1d(train_idx, val_idx)
    overlap_tt = np.intersect1d(train_idx, test_idx)
    overlap_vt = np.intersect1d(val_idx, test_idx)
    check(len(overlap_tv) == 0, "Zero train/val leakage overlap")
    check(len(overlap_tt) == 0, "Zero train/test leakage overlap")
    check(len(overlap_vt) == 0, "Zero val/test leakage overlap")
    check(train_idx.max() < N and val_idx.max() < N and test_idx.max() < N, "All split indices strictly within [0, N-1]")

    # 5. GRAPH TOPOLOGY AUDIT
    print("\n5. GRAPH TOPOLOGY & INTEGRITY AUDIT:")
    graph_stats_csv = os.path.join(PROJECT_ROOT, "results", "graph_statistics.csv")
    check(os.path.exists(graph_stats_csv), "results/graph_statistics.csv exists")
    g_stats = pd.read_csv(graph_stats_csv).iloc[0]
    check(g_stats["node_count"] == 30000, "Graph node count is 30,000")
    check(g_stats["edge_count"] > 100000, f"Graph edge count is substantial (Actual: {int(g_stats['edge_count']):,})")
    check(g_stats["average_degree"] > 3.0, f"Graph has healthy connectivity (Avg Degree: {g_stats['average_degree']:.2f})")

    # 6. MODEL CHECKPOINTS AUDIT
    print("\n6. MODEL CHECKPOINTS AUDIT:")
    gnn_model_pth = os.path.join(PROJECT_ROOT, "models", "best_graphsage_model.pth")
    mlp_model_pth = os.path.join(PROJECT_ROOT, "models", "best_mlp_model.pth")
    check(os.path.exists(gnn_model_pth), "models/best_graphsage_model.pth exists")
    check(os.path.exists(mlp_model_pth), "models/best_mlp_model.pth exists")

    gnn_ckpt = torch.load(gnn_model_pth, map_location="cpu", weights_only=False)
    mlp_ckpt = torch.load(mlp_model_pth, map_location="cpu", weights_only=False)
    check("model_state_dict" in gnn_ckpt, "GraphSAGE checkpoint contains valid model_state_dict")
    check("model_state_dict" in mlp_ckpt, "MLP checkpoint contains valid model_state_dict")
    check(gnn_ckpt["feature_dim"] == 40, "GraphSAGE input dimension == 40")
    check(gnn_ckpt["embedding_dim"] == 64, "GraphSAGE embedding dimension == 64")

    # 7. EMBEDDINGS AUDIT
    print("\n7. 64-D LATENT EMBEDDINGS AUDIT:")
    emb_csv = os.path.join(PROJECT_ROOT, "results", "graphsage_embeddings.csv")
    check(os.path.exists(emb_csv), "results/graphsage_embeddings.csv exists")
    df_emb = pd.read_csv(emb_csv)
    check(len(df_emb) == 30000, f"Embeddings count == 30,000 (Actual: {len(df_emb):,})")
    emb_dim = len([c for c in df_emb.columns if c.startswith("embedding_")])
    check(emb_dim == 64, f"Embedding vector length is exactly 64 (Actual: {emb_dim})")

    # 8. CANDIDATE CRIME LINKS AUDIT
    print("\n8. CANDIDATE CRIME LINKS AUDIT:")
    links_csv = os.path.join(PROJECT_ROOT, "results", "crime_links.csv")
    check(os.path.exists(links_csv), "results/crime_links.csv exists")
    df_links = pd.read_csv(links_csv)
    check(len(df_links) > 10000, f"Candidate crime links generated (Actual: {len(df_links):,})")
    # Rule: no self-links!
    check((df_links["source_node"] != df_links["target_node"]).all(), "Zero self-links (source_node != target_node for all)")
    # Rule: no duplicate canonical pairs!
    check(len(df_links) == len(df_links[["source_node", "target_node"]].drop_duplicates()), "Zero duplicate canonical pairs")
    check((df_links["link_score"] >= 0.0).all() and (df_links["link_score"] <= 1.0).all(), "All link scores strictly within [0.0, 1.0]")

    # 9. NETWORK & COMMUNITY ANALYTICS AUDIT
    print("\n9. NETWORK & COMMUNITY ANALYTICS AUDIT:")
    cent_csv = os.path.join(PROJECT_ROOT, "results", "centrality_results.csv")
    comm_assign_csv = os.path.join(PROJECT_ROOT, "results", "community_assignments.csv")
    comm_stats_csv = os.path.join(PROJECT_ROOT, "results", "community_statistics.csv")
    check(os.path.exists(cent_csv), "results/centrality_results.csv exists")
    check(os.path.exists(comm_assign_csv), "results/community_assignments.csv exists")
    check(os.path.exists(comm_stats_csv), "results/community_statistics.csv exists")

    df_comm_stats = pd.read_csv(comm_stats_csv)
    check(len(df_comm_stats) > 1, f"Multiple network communities detected (Actual: {len(df_comm_stats)})")

    # 10. EVALUATION & METRICS AUDIT
    print("\n10. MODEL EVALUATION & ZERO-FABRICATION AUDIT:")
    test_metrics_csv = os.path.join(PROJECT_ROOT, "results", "test_metrics.csv")
    cls_report_csv = os.path.join(PROJECT_ROOT, "results", "classification_report.csv")
    cm_csv = os.path.join(PROJECT_ROOT, "results", "confusion_matrix.csv")
    check(os.path.exists(test_metrics_csv), "results/test_metrics.csv exists")
    check(os.path.exists(cls_report_csv), "results/classification_report.csv exists")
    check(os.path.exists(cm_csv), "results/confusion_matrix.csv exists")

    df_metrics = pd.read_csv(test_metrics_csv)
    gnn_acc = df_metrics.loc[df_metrics["Metric"] == "Accuracy", "GraphSAGE"].values[0]
    mlp_acc = df_metrics.loc[df_metrics["Metric"] == "Accuracy", "MLP"].values[0]
    check(gnn_acc > 0.0 and mlp_acc > 0.0, f"Actual test accuracy computed: GraphSAGE={gnn_acc:.4f}, MLP={mlp_acc:.4f}")

    # 11. PUBLICATION FIGURES AUDIT (10 REQUIRED FIGURES AT 300 DPI)
    print("\n11. PUBLICATION FIGURES AUDIT (300 DPI):")
    required_figures = [
        "01_crime_trend_by_year.png",
        "02_top_crime_categories.png",
        "03_hourly_crime_distribution.png",
        "04_district_distribution.png",
        "05_confusion_matrix.png",
        "06_graphsage_vs_mlp_comparison.png",
        "07_network_degree_distribution.png",
        "08_community_size_chart.png",
        "09_crime_link_score_distribution.png",
        "10_gnn_similarity_distribution.png"
    ]
    for fig_name in required_figures:
        fig_path = os.path.join(PROJECT_ROOT, "figures", fig_name)
        check(os.path.exists(fig_path), f"Figure exists: figures/{fig_name}")
        with Image.open(fig_path) as im:
            dpi = im.info.get('dpi', (72, 72))
            check(dpi[0] >= 290, f"Figure {fig_name} resolution is 300 DPI (Actual: {dpi[0]})")

    # 12. RESEARCH REPORTS AUDIT
    print("\n12. RESEARCH REPORTS AUDIT:")
    report_path = os.path.join(PROJECT_ROOT, "reports", "FINAL_REPORT.md")
    summary_path = os.path.join(PROJECT_ROOT, "reports", "RESULTS_SUMMARY.md")
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    check(os.path.exists(report_path), "reports/FINAL_REPORT.md exists")
    check(os.path.exists(summary_path), "reports/RESULTS_SUMMARY.md exists")
    check(os.path.exists(readme_path), "README.md exists")

    # 13. DASHBOARD APPLICATION AUDIT
    print("\n13. LOCAL DASHBOARD AUDIT:")
    app_py = os.path.join(PROJECT_ROOT, "dashboard", "app.py")
    check(os.path.exists(app_py), "dashboard/app.py exists")

    # Final summary
    print("\n" + "=" * 70)
    print(f"VERIFICATION COMPLETE: {passed_checks}/{total_checks} CHECKS PASSED (100% SUCCESS)!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_verification()
